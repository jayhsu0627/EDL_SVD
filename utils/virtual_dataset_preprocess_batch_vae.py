import os
import glob
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import kornia.augmentation as K
from kornia.augmentation.container import ImageSequential
import math
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from einops import rearrange
import torch.nn as nn



import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp():
    dist.init_process_group(backend="nccl")
setup_ddp()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# def tensor_to_vae_latent(t, vae):
#     # video_length = t.shape[1]

#     # If it’s wrapped in DDP, grab the underlying module
#     net = vae.module if isinstance(vae, DDP) else vae
#     net.to(device)
#     t = t.to(device)

#     with torch.no_grad():
#         latents = net.encode(t).latent_dist.sample()
#     # latents = latents * net.config.scaling_factor
#     return latents.detach().cpu()

def tensor_to_vae_latent(t, vae):
    net = vae.module if isinstance(vae, DDP) else vae
    # t   = t.to(dtype=torch.float32, device=device)          # keep fp32 inside VAE
    t = t.to(device)
    with torch.no_grad():
        latents = net.encode(t).latent_dist.sample()
    latents = latents * net.config.scaling_factor           # ← ADD THIS BACK
    return latents.cpu()                                    # keep fp32 or fp16 later


# def latent_to_tensor(latents, vae, frames=2):

#     video_length = latents.shape[1]
#     # If it’s wrapped in DDP, grab the underlying module
#     net = vae.module if isinstance(vae, DDP) else vae
#     net.to(device)
#     latents = latents.to(device)

#     # latents = latents / vae.config.scaling_factor
#     # latents = rearrange(latents, "b f c h w -> (b f) c h w")
#     with torch.no_grad():
#         t = net.decode(latents, num_frames = frames).sample
#     del latents  # 🚀 Free `latents` after decoding
#     # t = rearrange(t, "(b f) c h w -> b f c h w", f=video_length)
#     torch.cuda.empty_cache()  # 🚀 Free GPU memory
#     return t

def latent_to_tensor(latents, vae, num_frames):
    net = vae.module if isinstance(vae, DDP) else vae
    # latents = latents.to(dtype=torch.float32, device=device)
    latents = latents / net.config.scaling_factor           # ← ADD THIS BACK
    with torch.no_grad():
        t = net.decode(latents, num_frames=num_frames).sample
    return t                                                # [-1,1] fp32

def save_array_as_image(array, filename):
    # Ensure the array has the correct data type (uint8 for images)
    if array.dtype != np.uint8:
        array = array.numpy().astype(np.uint8)
    # array = (array + 1)
    array = array.transpose(1, 2, 0)

    # Convert the array to an image using PIL
    img = Image.fromarray(array)
    
    # Save the image to the specified filename
    img.save(filename)

def save_array_as_image_depth(array, filename):
    # Ensure the array has the correct data type (uint8 for images)
    if array.dtype != np.uint8:
        array = array.numpy().astype(np.uint8)
    # array = (array + 1)
    array = array.transpose(1, 2, 0)[:,:,0]
    # Convert the array to an image using PIL
    img = Image.fromarray(array, mode="L")
    
    # Save the image to the specified filename
    img.save(filename)


# # Helper functions for image conversion
# def pil_image_to_numpy(image):
#     """
#     This is for 3D-FRONT data read-in, only
#     """
#     if image.mode == 'RGBA':
#         # Keep the alpha channel for RGBA images
#         return np.array(image)[:, :, :3]
#     elif image.mode != 'RGB':
#         # Convert other modes (like L, P, etc.) to RGB
#         image = image.convert('RGB')
#     return np.array(image)

def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    # print(image.mode)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255  # Normalize to [0,1]

    # return (images.float() / 127.5 -1).to(torch.float16)  # Normalize to [-1,1]

class SafeRandomCrop(K.AugmentationBase2D):
    def __init__(self, size=(256, 256), p=1.0, same_on_batch=False, keepdim=False):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.size = size if isinstance(size, tuple) else (size, size)
        
    def compute_transformation(self, input, params):
        return None

    def apply_transform(self, input, params=None, transform_matrix=None, mask=None):
        B_full, C, H, W = input.shape
        crop_h, crop_w = self.size
        crops = []

        # Determine cropping coordinates only once
        if mask is not None:
            B_mask = mask.shape[0]
            repeat_factor = B_full // B_mask
            mask = mask.repeat_interleave(repeat_factor, dim=0)

            # Use the **first mask** only to determine the crop (assumes temporal alignment)
            mask_i = mask[0, 0] if mask[0].ndim == 3 else mask[0]
            coords = mask_i.nonzero(as_tuple=False)

            if coords.numel() > 0:
                y_min, x_min = coords.min(dim=0).values
                y_max, x_max = coords.max(dim=0).values
                center_y = (y_min + y_max) // 2
                center_x = (x_min + x_max) // 2

                max_offset_y = max(0, min(center_y, H - crop_h - center_y))
                max_offset_x = max(0, min(center_x, W - crop_w - center_x))
                offset_y = torch.randint(-max_offset_y, max_offset_y + 1, (1,), device=input.device).item()
                offset_x = torch.randint(-max_offset_x, max_offset_x + 1, (1,), device=input.device).item()

                start_h = int(min(max(center_y + offset_y - crop_h // 2, 0), H - crop_h))
                start_w = int(min(max(center_x + offset_x - crop_w // 2, 0), W - crop_w))
            else:
                start_h = (H - crop_h) // 2
                start_w = (W - crop_w) // 2
        else:
            # Fall back to safe regions logic from the first element
            y_min, y_max, x_min, x_max = self.get_safe_regions(transform_matrix, (H, W))[0]
            available_h = y_max - y_min - crop_h + 1
            available_w = x_max - x_min - crop_w + 1

            if available_h <= 0 or available_w <= 0:
                start_h = (H - crop_h) // 2
                start_w = (W - crop_w) // 2
            else:
                start_h = y_min + torch.randint(0, available_h, (1,), device=input.device).item()
                start_w = x_min + torch.randint(0, available_w, (1,), device=input.device).item()

        # Apply the same crop to all images
        for i in range(B_full):
            crop = input[i:i+1, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
            crops.append(crop)

        return torch.cat(crops, dim=0)

        return torch.cat(crops, dim=0)

    def get_safe_regions(self, transform_matrix, image_size):
        H, W = image_size
        B = transform_matrix.shape[0]
        safe_regions = []

        for i in range(B):
            matrix = transform_matrix[i]
            angle_rad = torch.atan2(matrix[0, 1], matrix[0, 0])
            center_y, center_x = H // 2, W // 2

            corners = torch.tensor([
                [-center_x, -center_y],
                [W - center_x - 1, -center_y],
                [-center_x, H - center_y - 1],
                [W - center_x - 1, H - center_y - 1]
            ], dtype=torch.float32, device=transform_matrix.device)

            distances = torch.sqrt(corners[:, 0]**2 + corners[:, 1]**2)
            max_distance = torch.max(distances)

            abs_angle_rad = torch.abs(angle_rad)
            safe_radius = max_distance * torch.cos(abs_angle_rad) / math.sqrt(2)
            safe_size = int(safe_radius.item())

            y_min = max(0, center_y - safe_size)
            y_max = min(H, center_y + safe_size)
            x_min = max(0, center_x - safe_size)
            x_max = min(W, center_x + safe_size)

            safe_regions.append((y_min, y_max, x_min, x_max))

        return safe_regions
    
class PreprocessAndSaveDataset:
    def __init__(self, root_dir, save_dir, sample_n_frames=14, width=256, height=256, num_samples=1):
        self.root_dir = root_dir
        self.filter_list = '/fs/gamma-projects/svd_relight/sketchfab/rendering_2/filter.txt'
        # self.filter_list = '~/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/rendering/filter_list.txt'
        self.save_dir = save_dir
        self.sample_n_frames = sample_n_frames
        self.width = width
        self.height = height
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained("stabilityai/stable-video-diffusion-img2vid",
                     subfolder="vae", revision=None, variant="fp16").to(device, dtype=torch.float16)
        # self.vae = AutoencoderKLTemporalDecoder.from_pretrained("stabilityai/stable-video-diffusion-img2vid",
        #              subfolder="vae", revision=None).to(device)

        #     vae = AutoencoderKLTemporalDecoder.from_pretrained(
        # args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16")

        # if multiple GPUs are available, wrap:
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.vae = DDP(self.vae, device_ids=[local_rank], output_device=local_rank)

        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        self.num_samples = num_samples
        # List all scene directories
        self.scene_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))
        
        # == ADD THESE LINES ==
        world_size = dist.get_world_size()
        rank       = dist.get_rank()
        # keep only every world_size-th element, starting at your rank
        self.scene_dirs = self.scene_dirs[rank::world_size]
        print(f"[rank {rank}] will process {len(self.scene_dirs)} scenes")

        # Stage 1: Resize
        self.resize = K.SmallestMaxSize(self.width, p=1.0)

        # Stage 2: RandomAffine with matrix output
        self.affine = K.RandomAffine(degrees=(-15., 20.), p=1.0, same_on_batch=True)

        # Stage 3: Crop and color augmentation
        self.safe_crop = SafeRandomCrop((self.width, self.height), p=1.0, same_on_batch=True)

        self.transforms_1 = ImageSequential(
            K.RandomHorizontalFlip(p=0.8),
            same_on_batch=True
        )
        self.transforms_2 = ImageSequential(
            # K.RandomClahe(p=0.6),
            # K.RandomChannelShuffle(p=0.6),
            K.ColorJiggle(0.05, 0.05, 0.05, 0.05, p=0.6),
            same_on_batch=True
        )

        self.transforms_3 = ImageSequential(
            K.RandomGaussianIllumination(gain=(0.01, 0.15), sign=(-0.06, 0.3), p=1.0, keepdim=True),
            K.RandomLinearCornerIllumination(gain=0.3, p=1.0, sign=(-0.3, 0.8), keepdim=True),
            same_on_batch=True
        )


        # self.transforms = ImageSequential(
        #     K.SmallestMaxSize(256, p=1.0),  # First resize to maintain aspect ratio
        #     K.RandomAffine((-45., 45.), p=1.0),  # Then rotate
        #     K.RandomCrop((256, 256), p=1.0),  # Crop center part to avoid black borders
        #     K.RandomHorizontalFlip(p=0.8),
        #     K.RandomClahe(p=0.5),
        #     same_on_batch=True
        # )

        self.list = []
        try:
            with open(self.filter_list, 'r') as f:
                # Iterate through each line in the file
                for line in f:
                    # .strip() removes leading/trailing whitespace, including the newline character (\n)
                    self.list.append(line.strip())
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def sort_frames_sync(self, frame_name):
        # Extract the filename without extension
        base_name = frame_name.split('.')[0]
        
        # Check if the filename has the format with underscores
        if '_' not in base_name:
            # Find where the digits start
            i = 0
            while i < len(base_name) and not base_name[i].isdigit():
                i += 1
            
            # Extract the numeric part
            if i < len(base_name):
                digits = ''
                while i < len(base_name) and base_name[i].isdigit():
                    digits += base_name[i]
                    i += 1
                
                if digits:
                    return int(digits)

        # Default return if no valid numeric part is found
        return 9999

    def process_and_save(self):
        """Preprocess and save all dataset files as .pkl"""
        for idx, scene_path in enumerate(self.scene_dirs):
            # index_i = np.random.randint(0, self.sample_n_frames)
            index_i = 0

            lookup_scene = os.path.basename(scene_path)
            if lookup_scene in self.list:
                continue
            # if lookup_scene != '00ad8345-45e0-45b3-867d-4a3c88c2517a':
            #     continue
            print('process: ', lookup_scene)

            # Collect file lists
            image_files = sorted(glob.glob(os.path.join(scene_path, "relit*.png")))[index_i : index_i + self.sample_n_frames]
            depth_files = sorted(glob.glob(os.path.join(scene_path, "depth*.png")))[index_i : index_i + self.sample_n_frames]
            normal_files = sorted(glob.glob(os.path.join(scene_path, "normal*.png")))[index_i : index_i + self.sample_n_frames]
            alb_files = sorted(glob.glob(os.path.join(scene_path, "diffuse*.png")))[index_i : index_i + self.sample_n_frames]
            scb_files = sorted(glob.glob(os.path.join(scene_path, "mask*.png")))[index_i : index_i + self.sample_n_frames]
            rgb_files = sorted(glob.glob(os.path.join(scene_path, "colors*.png")))[index_i : index_i + self.sample_n_frames]
            
            if not (len(image_files) == len(depth_files) == len(normal_files) ==len(alb_files) ==len(scb_files) == len(rgb_files) == self.sample_n_frames):
                print('drop ===>', lookup_scene)
                continue
                
            # Load and convert images
            def load_images(file_list):
                return numpy_to_pt(np.array([pil_image_to_numpy(Image.open(f)) for f in file_list]))
        
            img_org = load_images(image_files)
            depth_org = load_images(depth_files)
            normal_org = load_images(normal_files)
            alb_org = load_images(alb_files)
            scb_org = load_images(scb_files)
            rgb_org = load_images(rgb_files)

            # for var in range(self.num_samples):
            # Extract scene name
            scene_name = os.path.basename(scene_path)
            # scene_name = 'train_' + str(var)
            save_path = os.path.join(self.save_dir, f"{scene_name}_"+str(self.width)+".pkl")
            batch_size = img_org.shape[0]

            combined = torch.cat([img_org, depth_org, normal_org, alb_org, scb_org, rgb_org], dim=0)
            combined = self.resize(combined)

            # print('resized_combined size check =>>>:', resized_combined.shape)

            # rotated_combined = self.affine(resized_combined, return_transform=True)
            # transform_matrix = self.affine.transform_matrix

            # scb_mask = rotated_combined[batch_size*4: batch_size*5]

            # # print('rotated_combined size check =>>>:', rotated_combined.shape)
            # # cropped_combined = self.safe_crop.apply_transform(rotated_combined, transform_matrix=transform_matrix, masks=scb_org)
            # cropped_combined = self.safe_crop.apply_transform(
            #                                                 rotated_combined, 
            #                                                 transform_matrix=transform_matrix, 
            #                                                 mask=scb_mask  # must be shape [B, 1, H, W]
            #                                             )

            # combined = self.transforms_1(cropped_combined)
            # combined = self.transforms_1(rotated_combined)
            img, depth, normal, alb, scb, rgb = (
                combined[:batch_size],
                combined[batch_size:batch_size*2],
                combined[batch_size*2:batch_size*3],
                combined[batch_size*3:batch_size*4],
                combined[batch_size*4:batch_size*5],
                combined[batch_size*5:],
            )
            print(img.shape, depth.shape, normal.shape, alb.shape, scb.shape, rgb.shape)
            combined = torch.cat([img, rgb], dim=0)


            img = img.to(dtype=torch.float16)
            depth = depth.to(dtype=torch.float16)
            normal = normal.to(dtype=torch.float16)
            alb = alb.to(dtype=torch.float16)
            scb = scb.to(dtype=torch.float16)
            rgb = rgb.to(dtype=torch.float16)

            print('depth dtype:   ', depth.dtype)
            data_dict = {
                "pixel_values": img*2 - 1,
                "depth_pixel_values": tensor_to_vae_latent(depth*2 - 1, self.vae),  # Convert depth to 1-channel
                "normal_pixel_values": tensor_to_vae_latent(normal*2 - 1, self.vae),
                "alb_pixel_values": tensor_to_vae_latent(alb*2 - 1, self.vae),
                "scb_pixel_values": tensor_to_vae_latent(scb*2 - 1, self.vae),  # Convert to 1-channel
                "rgb_pixel_values": rgb*2 - 1,
                "tgt_latents": tensor_to_vae_latent(rgb*2 - 1, self.vae)
            }

            # data_dict = {
            #     "pixel_values": img*2 - 1,
            #     "depth_pixel_values": depth*2 - 1,  # Convert depth to 1-channel
            #     "normal_pixel_values": normal*2 - 1,
            #     "alb_pixel_values": alb*2 - 1,
            #     "scb_pixel_values": scb[:, :, :, :]*2 - 1,  # Convert to 1-channel
            #     "rgb_pixel_values": rgb*2 - 1,
            #     "tgt_latents": rgb*2 - 1
            # }

            # [print(v.dtype) for k, v in data_dict.items()]
            # [print(v) for k, v in data_dict.items()]

            # Save as .pkl
            with open(save_path, "wb") as f:
                # pickle.dump({k: v.to(torch.float16) for k, v in data_dict.items()}, f)
                pickle.dump({k: v for k, v in data_dict.items()}, f)

            print(data_dict["pixel_values"].shape, data_dict["depth_pixel_values"].shape, data_dict["normal_pixel_values"].shape, data_dict["alb_pixel_values"].shape, data_dict["scb_pixel_values"].shape, data_dict["rgb_pixel_values"].shape, data_dict["tgt_latents"].shape)
            print(f"Processed and saved: {scene_name}")

        print("All files processed and saved.")

class SyncDataset(Dataset):
    def __init__(self, preprocessed_dir, max_frames=None, mixed_precision="fp16"):
        """
        Dataset class to load preprocessed `.pkl` files.
        
        Args:
            preprocessed_dir (str): Path to directory containing `.pkl` files.
            max_frames (int, optional): Maximum number of frames to load. 
                                        If None, loads all frames.
            mixed_precision (str, optional): Precision format (fp16, bf16, or default fp32).
        """
        self.preprocessed_dir = preprocessed_dir
        self.file_list = sorted(glob.glob(os.path.join(preprocessed_dir, "*.pkl")))
        print(self.file_list)
        self.max_frames = max_frames  # New variable to limit frame count
        self.weight_dtype = torch.float32

        if mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        print(f"Loaded {len(self.file_list)} preprocessed samples.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        base_name = os.path.basename(file_path)
        scene_name = base_name.split(".")[0]
        with open(file_path, "rb") as f:
            data_dict = pickle.load(f)

        # Clip to max_frames if specified
        if self.max_frames is not None:
            data_dict = {key: value[:self.max_frames] for key, value in data_dict.items()}

        # Convert data to correct dtype
        sample = {k: v.to(dtype=self.weight_dtype) for k, v in data_dict.items()}
        sample['scene_name'] = scene_name
        return sample


# Example usage:
if __name__ == "__main__":
    
    num = 1
    size = 128

    preprocessor = PreprocessAndSaveDataset(
        # root_dir="~/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/rendering",
        root_dir="/fs/gamma-projects/svd_relight/sketchfab/rendering_2",
        # save_dir="/fs/gamma-projects/svd_relight/sync/junk",
        save_dir="/fs/gamma-projects/svd_relight/Data",
        sample_n_frames=14,
        width=size,
        height=size,
        num_samples = num
    )
    preprocessor.process_and_save()

    # dataset = SyncDataset(preprocessed_dir="/fs/gamma-projects/svd_relight/3D_Front/train", max_frames=14)
    dataset = SyncDataset(preprocessed_dir="/fs/gamma-projects/svd_relight/Data/", max_frames=14)

    # Test loading a batch
    index_i = np.random.randint(0, num)

    sample = dataset[3]
    
    pixel_values, depth_pixel_values, normal_pixel_values, alb_pixel_values, scb_pixel_values, rgb_pixel_values = sample["pixel_values"], sample["depth_pixel_values"], sample["normal_pixel_values"], sample["alb_pixel_values"], sample["scb_pixel_values"], sample["rgb_pixel_values"]


RGBA    
    pixel_values = pixel_values.to(device)
    depth_pixel_values = depth_pixel_values.to(device)
    normal_pixel_values = normal_pixel_values.to(device)
    alb_pixel_values = alb_pixel_values.to(device)
    scb_pixel_values = scb_pixel_values.to(device)
    rgb_pixel_values = rgb_pixel_values.to(device)

    print('=== Validation ===')
    print(pixel_values.shape)
    print(depth_pixel_values.shape)
    print(normal_pixel_values.shape)
    print(alb_pixel_values.shape)
    print(scb_pixel_values.shape)
    print(rgb_pixel_values.shape)
    
    vae = AutoencoderKLTemporalDecoder.from_pretrained("stabilityai/stable-video-diffusion-img2vid",
                     subfolder="vae", revision=None, variant="fp16").to(device, dtype=torch.float16)
    # vae = AutoencoderKLTemporalDecoder.from_pretrained("stabilityai/stable-video-diffusion-img2vid",
    #                  subfolder="vae", revision=None).to(device)

    depth_pixel_values = latent_to_tensor(depth_pixel_values, vae, 14).cpu().detach()
    normal_pixel_values = latent_to_tensor(normal_pixel_values, vae, 14).cpu().detach()
    alb_pixel_values = latent_to_tensor(alb_pixel_values, vae, 14).cpu().detach()
    scb_pixel_values = latent_to_tensor(scb_pixel_values, vae, 14).cpu().detach()

    pixel_values = ((pixel_values+1)/2).cpu().detach()
    depth_pixel_values = (depth_pixel_values+1)/2
    normal_pixel_values = (normal_pixel_values+1)/2
    alb_pixel_values = (alb_pixel_values+1)/2
    scb_pixel_values = (scb_pixel_values+1)/2
    rgb_pixel_values = ((rgb_pixel_values+1)/2).cpu().detach()

    save_array_as_image(pixel_values[0]*255, "~/DiffusionMaskRelight/outputs/sync/relit_01.png")
    save_array_as_image(pixel_values[5]*255, "~/DiffusionMaskRelight/outputs/sync/relit_06.png")
    save_array_as_image(pixel_values[-1]*255, "~/DiffusionMaskRelight/outputs/sync/relit_16.png")

    save_array_as_image_depth(depth_pixel_values[0]*255, "~/DiffusionMaskRelight/outputs/sync/dep_01.png")
    save_array_as_image_depth(depth_pixel_values[5]*255, "~/DiffusionMaskRelight/outputs/sync/dep_06.png")
    save_array_as_image_depth(depth_pixel_values[-1]*255, "~/DiffusionMaskRelight/outputs/sync/dep_16.png")
    
    save_array_as_image(normal_pixel_values[0]*255, "~/DiffusionMaskRelight/outputs/sync/nrm_01.png") 
    save_array_as_image(normal_pixel_values[5]*255, "~/DiffusionMaskRelight/outputs/sync/nrm_06.png") 
    save_array_as_image(normal_pixel_values[-1]*255, "~/DiffusionMaskRelight/outputs/sync/nrm_16.png")

    save_array_as_image(alb_pixel_values[0]*255, "~/DiffusionMaskRelight/outputs/sync/alb_01.png")
    save_array_as_image(alb_pixel_values[5]*255, "~/DiffusionMaskRelight/outputs/sync/alb_06.png")
    save_array_as_image(alb_pixel_values[-1]*255, "~/DiffusionMaskRelight/outputs/sync/alb_16.png")

    save_array_as_image_depth(scb_pixel_values[0]*255, "~/DiffusionMaskRelight/outputs/sync/scb_01.png")
    save_array_as_image_depth(scb_pixel_values[5]*255, "~/DiffusionMaskRelight/outputs/sync/scb_06.png")
    save_array_as_image_depth(scb_pixel_values[-1]*255, "~/DiffusionMaskRelight/outputs/sync/scb_16.png")

    save_array_as_image(rgb_pixel_values[0]*255, "~/DiffusionMaskRelight/outputs/sync/rgb_01.png")
    save_array_as_image(rgb_pixel_values[5]*255, "~/DiffusionMaskRelight/outputs/sync/rgb_06.png")
    save_array_as_image(rgb_pixel_values[-1]*255, "~/DiffusionMaskRelight/outputs/sync/rgb_16.png")

    print('done')