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
 


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp():
    dist.init_process_group(backend="nccl")
setup_ddp()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

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


# Helper functions for image conversion
def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    print(images.shape)
    return images.float() / 255  # Normalize to [0,1]

    # return (images.float() / 127.5 -1).to(torch.float16)  # Normalize to [-1,1]
    
class PreprocessAndSaveDataset:
    def __init__(self, root_dir, save_dir, sample_n_frames=14, width=256, height=256, num_samples=1):
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.sample_n_frames = sample_n_frames
        self.width = width
        self.height = height
        # self.filter_list = '/fs/gamma-projects/svd_relight/sketchfab/rendering_2/filter.txt'
        self.filter_list = '~/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/rendering/filter_list.txt'

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

    def process_and_save(self):
        """Preprocess and save all dataset files as .pkl"""
        for idx, scene_path in enumerate(self.scene_dirs):
            # index_i = np.random.randint(0, self.sample_n_frames)
            index_i = 0

            lookup_scene = os.path.basename(scene_path)
            
            # if lookup_scene not in self.list:
            #     continue

            # if lookup_scene == 'gifs' or lookup_scene == 'failed.txt' or lookup_scene == 'filter.txt' or '_r_' in lookup_scene or '_v_' in lookup_scene:
            #     continue
            if lookup_scene == 'gifs' or lookup_scene == 'failed.txt' or lookup_scene == 'filter.txt' :
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
            
            # Load and convert images
            def load_images(file_list):
                return numpy_to_pt(np.array([pil_image_to_numpy(Image.open(f)) for f in file_list]))

            print(lookup_scene, image_files)

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

            # combined = self.transforms_1(rotated_combined)
            img, depth, normal, alb, scb, rgb = (combined[:batch_size] ), \
                                                (combined[batch_size: batch_size*2] ), \
                                                (combined[batch_size*2: batch_size*3] ), \
                                                (combined[batch_size*3: batch_size*4] ), \
                                                (combined[batch_size*4: batch_size*5] ), \
                                                (combined[batch_size*5: ] ) \

            combined = torch.cat([img, rgb], dim=0)

            data_dict = {
                "pixel_values": img*2 - 1,
                "depth_pixel_values": depth[:, 0:1, :, :]*2 - 1,  # Convert depth to 1-channel
                "normal_pixel_values": normal*2 - 1,
                "alb_pixel_values": alb*2 - 1,
                "scb_pixel_values": scb[:, 0:1, :, :]*2 - 1,  # Convert to 1-channel
                "rgb_pixel_values": rgb*2 - 1
            }

            # Save as .pkl
            with open(save_path, "wb") as f:
                pickle.dump({k: v.to(torch.float16) for k, v in data_dict.items()}, f)
            
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
    
    num = 6
    size = 128

    preprocessor = PreprocessAndSaveDataset(
        # root_dir="~/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/rendering",
        # root_dir="/fs/gamma-projects/svd_relight/sketchfab/rendering_2",
        root_dir="/fs/gamma-projects/svd_relight/sketchfab/rendering_pp",
        # save_dir="/fs/gamma-projects/svd_relight/sync/junk",
        save_dir="/fs/gamma-projects/svd_relight/paper_fin/train",
        sample_n_frames=14,
        width=size,
        height=size,
        num_samples = num
    )
    preprocessor.process_and_save()

    dataset = SyncDataset(preprocessed_dir="/fs/gamma-projects/svd_relight/paper_fin/train", max_frames=14)

    # Test loading a batch
    index_i = np.random.randint(0, num)

    sample = dataset[index_i]
    # sample = dataset[3]
    

    print(sample["pixel_values"].shape)  # Check dimensions

    pixel_values, depth_pixel_values, normal_pixel_values, alb_pixel_values, scb_pixel_values, rgb_pixel_values = sample["pixel_values"], sample["depth_pixel_values"], sample["normal_pixel_values"], sample["alb_pixel_values"], sample["scb_pixel_values"], sample["rgb_pixel_values"]

    print(pixel_values.shape)
    print(depth_pixel_values.shape)
    print(normal_pixel_values.shape)
    print(alb_pixel_values.shape)
    print(scb_pixel_values.shape)
    print(rgb_pixel_values.shape)

    pixel_values = (pixel_values+1)/2
    depth_pixel_values = (depth_pixel_values+1)/2
    normal_pixel_values = (normal_pixel_values+1)/2
    alb_pixel_values = (alb_pixel_values+1)/2
    scb_pixel_values = (scb_pixel_values+1)/2
    rgb_pixel_values = (rgb_pixel_values+1)/2

    save_array_as_image(pixel_values[0]*255, "~/DiffusionMaskRelight/outputs/sync/relit_01.png")
    save_array_as_image(pixel_values[2]*255, "~/DiffusionMaskRelight/outputs/sync/relit_06.png")
    save_array_as_image(pixel_values[-1]*255, "~/DiffusionMaskRelight/outputs/sync/relit_16.png")

    save_array_as_image_depth(depth_pixel_values[0]*255, "~/DiffusionMaskRelight/outputs/sync/dep_01.png")
    save_array_as_image_depth(depth_pixel_values[2]*255, "~/DiffusionMaskRelight/outputs/sync/dep_06.png")
    save_array_as_image_depth(depth_pixel_values[-1]*255, "~/DiffusionMaskRelight/outputs/sync/dep_16.png")
    
    save_array_as_image(normal_pixel_values[0]*255, "~/DiffusionMaskRelight/outputs/sync/nrm_01.png") 
    save_array_as_image(normal_pixel_values[2]*255, "~/DiffusionMaskRelight/outputs/sync/nrm_06.png") 
    save_array_as_image(normal_pixel_values[-1]*255, "~/DiffusionMaskRelight/outputs/sync/nrm_16.png")

    save_array_as_image(alb_pixel_values[0]*255, "~/DiffusionMaskRelight/outputs/sync/alb_01.png")
    save_array_as_image(alb_pixel_values[2]*255, "~/DiffusionMaskRelight/outputs/sync/alb_06.png")
    save_array_as_image(alb_pixel_values[-1]*255, "~/DiffusionMaskRelight/outputs/sync/alb_16.png")

    save_array_as_image_depth(scb_pixel_values[0]*255, "~/DiffusionMaskRelight/outputs/sync/scb_01.png")
    save_array_as_image_depth(scb_pixel_values[2]*255, "~/DiffusionMaskRelight/outputs/sync/scb_06.png")
    save_array_as_image_depth(scb_pixel_values[-1]*255, "~/DiffusionMaskRelight/outputs/sync/scb_16.png")

    save_array_as_image(rgb_pixel_values[0]*255, "~/DiffusionMaskRelight/outputs/sync/rgb_01.png")
    save_array_as_image(rgb_pixel_values[2]*255, "~/DiffusionMaskRelight/outputs/sync/rgb_06.png")
    save_array_as_image(rgb_pixel_values[-1]*255, "~/DiffusionMaskRelight/outputs/sync/rgb_16.png")
    # save_array_as_image(rgb_pixel_values[15]*255, "~/DiffusionMaskRelight/outputs/sync/rgb_16.png")

    # save_array_as_image(rgb_pixel_values[-1]*255, "~/DiffusionMaskRelight/outputs/sync/shd_16.png")

    print('done')