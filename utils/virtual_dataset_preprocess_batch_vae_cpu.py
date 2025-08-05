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
from torch.cuda.amp import autocast


# pick GPU 0 if available, else fall back to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    # t = rearrange(t, "b f c h w -> (b f) c h w")
    with torch.no_grad():
        latents = vae.encode(t).latent_dist.sample()
        
    del t
    # latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor
    torch.cuda.empty_cache()  # 🚀 Free GPU memory
    return latents

def latent_to_tensor(latents, vae, frames=2):

    video_length = latents.shape[1]
    # print(vae.config.scaling_factor)

    latents = latents / vae.config.scaling_factor
    # latents = rearrange(latents, "b f c h w -> (b f) c h w")
    
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        t = vae.decode(latents, num_frames=frames).sample
    del latents  # 🚀 Free `latents` after decoding
    # t = rearrange(t, "(b f) c h w -> b f c h w", f=video_length)
    torch.cuda.empty_cache()  # 🚀 Free GPU memory
    return t

def save_generated_images(tensor, output_dir, pre):
    """
    Save generated images from a tensor.

    Args:
        tensor (torch.Tensor): The tensor containing generated images (b, f, c, h, w).
        output_dir (str): Directory to save the images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert tensor to NumPy array and move to CPU
    images_np = tensor.cpu().detach().numpy()  # Shape: (b, f, c, h, w)

    # Iterate over batches and frames
    for b in range(images_np.shape[0]):
        for f in range(images_np.shape[1]):
            image = images_np[b, f]  # Shape: (c, h, w)

            # Convert channels from CHW to HWC
            if image.shape[0] >= 3:  # RGB image
                image = image.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            else:  # Grayscale image
                image = image[0]  # Use the first channel

            # Normalize to [0, 255] for saving
            image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.astype(np.uint8)

            # Save the image
            image_path = os.path.join(output_dir, f"image_b{b}_f{f}_"+ str(pre) +".png")
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, image)
            print(f"Saved image: {image_path}")

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

    dataset = SyncDataset(preprocessed_dir="~/outputs/quick", max_frames=14)

    # Test loading a batch
    index_i = np.random.randint(0, num)

    sample = dataset[index_i]

    pixel_values, depth_pixel_values, normal_pixel_values, alb_pixel_values, scb_pixel_values, rgb_pixel_values = sample["pixel_values"], sample["depth_pixel_values"], sample["normal_pixel_values"], sample["alb_pixel_values"], sample["scb_pixel_values"], sample["rgb_pixel_values"]
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", subfolder="vae", revision=None, torch_dtype=torch.float16 )
    vae = vae.to(device)       # <<< send all params (weights + biases) to GPU

    print(pixel_values.shape)
    print(depth_pixel_values.shape)
    print(normal_pixel_values.shape)
    print(alb_pixel_values.shape)
    print(scb_pixel_values.shape)
    print(rgb_pixel_values.shape)

    pixel_values = pixel_values.to(device)
    depth_pixel_values = depth_pixel_values.to(device)
    normal_pixel_values = normal_pixel_values.to(device)
    alb_pixel_values = alb_pixel_values.to(device)
    scb_pixel_values = scb_pixel_values.to(device)
    rgb_pixel_values = rgb_pixel_values.to(device)

    input = pixel_values.cpu().detach()
    depth = latent_to_tensor(depth_pixel_values, vae).cpu().detach()
    nrm = latent_to_tensor(normal_pixel_values, vae).cpu().detach()
    alb = latent_to_tensor(alb_pixel_values, vae).cpu().detach()
    mask = latent_to_tensor(scb_pixel_values, vae).cpu().detach()
    rgb = rgb_pixel_values.cpu().detach()

    input = (input+1)/2
    depth = (depth+1)/2
    nrm =  (nrm+1)/2
    alb = (alb+1)/2
    rgb = (rgb+1)/2
    mask = (mask+1)/2


    print(mask.shape)
    save_array_as_image(input[0]*255, "~/outputs/input_1.png")
    save_array_as_image(depth[0]*255, "~/outputs/depth_01.png")
    save_array_as_image(nrm[0]*255, "~/outputs/normal_01.png")
    save_array_as_image(alb[0]*255, "~/outputs/alb_01.png")
    save_array_as_image(mask[0]*255, "~/outputs/scb_01.png")
    save_array_as_image(rgb[0]*255, "~/outputs/rgb.png")

  
    print('done')