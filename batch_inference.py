#!/usr/bin/env python
import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from accelerate import Accelerator
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from diffusers import AutoencoderKLTemporalDecoder
from transformers import CLIPVisionModelWithProjection
from utils.virtual_dataset_preprocess import SyncDataset

def export_frames_to_pngs(frames, output_dir, filename_prefix='frame'):
    """
    Export a list of frames to individual PNG images.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_dir (str): Directory to save the output PNG images.
    - filename_prefix (str, optional): Prefix for the output filenames.
                                       Images will be named like prefix_0000.png, prefix_0001.png, etc.
                                       Defaults to 'frame'.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Convert numpy arrays to PIL Images if needed, and save each one
    for i, frame in enumerate(frames):
        try:
            # Convert if it's a numpy array
            if isinstance(frame, np.ndarray):
                # Ensure correct format for PIL (e.g., convert BGR to RGB if needed)
                # If your frames are BGR from cv2, you might need: frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # For this generic function, assuming RGB or grayscale numpy arrays suitable for PIL
                pil_frame = Image.fromarray(frame)
            # If it's already a PIL Image, use it directly
            elif isinstance(frame, Image.Image):
                pil_frame = frame
            else:
                print(f"Skipping frame {i}: Unsupported format {type(frame)}")
                continue

            # Define the output filename with zero-padding for consistent sorting
            output_filename = f"{filename_prefix}_{i:04d}.png" # e.g., frame_0000.png, frame_0001.png
            output_filepath = os.path.join(output_dir, output_filename)
            print(f"Saving {output_filepath}")

            # Save the individual frame as a PNG
            pil_frame.save(output_filepath, format='PNG')

            # Optional: Print progress
            # print(f"Saved {output_filepath}")

        except Exception as e:
            print(f"Error saving frame {i}: {e}")

    print(f"Exported {len(frames)} frames to {output_dir}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--pretrain_unet", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--inferencemode", default="test")
    p.add_argument("--video_folder", required=True)
    p.add_argument("--output_dir",      default="./outputs")
    p.add_argument("--num_frames",      type=int, default=14)
    p.add_argument("--width",      type=int, default=14)
    p.add_argument("--per_gpu_batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=12345, help="A seed for reproducible training.")
    return p.parse_args()

def worker(rank, world_size, args):
    # pin to one GPU
    device = torch.device(f"cuda:{rank}")

    # build dataset & dataloader
    test_ds = SyncDataset(
        args.video_folder + f"/{args.inferencemode}/{args.width}",
        max_frames=args.num_frames
        )

    # pick only those idx that match this rank
    sampler = torch.utils.data.SubsetRandomSampler(
        [i for i in range(len(test_ds)) if i % world_size == rank]
    )
    dl = DataLoader(
        test_ds,
        batch_size=args.per_gpu_batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    weight_dtype = torch.float32
    
    # load models + pipeline on this GPU
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision, torch_dtype=weight_dtype).to(device)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16").to(device)
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=True
    ).to(device, dtype=weight_dtype)

    print(f"VAE dtype: {vae.dtype}")
    print(f"UNet dtype: {unet.dtype}")  

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        revision=args.revision,
        torch_dtype=weight_dtype
    ).to(device, dtype=weight_dtype)
    
    print(pipe.vae.dtype)  # should now be weight_dtype

    print(f"ImageEnc dtype: {image_encoder.dtype}")
    print(f"Pipeline dtype: {pipe.dtype}")
    
    pipe.vae.to(dtype=weight_dtype)
    print(pipe.vae.dtype)  # should now be weight_dtype

    pbar = tqdm(dl, desc=f"GPU {rank}", position=rank)
    for batch_idx, batch in enumerate(pbar):
        # batch["rgb_pixel_values"] etc. shape (B, F, C, H, W)
        rgb = batch["rgb_pixel_values"].to(weight_dtype).to(device)
        depth   = batch["depth_pixel_values"].to(weight_dtype).to(device)
        normal  = batch["normal_pixel_values"].to(weight_dtype).to(device)
        albedo  = batch["alb_pixel_values"].to(weight_dtype).to(device)
        scribble= batch["scb_pixel_values"].to(weight_dtype).to(device)
        
        print(rgb.dtype)
        print(depth.dtype)
        print(normal.dtype)
        print(albedo.dtype)
        print(scribble.dtype)
        
        # print(batch["scene_name"])
        scene_folder_name = batch["scene_name"][0].rsplit('_', 1)[0]
        print(scene_folder_name)
        # run inference
        out = pipe(
            rgb,
            g_buffer=[rgb, depth, normal, albedo, scribble, rgb],
            height=rgb.shape[-2],
            width= rgb.shape[-1],
            num_frames=args.num_frames,
            fps=8,
            generator=torch.Generator(device=device).manual_seed(args.seed + rank)
        ).frames[0]  # list of PIL Images length=num_frames

        # for i in range(args.num_frames):
        #     img = video_frames[i]
        #     video_frames[i] = np.array(img)

        export_frames_to_pngs(out, '~/svd_relight/sketchfab/rendering_pp/'+ scene_folder_name, filename_prefix=f"infer{args.width}")

        # # save frames
        # for f_idx, img in enumerate(out):
        #     # sample_folder = os.path.join(scene_folder_name, f"infer128_{batch_idx:04d}")
        #     # sample_folder = os.path.join(scene_folder_name, f"infer{args.width}_{batch_idx:04d}")
        #     # os.makedirs(sample_folder, exist_ok=True)
        #     img.save(os.path.join(scene_folder_name, f"infer{args.width}_{batch_idx:04d}"))

    pbar.close()

def main():
    args = parse_args()
    # figure out number of GPUs from CUDA_VISIBLE_DEVICES
    world_size = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
    mp.spawn(worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
