import os
import json
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers import (
	AutoencoderKL,
	DDPMScheduler,
	StableDiffusionControlNetPipeline,
	UNet2DConditionModel,
	UniPCMultistepScheduler,
	DDIMScheduler
)
from diffusers.utils import load_image
from network_controlnet import ControlNetModel
from pipeline_cn import CustomControlNetPipeline
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
import argparse
import torch.multiprocessing as mp

def image_to_tensor(image):
	image = image / 255.0
	image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
	return image.float()

def replace_unet_conv_in(unet):
	# replace the first layer to accept 8 in_channels
	_weight = unet.conv_in.weight.clone()  # [320, 4, 3, 3]
	_bias = unet.conv_in.bias.clone()  # [320]
	_weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)

	# new conv_in channel
	_n_convin_out_channel = unet.conv_in.out_channels
	_new_conv_in = Conv2d(
		8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
	)
	_new_conv_in.weight = Parameter(_weight)
	_new_conv_in.bias = Parameter(_bias)
	unet.conv_in = _new_conv_in
	
	# replace config
	unet.config["in_channels"] = 8
	return

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	return torch.manual_seed(seed)

def worker(gpu_id, args, all_lines):
    # pin this process to GPU gpu_id
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    # load model once per process
    base_model = "stabilityai/stable-diffusion-2-1"
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    replace_unet_conv_in(unet)
    best = natsorted(os.listdir(args.n))
    best = [f for f in best if f.startswith("checkpoint")][-1]
    unet.load_state_dict(torch.load(f"{args.n}/{best}/custom_unet.pth"), strict=False)
    unet.to(device).eval()

    controlnet = ControlNetModel.from_pretrained(f"{args.n}/{best}/controlnet", torch_dtype=torch.float32).to(device)
    # pipe = CustomControlNetPipeline.from_pretrained(
    #     base_model, controlnet=controlnet, torch_dtype=torch.float32, unet=unet
    # ).to(device)

    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.enable_model_cpu_offload()
    pipe = CustomControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=torch.float32,
        unet=unet,
    )
    # replace the scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # move *all* sub-modules onto this worker’s GPU
    pipe = pipe.to(device)
    
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(device)
    noise_sched = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    # transforms
    image_tf = transforms.Compose([
        transforms.Resize((512,512)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])

    # each worker gets every 4th line, starting at its gpu_id
    for idx in tqdm(range(gpu_id, len(all_lines), 4), desc=f"GPU {gpu_id}", position=gpu_id):
        data = json.loads(all_lines[idx])
        base = os.path.dirname(data["normal"])
        frame = os.path.basename(data["normal"]).split("_")[1].split(".")[0]

        # prepare conditioning
        cn = load_image(data["normal"]).resize((512,512))
        cs = load_image(data["shading"])
        arr = np.array(cs)
        arr[arr==0] = 127
        cs = Image.fromarray(arr.astype(np.uint8)).resize((512,512))

        img = image_tf(load_image(data["albedo"])).unsqueeze(0).to(device)
        with torch.no_grad():
            lat = vae.encode(img).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(lat)
            t = torch.randint(200,201,(1,)).to(device)
            lat = noise_sched.add_noise(lat, noise, t)

            gen = set_seed(args.seed)  # re-seeds each run
            out = pipe(
                data["prompt"],
                num_inference_steps=20,
                generator=gen,
                image=(cn, cs),
                albedo_latents=lat
            ).images[0]

        out.save(f"{base}/output_{frame}.png")
        cs.convert("RGB").save(f"{base}/shd_{frame}.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",  "--n",    required=True)
    parser.add_argument("-data", "--data", default="openroom")
    parser.add_argument("-seed", "--seed", type=int, default=6071)
    args = parser.parse_args()

    with open(f"dataset/{args.data}/output.json") as f:
        lines = f.readlines()

    mp.spawn(worker, args=(args, lines), nprocs=4, join=True)

if __name__ == "__main__":
    main()
