# batch_iclight_bg.py

import os
import json
import math
import argparse

import numpy as np
import torch
from PIL import Image
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler
)
from transformers import pipeline

from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTokenizer, CLIPTextModel
from briarmbg import BriaRMBG
import safetensors.torch as sf
from torch.hub import download_url_to_file
from diffusers.utils import load_image

# Model & Scheduler setup
sd15_name = 'stablediffusionapi/realistic-vision-v51'

tokenizer   = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder= CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae         = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet        = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
rmbg        = BriaRMBG.from_pretrained("camenduru/RMBG-1.4")


# Patch UNet to accept 8 channels
with torch.no_grad():
    new_conv = torch.nn.Conv2d(
        8, unet.conv_in.out_channels,
        unet.conv_in.kernel_size,
        unet.conv_in.stride,
        unet.conv_in.padding
    )
    new_conv.weight.zero_()
    new_conv.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv.bias = unet.conv_in.bias
    unet.conv_in = new_conv

unet_original_forward = unet.forward

def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

unet.forward = hooked_unet_forward




# Load IC-Light weights
model_path = 'iclight_sd15_fc.safetensors'
if not os.path.exists(model_path):
    download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', dst=model_path)
sd_offset  = sf.load_file(model_path)
sd_origin  = unet.state_dict()
sd_merged  = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged


# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
text_encoder = text_encoder.to(device, dtype=torch.float16)
vae          = vae.to(device, dtype=torch.bfloat16)
unet         = unet.to(device, dtype=torch.float16)
rmbg         = rmbg.to(device, dtype=torch.float32)

# Enable efficient attention
unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Sampler setup
dpmpp_scheduler = DPMSolverMultistepScheduler.from_pretrained(
    sd15_name, subfolder="scheduler", torch_dtype=torch.float32
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
).to(device, dtype=torch.bfloat16)

captioner = pipeline(
    task="image-to-text",                          # <-- changed!
    model="Salesforce/blip-image-captioning-base",
    device=0                                       # or -1 for CPU
)

# Utilities
def encode_prompt_inner(txt: str):
    max_len     = tokenizer.model_max_length
    chunk_len   = max_len - 2
    id_start    = tokenizer.bos_token_id
    id_end      = tokenizer.eos_token_id
    id_pad      = id_end
    tokens      = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks      = [[id_start] + tokens[i:i+chunk_len] + [id_end]
                   for i in range(0, len(tokens), chunk_len)]
    chunks      = [c + [id_pad]*(max_len-len(c)) for c in chunks]
    token_ids   = torch.tensor(chunks).to(device)
    return text_encoder(token_ids).last_hidden_state

def encode_prompt_pair(pos, neg):
    c  = encode_prompt_inner(pos)
    uc = encode_prompt_inner(neg)
    # repeat to same length
    max_len = max(len(c), len(uc))
    c  = c.repeat(math.ceil(max_len/len(c)), 1, 1)[:max_len]
    uc = uc.repeat(math.ceil(max_len/len(uc)), 1, 1)[:max_len]
    c  = torch.cat([p[None,...] for p in c], dim=1)
    uc = torch.cat([p[None,...] for p in uc], dim=1)
    return c, uc

def pytorch2numpy(imgs, quant=True):
    out=[]
    for x in imgs:
        y = x.movedim(0,-1)
        if quant:
            y = (y*127.5+127.5).detach().cpu().numpy().clip(0,255).astype(np.uint8)
        else:
            y = (y*0.5+0.5).detach().cpu().numpy().clip(0,1).astype(np.float32)
        out.append(y)
    return out

def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs)/127.0-1.0).permute(0,3,1,2).to(device)
    return h

def resize_and_center_crop(img, tw, th):
    pil = Image.fromarray(img)
    ow, oh = pil.size
    sf = max(tw/ow, th/oh)
    nw, nh = int(ow*sf), int(oh*sf)
    pil = pil.resize((nw,nh), Image.LANCZOS)
    left, top = (nw-tw)/2,(nh-th)/2
    return np.array(pil.crop((left, top, left+tw, top+th)))

def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

# def run_rmbg(img_np, sigma=0.0):
#     H,W,_=img_np.shape
#     k = (256.0/(H*W))**0.5
#     # print(int(64*round(W*k)),int(64*round(H*k)))

#     feed = resize_and_center_crop(img_np,int(32*round(W*k)),int(32*round(H*k)))
#     feed = numpy2pytorch([feed]).to(dtype=torch.float32)
#     alpha = rmbg(feed)[0][0]
#     alpha = torch.nn.functional.interpolate(alpha[None], size=(H,W), mode="bilinear")[0]
#     alpha = alpha.movedim(0,-1).cpu().numpy().clip(0,1)
#     return (127+(img_np.astype(np.float32)-127+sigma)*alpha).clip(0,255).astype(np.uint8), alpha

@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(32 * round(W * k)), int(32 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha

base_ids = [
    "0b5da073be88481091dbef7e55f1d180",
    "4e6688dcb7b34c36ba81c8303ed078d1",
    "907ac12c61744803b22a49efd74ec40a",
]

test_dict = {}
for base_id in base_ids:
    if base_id == "0b5da073be88481091dbef7e55f1d180":
        target = [  4, 5, 15,  28, 38, 39, 42, 46, 50, 53, 54]
        # target = [ ]
        id_list = [f"{base_id}_r_{i:02d}" for i in target]
    elif base_id == "4e6688dcb7b34c36ba81c8303ed078d1":
        target = [ 1, 3, 5, 20, 29, 38, 43, 45, 46, 47, 51, 57]
        # target = [ ]
        id_list = [f"{base_id}_r_{i:02d}" for i in target]

    elif base_id == "907ac12c61744803b22a49efd74ec40a":
        target = [1, 3, 4, 7, 21, 26, 40, 46, 48, 49, 50, 60]
        # target = [20]
        id_list = [f"{base_id}_r_{i:02d}" for i in target]
    test_dict[base_id] = id_list

# Core relight function 
def batch_relight(jsonl,
                  width=512, height=512,
                  seed=42, steps=25,
                  cfg=7.5, highres_scale=1.5,
                  highres_denoise=0.5, lowres_denoise=0.9):
    with open(jsonl) as f: lines=f.readlines()
    for i, line in enumerate(lines,1):
        data = json.loads(line)
        base = os.path.dirname(data["normal"])

        folder_name = os.path.dirname(data["normal"]).split("/")[-1]
        uid = folder_name.split("_")[0]
        print(folder_name, uid)
        if folder_name in test_dict[uid]:
            print(f"Skipping {folder_name} as it is in the test set")
            continue

        frame = os.path.basename(data["normal"]).split("_")[1].split(".")[0]
        input_fg = data['target']
        input_fg = load_image(input_fg)

        result = captioner(input_fg, max_new_tokens=20)[0]
        caption = result.get("generated_text", result.get("caption"))
        # print("🖼️  Caption:", caption)
        input_fg = np.asarray(input_fg)

        concat_conds = numpy2pytorch([input_fg]).to(device=vae.device, dtype=vae.dtype)
        concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

        mask_pil = load_image(data['shading']).resize((width, height), Image.LANCZOS)
        # mask_np  = np.where(np.asarray(mask_pil)>128,255,0).astype(np.uint8)
        mask_np = np.asarray(mask_pil).copy()

        # let the mask range from 0 to 255, but keep unchanged area to 127
        mask_np[mask_np == 0] = 127

        fg_proc, _ = run_rmbg(input_fg)

        fg = resize_and_center_crop(fg_proc, width, height)
        concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
        concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

        conds, unconds = encode_prompt_pair( caption + ", best quality", 'bad quality')

        # conds, unconds = encode_prompt_pair( "best quality", 'bad quality')

        conds = conds.to(device=device, dtype=torch.float16)
        unconds = unconds.to(device=device, dtype=torch.float16)
        # encode mask as latent
        tmp = numpy2pytorch([mask_np]).to(device=vae.device, dtype=vae.dtype)
        mask_latent = vae.encode(tmp).latent_dist.mode()*vae.config.scaling_factor

        imgs = i2i_pipe(
            image=mask_latent,
            strength=lowres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=width, height=height,
            num_inference_steps=int(round(steps/lowres_denoise)),
            generator=torch.Generator(device).manual_seed(seed),
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds}
        ).images
        
        tmp = numpy2pytorch(imgs).to(device=vae.device, dtype=vae.dtype)

        # highres pass
        lat = vae.encode(tmp).latent_dist.mode()*vae.config.scaling_factor
        imgs2 = i2i_pipe(
            image=lat,
            strength=highres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=width, height=height,
            num_inference_steps=int(round(steps/highres_denoise)),
            generator=torch.Generator(device).manual_seed(seed),
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': mask_latent}
        ).images
        out = imgs2[0]
        out.save(os.path.join(base,f"iclight_{frame}.png"))
        print(f"Saved {i}/{len(lines)}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',    required=True, help='Path to JSONL file')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    batch_relight(args.data)
# python batch_inference.py --data /fs/nexus-scratch/sjxu/scriblit/dataset/paper_bike/prompt.json