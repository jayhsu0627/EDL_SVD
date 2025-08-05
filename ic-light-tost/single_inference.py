import os
import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler
)
from transformers import CLIPTokenizer, CLIPTextModel, pipeline
from briarmbg import BriaRMBG
import safetensors.torch as sf
from torch.hub import download_url_to_file
from diffusers.utils import load_image

# Model setup (same as batch script)
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
    task="image-to-text",
    model="Salesforce/blip-image-captioning-base",
    device=0 if torch.cuda.is_available() else -1
)

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
    max_len = max(len(c), len(uc))
    c  = c.repeat(torch.ceil(torch.tensor(max_len/len(c))).int(), 1, 1)[:max_len]
    uc = uc.repeat(torch.ceil(torch.tensor(max_len/len(uc))).int(), 1, 1)[:max_len]
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

def main(args):
    # Load images
    print(args.rgb, args.mask)
    rgb_img     = load_image(args.rgb)
    mask_img    = load_image(args.mask)
    # print(f"Loaded RGB image of shape {rgb_img.shape} and mask image of shape {mask_img.shape}")
    width, height = args.width, args.height

    # Preprocess foreground
    fg_proc, _ = run_rmbg(np.array(rgb_img))
    fg = resize_and_center_crop(fg_proc, width, height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    # Preprocess mask
    mask_pil = mask_img.resize((width, height), Image.LANCZOS)
    mask_np = np.asarray(mask_pil).copy()
    mask_np[mask_np == 0] = 127
    tmp = numpy2pytorch([mask_np]).to(device=vae.device, dtype=vae.dtype)
    mask_latent = vae.encode(tmp).latent_dist.mode()*vae.config.scaling_factor

    # Compose prompt from captioner
    result = captioner(rgb_img, max_new_tokens=20)[0]
    caption = result.get("generated_text", result.get("caption"))
    conds, unconds = encode_prompt_pair(caption + ", best quality", 'bad quality')
    conds = conds.to(device=device, dtype=torch.float16)
    unconds = unconds.to(device=device, dtype=torch.float16)

    # Run inference
    imgs = i2i_pipe(
        image=mask_latent,
        strength=args.denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=width, height=height,
        num_inference_steps=args.steps,
        generator=torch.Generator(device).manual_seed(args.seed),
        guidance_scale=args.cfg,
        cross_attention_kwargs={'concat_conds': concat_conds}
    ).images

    out = imgs[0]
    out.save(args.output)
    print(f"Saved output to {args.output}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--rgb',     required=True, help='Path to RGB image')
    p.add_argument('--depth',   required=False, help='Path to depth image')
    p.add_argument('--normal',  required=False, help='Path to normal image')
    p.add_argument('--mask',    required=True, help='Path to mask image')
    p.add_argument('--albedo',  required=False, help='Path to albedo image')
    p.add_argument('--output',  required=True, help='Path to save output image')
    p.add_argument('--width',   type=int, default=512)
    p.add_argument('--height',  type=int, default=512)
    p.add_argument('--seed',    type=int, default=42)
    p.add_argument('--steps',   type=int, default=25)
    p.add_argument('--cfg',     type=float, default=7.5)
    p.add_argument('--denoise', type=float, default=0.9)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)

# python single_inference.py \
#   --rgb "~/ic-light-tost/colors_000.png" \
#   --mask "~/ic-light-tost/mask_000.png" \
#   --output "~/ic-light-tost/output_000.png" \
#   --width 512 \
#   --height 512 \
#   --seed 42 \
#   --steps 25 \
#   --cfg 7.5 \
#   --denoise 0.9


# python single_inference.py \
#   --rgb "~/ic-light-tost/colors_001.png" \
#   --mask "~/ic-light-tost/mask_001.png" \
#   --output "~/ic-light-tost/output_001.png" \
#   --width 512 \
#   --height 512 \
#   --seed 42 \
#   --steps 25 \
#   --cfg 7.5 \
#   --denoise 0.9