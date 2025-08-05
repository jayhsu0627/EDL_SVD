# import json
# from pathlib import Path
# from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# import torch
# from tqdm import tqdm

# # ── CONFIG ────────────────────────────────────────────────────────────────
# DATASET = "sketchfab"  # or your args.data
# INPUT_JSON = Path(f"dataset/{DATASET}/output.json")
# OUTPUT_JSON = Path(f"dataset/{DATASET}/output_with_prompts.json")
# MODEL_NAME = "Salesforce/blip2-opt-2.7b"

# # ── SETUP DEVICE & MODEL ──────────────────────────────────────────────────
# device = "cuda" if torch.cuda.is_available() else "cpu"
# processor = Blip2Processor.from_pretrained(MODEL_NAME)
# model = Blip2ForConditionalGeneration.from_pretrained(
#     MODEL_NAME, torch_dtype=torch.float16
# )
# model.to(device)
# model.eval()

# # ── GENERATE CAPTIONS ────────────────────────────────────────────────────
# with INPUT_JSON.open("r") as fin, OUTPUT_JSON.open("w") as fout:
#     for line in tqdm(fin, desc="Generating prompts"):
#         entry = json.loads(line)
#         img_path = Path(entry["albedo"])
#         # load & preprocess
#         image = Image.open(img_path).convert("RGB")
#         inputs = processor(
#             images=image, return_tensors="pt"
#         ).to(device, torch.float16)
#         # generate
#         with torch.no_grad():
#             ids = model.generate(**inputs)
#         caption = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
#         # update & write
#         entry["prompt"] = caption
#         fout.write(json.dumps(entry) + "\n")
#         break

# from PIL import Image
# import requests
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
# )
# model.to(device)
# # url = "~/svd_relight/sketchfab/rendering_pp/0b5da073be88481091dbef7e55f1d180_r_01/diffuse_000.png"
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# # image = Image.open(url)

# inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

# generated_ids = model.generate(**inputs)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
# print(generated_text)



# from transformers import BlipForConditionalGeneration, AutoProcessor
# from PIL import Image
# import torch
# from transformers import pipeline

# device = "cuda" if torch.cuda.is_available() else "cpu"
# image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
# generated_text = image_to_text("/fs/nexus-scratch/sjxu/scriblit/dataset/data/albedo/0.png")

# print(f"Generated caption: {generated_text}")




# from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
# import torch
# from PIL import Image

# model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)



# max_length = 16
# num_beams = 4
# gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
# def predict_step(image_paths):
#   images = []
#   for image_path in image_paths:
#     i_image = Image.open(image_path)
#     if i_image.mode != "RGB":
#       i_image = i_image.convert(mode="RGB")

#     images.append(i_image)

#   pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
#   pixel_values = pixel_values.to(device)

#   output_ids = model.generate(pixel_values, **gen_kwargs)

#   preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#   preds = [pred.strip() for pred in preds]
#   return preds


# predict_step(["/fs/nexus-scratch/sjxu/scriblit/dataset/data/albedo/0.png"]) # ['a woman in a hospital bed with a woman in a hospital bed']


import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor
)



model_id = "Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed"
image_path = "path/to/your/image.jpg"

# the model requires more than 16GB of VRAM, 
# if you don't have you can use bitsandbytes to quantize the model to 8bit or 4bit


model = AutoModelForImageTextToText.from_pretrained(
  model_id,
  device_map="auto",
  torch_dtype=torch.bfloat16,
  attn_implementation="flash_attention_2", # Use "flash_attention_2" when running on Ampere or newer GPU or use "eager" for older GPUs
)


#### For lower precision less than 12GB VRAM ####

# Configure 4-bit quantization using BitsAndBytesConfig

#from transformers import BitsAndBytesConfig

# quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_quant_storage=torch.bfloat16,
#     )
# model = AutoModelForImageTextToText.from_pretrained(
#     model_id,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2", # Use "flash_attention_2" when running on Ampere or newer GPU or use "eager" for older GPUs
#     quantization_config=quantization_config,  # Use BitsAndBytesConfig instead of load_in_4bit
# )

########################################################################

# you can change the min and max pixels to fit your needs to decrease compute cost to trade off quality
min_pixels = 256*28*28
max_pixels = 1280*28*28

processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels, min_pixels=min_pixels)
system_message = "You are an expert image describer."

def generate_description(path, model, processor):
    image_inputs = Image.open(path).convert("RGB")
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image", "image": image_inputs},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # min_p and temperature are experemental parameters, you can change them to fit your needs
    generated_ids = model.generate(**inputs, max_new_tokens=512, min_p=0.1, do_sample=True, temperature=1.5)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

description = generate_description(image_path, model, processor)
print(description)
