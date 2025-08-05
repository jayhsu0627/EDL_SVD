
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
