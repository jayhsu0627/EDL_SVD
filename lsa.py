from PIL import Image
from lang_sam import LangSAM
import numpy as np
import cv2
import os
import supervision as sv

# Initialize the LangSAM model and load the image
model = LangSAM()
image_pil = Image.open("./test_3.png").convert("RGB")

# Split the prompt string by '.' and filter out empty prompts.
text_prompt = "statue. box. pillar."
prompts = [p.strip() for p in text_prompt.split('.') if p.strip()]

print(prompts)

# Replicate the image if there are multiple prompts.
images = [image_pil] * len(prompts)

# Run predictions. Each image/prompt pair returns its own output.
results_list = model.predict(images, prompts)
print(results_list)

def merge_results(results_list):
    """Merge multiple prediction outputs into one."""
    merged = {'boxes': [], 'labels': [], 'scores': [], 'masks': []}
    for res in results_list:
        # Here we assume each key holds a list; adjust if they are numpy arrays.
        merged['boxes'].extend(res['boxes'])
        merged['labels'].extend(res['labels'])
        merged['scores'].extend(res['scores'])
        merged['masks'].extend(res['masks'])
    # Optionally, convert lists to numpy arrays if needed.
    merged['boxes'] = np.array(merged['boxes'])
    merged['scores'] = np.array(merged['scores'])
    return merged

# Merge results from different prompts into one dictionary.
merged_results = merge_results(results_list)

def visualize_segmentation_with_supervision(pil_image, output, output_path='segmentation_result.jpg', mask_save_dir=None):
    """
    Visualize segmentation results using Supervision's annotators and save mask copies.
    
    Parameters:
      - pil_image: Input image as a PIL Image object.
      - output: Segmentation output dictionary (expects keys: 'boxes', 'labels', 'scores', 'masks').
      - output_path: File path to save the annotated output image.
      - mask_save_dir: Directory to save individual mask images. If None, masks are not saved.
    
    Returns:
      - Annotated image as a PIL Image.
    """
    # Convert the PIL image to a NumPy array (RGB format)
    image_np = np.array(pil_image)
    
    # Unpack segmentation results
    boxes = output['boxes']
    labels = output['labels']
    scores = output['scores']
    masks = output['masks']
    
    # Process each mask: ensure it is binary and matches the image size
    processed_masks = []
    for mask in masks:
        # If mask is provided as a list (e.g. [mask_tensor]), take the first element.
        full_mask = mask[0] if isinstance(mask, list) else mask
        # If the mask has more than one channel, take the first one.
        if full_mask.ndim > 2:
            full_mask = full_mask[0]
        # Resize mask to match the image dimensions.
        full_mask_resized = cv2.resize(full_mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Threshold to create a binary mask.
        binary_mask = (full_mask_resized > 0.5).astype(np.uint8)
        processed_masks.append(binary_mask)
    processed_masks = np.stack(processed_masks, axis=0)
    
    # Optionally, save each binary mask as an image.
    if mask_save_dir is not None:
        os.makedirs(mask_save_dir, exist_ok=True)
        for i, mask in enumerate(processed_masks):
            # Multiply by 255 to convert from binary (0,1) to (0,255)
            mask_img = (mask * 255).astype(np.uint8)
            mask_path = os.path.join(mask_save_dir, f"mask_{i}.png")
            cv2.imwrite(mask_path, mask_img)
            print(f"Mask {i} saved to {mask_path}")
    
    # Map each unique label to a class id.
    unique_labels = list(set(labels))
    class_id_map = {label: idx for idx, label in enumerate(unique_labels)}
    class_ids = [class_id_map[label] for label in labels]
    
    # Create a Supervision detections object.
    detections = sv.Detections(
        xyxy=np.array(boxes),
        mask=processed_masks.astype(bool),
        confidence=np.array(scores),
        class_id=np.array(class_ids)
    )
    
    # Create annotators from Supervision.
    box_annotator = sv.BoxCornerAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()
    
    # Annotate image step-by-step.
    annotated_image = box_annotator.annotate(scene=image_np.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
    
    # Convert annotated image from RGB to BGR before saving.
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Ensure output directory exists and save the annotated image.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated_image_bgr)
    print(f"Visualization saved to {output_path}")
    
    return Image.fromarray(annotated_image_bgr)

# Visualize and save the merged segmentation output.
annotated_img = visualize_segmentation_with_supervision(
    image_pil, merged_results, 
    output_path='~/DiffusionMaskRelight/test_3_lsa.png',
    mask_save_dir='~/DiffusionMaskRelight/test_masks'
)
