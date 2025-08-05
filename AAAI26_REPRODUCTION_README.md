# AAAI-26 Submission: Reproduction Instructions

This README provides comprehensive instructions for reproducing the training and evaluation results for our AAAI-26 submission.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Single Image Inference](#single-image-inference)
- [Expected Results](#expected-results)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- CUDA-enabled GPUs (minimum 4 GPUs recommended for training, 1-4 GPUs for inference)
- CUDA toolkit >= 11.6
- Python >= 3.8
- conda or miniconda
- Git LFS (for large model files)

## Installation

### 1. Clone the Repository
```bash
git clone [REPOSITORY_URL]
cd EDL_SVD
```

### 2. Setup Environment
```bash
conda env create -f environment.yml
conda activate diffusion_relight
```

### 3. Install Additional Dependencies
```bash
pip install accelerate wandb natsort
```

## Data Preparation

### Download Pretrained Models and Data

1. **Pretrained Models**: Download from Google Drive link [PLACEHOLDER_LINK]
   - Extract to appropriate model directories as specified in training commands

2. **Preprocessed Data**: 
   - Currently available at: `/fs/gamma-projects/svd_relight/sketchfab/rendering_pp`
   - For reproduction, download from [PLACEHOLDER_LINK]

### Data Structure
The preprocessed data should follow this structure:
```
data/
├── train/
│   ├── scene1/
│   │   ├── RGB0000.png
│   │   ├── albedo0000.png
│   │   ├── normal0000.png
│   │   ├── depth0000.png
│   │   ├── mask0000.png
│   │   └── relight0000.png
│   └── ...
└── test/
    └── [similar structure]
```

## Model Training

### Multi-GPU Training Command

Use the following command to train the model on 8 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu train_svd_relight_syn.py \
  --pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid" \
  --mixed_precision='fp16' \
  --enable_xformers_memory_efficient_attention \
  --video_folder="/workspace/Data_img" \
  --report_to="wandb" \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine_with_restarts" \
  --per_gpu_batch_size=2 \
  --gradient_accumulation_steps=8 \
  --num_train_epochs=5000 \
  --output_dir="/workspace/model_512/" \
  --num_workers=16 \
  --num_n_frames=14 \
  --num_frames=14 \
  --validation_steps=100 \
  --checkpointing_steps=500 \
  --width=512 \
  --height=512
```

### Training Configuration

- **Model**: Stable Video Diffusion (SVD) fine-tuned for relighting
- **Input Resolution**: 512×512
- **Batch Size**: 2 per GPU × 8 GPUs × 8 gradient accumulation = effective batch size of 128
- **Learning Rate**: 1e-4 with cosine restarts scheduler
- **Training Duration**: 5000 epochs
- **Mixed Precision**: FP16 for memory efficiency

### Monitoring Training

- Training progress is logged to Weights & Biases (wandb)
- Checkpoints are saved every 500 steps
- Validation runs every 100 steps

## Evaluation

We provide three evaluation scripts for different model comparisons:

### 1. EDL (Our Method) Evaluation
```bash
cd /fs/nexus-scratch/sjxu/EDL_SVD
python evaluate_edl.py
```

### 2. IC-Light Baseline Evaluation
```bash
cd /fs/nexus-scratch/sjxu/EDL_SVD
python evaluate_iclight.py
```

### 3. ScribbleLight Baseline Evaluation
```bash
cd /fs/nexus-scratch/sjxu/EDL_SVD
python evaluate_scribble.py
```

### Evaluation Metrics

The evaluation scripts compute the following metrics:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **LPIPS** (Learned Perceptual Image Patch Similarity)
- **MSE** (Mean Squared Error)

Results are saved as JSON files and logged to `metrics.log`.

## Single Image Inference

For single image inference, we provide three different methods:

### 1. EDL (Our Method)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python batch_inference.py \
  --pretrained_model_name_or_path "stabilityai/stable-video-diffusion-img2vid" \
  --pretrain_unet "/fs/gamma-projects/svd_relight/paper_model_fin/checkpoint-6000/" \
  --video_folder "/fs/gamma-projects/svd_relight/paper_fin" \
  --inferencemode train \
  --output_dir "/fs/gamma-projects/svd_relight/sketchfab/rendering_pp" \
  --num_frames 14 \
  --per_gpu_batch_size 1 \
  --width 128 \
  --height 128
```

### 2. IC-Light Baseline
Navigate to the IC-Light directory and run:
```bash
cd /fs/nexus-scratch/sjxu/EDL_SVD/ic-light-tost
python single_inference.py \
  --rgb "/fs/nexus-scratch/sjxu/ic-light-tost/colors_000.png" \
  --mask "/fs/nexus-scratch/sjxu/ic-light-tost/mask_000.png" \
  --output "/fs/nexus-scratch/sjxu/ic-light-tost/output_000.png"
```

### 3. ScribbleLight Baseline
Navigate to the ScribbleLight directory and run:
```bash
cd /fs/nexus-scratch/sjxu/EDL_SVD/scriblit
CUDA_VISIBLE_DEVICES=0 python inference_sketch.py \
  -n scribblelight_controlnet \
  -data paper_bike \
  -seed 1234
```

## Expected Results

### Quantitative Results
After running the evaluation scripts, you should expect the following approximate results:

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|--------|--------|---------|
| EDL (Ours) | XX.XX | X.XXX | X.XXX |
| IC-Light | XX.XX | X.XXX | X.XXX |
| ScribbleLight | XX.XX | X.XXX | X.XXX |

*Note: Please refer to the paper for exact numerical values.*

### Qualitative Results
- Output images will be saved in the specified output directories
- Visual comparisons can be found in the `inference/` subdirectories

## File Structure Overview

```
EDL_SVD/
├── train_svd_relight_syn.py     # Main training script
├── evaluate_edl.py              # EDL evaluation
├── evaluate_iclight.py          # IC-Light evaluation  
├── evaluate_scribble.py         # ScribbleLight evaluation
├── batch_inference.py           # Batch inference for EDL
├── environment.yml              # Conda environment
├── scriblit/                    # ScribbleLight implementation
│   ├── inference_sketch.py      # ScribbleLight inference
│   └── ...
├── ic-light-tost/              # IC-Light implementation
│   ├── single_inference.py     # IC-Light inference
│   └── ...
└── models/                     # Model architectures
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `per_gpu_batch_size` 
   - Increase `gradient_accumulation_steps` proportionally
   - Use `--enable_xformers_memory_efficient_attention`

2. **Environment Issues**
   - Ensure CUDA toolkit compatibility
   - Verify all dependencies are installed correctly
   - Check conda environment activation

3. **Data Loading Errors**
   - Verify data paths are correct
   - Check file permissions
   - Ensure data structure matches expected format

4. **Checkpoint Loading Issues**
   - Verify checkpoint paths exist
   - Check model compatibility
   - Ensure sufficient disk space

### Performance Optimization

- Use `--enable_xformers_memory_efficient_attention` for memory optimization
- Adjust `num_workers` based on your system's CPU cores
- Use mixed precision (`fp16`) to reduce memory usage

## Hardware Requirements

### Training
- **Minimum**: 4 × RTX 3090 (24GB VRAM each)
- **Recommended**: 8 × RTX 4090 (24GB VRAM each)
- **RAM**: 64GB system memory
- **Storage**: 500GB free space for models and data

### Inference
- **Minimum**: 1 × RTX 3080 (10GB VRAM)
- **Recommended**: 1 × RTX 4090 (24GB VRAM)

## Citation

If you use this code for your research, please cite:

```bibtex
@inproceedings{anonymous2026edl,
  title={[Paper Title]},
  author={[Anonymous for Review]},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## Support

For questions regarding reproduction:
- Check the troubleshooting section above
- Verify all prerequisites are met
- Ensure data and model paths are correctly set

---

**Note for Reviewers**: This reproduction guide provides complete instructions for training and evaluation. All placeholders (model downloads, data links) will be provided upon paper acceptance. The current file paths reference our development environment but can be adapted to any system following the same structure.
