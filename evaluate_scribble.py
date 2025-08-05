import os
import glob
import json
import logging
import numpy as np
from PIL import Image
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity
)
import lpips
import torch
from tqdm import tqdm

eval_width = 512
mask_evaluation = True

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ── 1) Setup logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    filename='~/DiffusionMaskRelight/metrics.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s',
)
logger = logging.getLogger()

# ── 2) Constants ────────────────────────────────────────────────────────────
base_ids = [
    "0b5da073be88481091dbef7e55f1d180",
    "4e6688dcb7b34c36ba81c8303ed078d1",
    "907ac12c61744803b22a49efd74ec40a",
]

test_dict = {}
for base_id in base_ids:
    if base_id == "0b5da073be88481091dbef7e55f1d180":
        target = [ 1, 4, 5, 15,  28, 38, 39, 42, 46, 50, 53, 54]
        id_list = [f"{base_id}_r_{i:02d}" for i in target]
    elif base_id == "4e6688dcb7b34c36ba81c8303ed078d1":
        target = [ 1, 3, 5, 20, 29, 38, 43, 45, 46, 47, 51, 57]
        id_list = [f"{base_id}_r_{i:02d}" for i in target]

    elif base_id == "907ac12c61744803b22a49efd74ec40a":
        target = [1, 3, 4, 7, 21, 26, 40, 46, 48, 49, 50, 60]
        id_list = [f"{base_id}_r_{i:02d}" for i in target]
    test_dict[base_id] = id_list


root_dir = "~/svd_relight/sketchfab/rendering_pp"

# ── 3) Load LPIPS model ─────────────────────────────────────────────────────
lpips_model = lpips.LPIPS(net='alex').cuda().eval()

# ── 4) Prepare accumulators ─────────────────────────────────────────────────
rmse_list, psnr_list, ssim_list, lpips_list = [], [], [], []
checkpoint_results = []

# ── 5) Setup progress bar ───────────────────────────────────────────────────
total_frames = len(base_ids) * 12 * 14
pbar = tqdm(total=total_frames, desc="Computing metrics")

# ── 6) Main loop ─────────────────────────────────────────────────────────────
count = 0
for base in base_ids:
    for r in range(1, 61):
        folder = f"{base}_r_{r:02d}"
        preds = sorted(glob.glob(os.path.join(root_dir, folder, "output*.png")))
        masks = sorted(glob.glob(os.path.join(root_dir, folder, "mask*.png")))
        if len(preds) == 0:
            continue
        if folder not in test_dict[base]:
            continue
        gts   = sorted(glob.glob(os.path.join(root_dir, folder, "relit_*.png")))[:14]
        
        assert len(preds) == len(gts) == 14, f"{folder} has {len(preds)}/{len(gts)} frames"
        print(f"Processing {folder} with {len(preds)} frames")
        
        for p_path, g_path, m_path in zip(preds, gts, masks):
            # load [0,1]
            pred = np.array(Image.open(p_path).convert("RGB"), dtype=np.float32)/255.0
            tmp_PIL = Image.open(g_path).convert("RGB")
            # — load & binarize mask
            mask_img = Image.open(m_path).convert("RGB")

            if eval_width == 128:
                tmp_PIL = tmp_PIL.resize((128, 128))
                mask_img = mask_img.resize((128,128), resample=Image.NEAREST)
            mask_arr = np.array(mask_img, dtype=np.uint8) > 128    # bool array H×W

            gt   = np.array(tmp_PIL,   dtype=np.float32)/255.0
            # print(f"pred: {pred.shape}, gt: {gt.shape}, mask: {mask_arr.shape}")
            if mask_evaluation:
                pred = pred * mask_arr
                gt   = gt * mask_arr

            # RMSE
            mse = mean_squared_error(gt, pred)
            rmse_list.append(np.sqrt(mse))
            
            # PSNR
            psnr_list.append(peak_signal_noise_ratio(gt, pred, data_range=1.0))
            
            # SSIM (use channel_axis)
            h, w = gt.shape[:2]
            win = 7 if min(h, w) >= 7 else (min(h, w)//2)*2 + 1
            ssim_list.append(
                structural_similarity(
                    gt, pred,
                    data_range=1.0,
                    channel_axis=2,
                    win_size=win
                )
            )
            
            # LPIPS
            t_pred = torch.from_numpy((pred*2-1).transpose(2,0,1)).unsqueeze(0).cuda()
            t_gt   = torch.from_numpy((gt*2-1).transpose(2,0,1)).unsqueeze(0).cuda()
            with torch.no_grad():
                d = lpips_model(t_gt, t_pred)
            lpips_list.append(d.item())
            
            # update progress
            count += 1
            pbar.update(1)

            # every 100: record and log interim stats
            if count % 100 == 0:
                stats = {
                    'count': count,
                    'rmse': np.mean(rmse_list),
                    'psnr': np.mean(psnr_list),
                    'ssim': np.mean(ssim_list),
                    'lpips': np.mean(lpips_list)
                }
                checkpoint_results.append(stats)
                pbar.set_postfix({
                    'RMSE': f"{stats['rmse']:.4f}",
                    'PSNR': f"{stats['psnr']:.2f}dB",
                    'SSIM': f"{stats['ssim']:.4f}",
                    'LPIPS': f"{stats['lpips']:.4f}"
                })
                logger.info(
                    f"Checkpoint {count}: "
                    f"RMSE={stats['rmse']:.4f}, "
                    f"PSNR={stats['psnr']:.2f}, "
                    f"SSIM={stats['ssim']:.4f}, "
                    f"LPIPS={stats['lpips']:.4f}"
                )

pbar.close()

final_stats = {
    'rmse': round(float(np.mean(rmse_list)), 3),
    'psnr': round(float(np.mean(psnr_list)), 3),
    'ssim': round(float(np.mean(ssim_list)), 3),
    'lpips': round(float(np.mean(lpips_list)), 3)
}

best_stats = {
    'rmse': round(float(np.min(rmse_list)), 3),
    'psnr': round(float(np.max(psnr_list)), 3),
    'ssim': round(float(np.max(ssim_list)), 3),
    'lpips': round(float(np.min(lpips_list)), 3)
}

summary = {
    'checkpoints': checkpoint_results,
    'final':      final_stats,
    'best':       best_stats
}


logger.info(
    f"Bests -> RMSE(min)={best_stats['rmse']:.4f}, "
    f"PSNR(max)={best_stats['psnr']:.2f}, "
    f"SSIM(max)={best_stats['ssim']:.4f}, "
    f"LPIPS(min)={best_stats['lpips']:.4f}"
)

output_dir = '~/DiffusionMaskRelight/scrib_summary.json'
if mask_evaluation:
    output_dir = '~/DiffusionMaskRelight/scrib_mask_summary.json'

with open(output_dir, 'w') as f:
    json.dump(summary, f, indent=2, cls=NumpyEncoder)


print("Done! Interim logs in metrics.log; full summary in metrics_summary.json")
