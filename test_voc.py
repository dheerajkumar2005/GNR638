import argparse
import os
import random
import time

import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from utils import (
    VOC2007Dataset,
    ValTransform,
    collate_fn,
    compute_voc_map,
)

VOC_CLASSES = [
    "__background__",
    "aeroplane", "bicycle", "bird",   "boat",        "bottle",
    "bus",       "car",     "cat",    "chair",        "cow",
    "diningtable", "dog",   "horse",  "motorbike",    "person",
    "pottedplant", "sheep", "sofa",   "train",        "tvmonitor",
]
NUM_CLASSES = len(VOC_CLASSES)          # 21  (background + 20 VOC classes)
CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}

_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


@torch.no_grad()
def run_inference(model, loader, device):
    """Return (all_preds, all_gts) lists over the full dataset."""
    model.eval()
    all_preds, all_gts = [], []

    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds  = model(images)

        for pred, gt in zip(preds, targets):
            all_preds.append({
                "boxes":  pred["boxes"].cpu(),
                "scores": pred["scores"].cpu(),
                "labels": pred["labels"].cpu(),
            })
            all_gts.append({
                "boxes":  gt["boxes"],
                "labels": gt["labels"],
            })

    return all_preds, all_gts


def compute_per_class_ap(all_preds, all_gts, num_classes=NUM_CLASSES, iou_threshold=0.5):
    """
    Same logic as compute_voc_map but returns per-class APs as well.
    Returns (mean_ap, {class_name: ap})
    """
    from train_voc import box_iou

    per_class = {}

    for cls_idx in range(1, num_classes):
        pred_scores_all, pred_boxes_all, pred_img_all = [], [], []
        n_gt = 0
        gt_boxes_per_img = []

        for img_idx, (pred, gt) in enumerate(zip(all_preds, all_gts)):
            gt_mask  = gt["labels"] == cls_idx
            gt_boxes = gt["boxes"][gt_mask]
            n_gt += int(gt_mask.sum().item())
            gt_boxes_per_img.append(gt_boxes)

            pd_mask = pred["labels"] == cls_idx
            if not pd_mask.any():
                continue
            pred_scores_all.append(pred["scores"][pd_mask])
            pred_boxes_all.append(pred["boxes"][pd_mask])
            pred_img_all.append(
                torch.full((pd_mask.sum(),), img_idx, dtype=torch.long)
            )

        cls_name = VOC_CLASSES[cls_idx]

        if n_gt == 0:
            continue
        if not pred_scores_all:
            per_class[cls_name] = 0.0
            continue

        all_scores = torch.cat(pred_scores_all)
        all_boxes  = torch.cat(pred_boxes_all)
        all_imgs   = torch.cat(pred_img_all)

        order      = all_scores.argsort(descending=True)
        all_scores = all_scores[order]
        all_boxes  = all_boxes[order]
        all_imgs   = all_imgs[order]

        matched_per_img = [
            torch.zeros(len(gt_boxes_per_img[i]), dtype=torch.bool)
            for i in range(len(all_preds))
        ]
        tp = torch.zeros(len(all_scores))

        for k in range(len(all_scores)):
            img_idx  = all_imgs[k].item()
            pb       = all_boxes[k]
            gt_boxes = gt_boxes_per_img[img_idx]
            if len(gt_boxes) == 0:
                continue
            iou_vals         = box_iou(pb.unsqueeze(0), gt_boxes)[0]
            best_iou, best_j = iou_vals.max(0)
            if best_iou.item() >= iou_threshold and not matched_per_img[img_idx][best_j]:
                tp[k] = 1.0
                matched_per_img[img_idx][best_j] = True

        cum_tp    = tp.cumsum(0)
        cum_fp    = (1.0 - tp).cumsum(0)
        precision = cum_tp / (cum_tp + cum_fp + 1e-9)
        recall    = cum_tp / n_gt

        ap = 0.0
        for t in torch.linspace(0, 1, 11):
            mask = recall >= t
            ap  += precision[mask].max().item() if mask.any() else 0.0
        per_class[cls_name] = ap / 11.0

    mean_ap = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return mean_ap, per_class


def draw_boxes(img_path, pred, gt, score_thresh=0.3):
    """
    Draw ground-truth (green) and predicted (coloured by class) boxes.
    Returns a PIL Image.
    """
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Ground truth – green outline
    for box, label in zip(gt["boxes"].tolist(), gt["labels"].tolist()):
        draw.rectangle(box, outline="lime", width=2)
        draw.text((box[0], max(0, box[1] - 14)), VOC_CLASSES[label], fill="lime", font=font)

    # Predictions – coloured outline
    for box, score, label in zip(
        pred["boxes"].tolist(), pred["scores"].tolist(), pred["labels"].tolist()
    ):
        if score < score_thresh:
            continue
        colour = _PALETTE[(label - 1) % len(_PALETTE)]
        draw.rectangle(box, outline=colour, width=2)
        text = f"{VOC_CLASSES[label]} {score:.2f}"
        draw.text((box[0], box[1] + 2), text, fill=colour, font=font)

    return img


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SSD300 on VOC2007 test set")
    p.add_argument("--checkpoint",  required=True, help="Path to .pth checkpoint")
    p.add_argument("--data-root",   default="VOCdevkit/VOC2007")
    p.add_argument("--batch",       type=int,  default=32)
    p.add_argument("--workers",     type=int,  default=4)
    p.add_argument("--score-thresh",type=float,default=0.01,
                   help="Score threshold passed to model (default 0.01)")
    p.add_argument("--iou-thresh",  type=float,default=0.5,
                   help="IoU threshold for mAP TP matching (default 0.5)")
    p.add_argument("--torchvision", action="store_true",
                   help="Load checkpoint into a torchvision SSD (train_voc_torchvision.py)")
    p.add_argument("--save-vis",    action="store_true",
                   help="Save visualisation images to --vis-dir")
    p.add_argument("--num-vis",     type=int,  default=20,
                   help="Number of images to visualise (default 20)")
    p.add_argument("--vis-dir",     default="vis_output",
                   help="Directory to save visualisations")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_epoch = ckpt.get("epoch", "?")
    saved_map   = ckpt.get("mAP", None)
    print(f"  Saved at epoch {saved_epoch}" +
          (f"  (checkpoint mAP = {saved_map*100:.2f}%)" if saved_map is not None else ""))

    if args.torchvision:
        from torchvision.models.detection import ssd300_vgg16
        model = ssd300_vgg16(
            weights=None,
            num_classes=NUM_CLASSES,
            weights_backbone=None,
            score_thresh=args.score_thresh,
        )
    else:
        from ssd_from_scratch import ssd300_vgg16
        model = ssd300_vgg16(
            num_classes=NUM_CLASSES,
            pretrained_backbone=False,
            score_thresh=args.score_thresh,
        )

    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    test_ds = VOC2007Dataset(args.data_root, split="test", transform=ValTransform())
    print(f"  Test images: {len(test_ds)}")

    loader = data.DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    print("\nRunning inference...", flush=True)
    t0 = time.time()
    all_preds, all_gts = run_inference(model, loader, device)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({len(test_ds)/elapsed:.0f} img/s)")

    print(f"\nComputing mAP (IoU ≥ {args.iou_thresh})...")
    mean_ap, per_class = compute_per_class_ap(
        all_preds, all_gts,
        num_classes=NUM_CLASSES,
        iou_threshold=args.iou_thresh,
    )

    print(f"\n{'Class':<16}  {'AP':>7}")
    print("-" * 26)
    for cls_name, ap in sorted(per_class.items(), key=lambda x: -x[1]):
        print(f"  {cls_name:<14}  {ap*100:>6.2f}%")
    print("-" * 26)
    print(f"  {'mAP':<14}  {mean_ap*100:>6.2f}%")

    if args.save_vis:
        os.makedirs(args.vis_dir, exist_ok=True)
        indices = random.sample(range(len(test_ds)), min(args.num_vis, len(test_ds)))
        print(f"\nSaving {len(indices)} visualisations to {args.vis_dir}/")

        for idx in indices:
            img_id  = test_ds.ids[idx]
            img_path = os.path.join(args.data_root, "JPEGImages", f"{img_id}.jpg")
            vis = draw_boxes(img_path, all_preds[idx], all_gts[idx])
            vis.save(os.path.join(args.vis_dir, f"{img_id}.jpg"))

        print("Done.")


if __name__ == "__main__":
    main()
