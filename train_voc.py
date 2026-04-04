import argparse
import os
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance

from ssd_from_scratch import ssd300_vgg16, box_iou

from utils import (
    VOC2007Dataset,
    SSDAugmentation,
    ValTransform,
    collate_fn,
    compute_voc_map,
    get_lr,
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


def train_one_epoch(model, optimizer, loader, scaler, device):
    model.train()
    t0 = time.time()
    total_loss = total_loc = total_cls = 0.0
    num_batches = len(loader)

    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [
            {"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)}
            for t in targets
        ]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda"):
            losses = model(images, targets)
            loss   = losses["bbox_regression"] + losses["classification"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_loc  += losses["bbox_regression"].item()
        total_cls  += losses["classification"].item()

    elapsed = time.time() - t0
    imgs_per_sec = num_batches * loader.batch_size / elapsed
    return {
        "loss": total_loss / num_batches,
        "loc":  total_loc  / num_batches,
        "cls":  total_cls  / num_batches,
        "time": elapsed,
        "img_s": imgs_per_sec,
    }

@torch.no_grad()
def evaluate(model, loader, device) -> float:
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

    return compute_voc_map(all_preds, all_gts)


def parse_args():
    p = argparse.ArgumentParser(description="SSD300-VGG16 training on VOC2007")

    # Data
    p.add_argument("--data-root", default="VOCdevkit/VOC2007",
                   help="Path to VOCdevkit/VOC2007")

    # Training
    p.add_argument("--epochs",    type=int,   default=200)
    p.add_argument("--batch",     type=int,   default=256)
    p.add_argument("--lr",        type=float, default=2e-4)
    p.add_argument("--lr-steps",  type=int,   nargs=2, default=[150, 180],
                   help="Epochs at which LR is divided by 10 (default 150 180)")
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="Linear LR warmup for first N epochs")
    p.add_argument("--weight-decay", type=float, default=1e-4)

    # Infrastructure
    p.add_argument("--workers",    type=int,  default=4)
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--resume",     default=None, help="Path to checkpoint .pth")
    p.add_argument("--no-amp",     action="store_true",
                   help="Disable automatic mixed precision")
    p.add_argument("--no-pretrained-backbone", action="store_true",
                   help="Train VGG16 backbone from scratch")
    p.add_argument("--weights-path", default=None,
                   help="Local path to VGG16 features weights (from download_weights.py). "
                        "Use this to avoid downloading at training time.")

    return p.parse_args()


def main():
    args = parse_args()
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_ds = VOC2007Dataset(
        args.data_root, split="trainval", transform=SSDAugmentation()
    )
    val_ds = VOC2007Dataset(
        args.data_root, split="test", transform=ValTransform()
    )
    print(f"  train: {len(train_ds)} images   val: {len(val_ds)} images")

    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    val_loader = data.DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    print("Building model...")
    model = ssd300_vgg16(
        num_classes=NUM_CLASSES,
        pretrained_backbone=not args.no_pretrained_backbone,
        weights_path=args.weights_path,
    ).to(device)

    total_params  = sum(p.numel() for p in model.parameters())
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Total params:  {total_params:,}")
    print(f"  Frozen params: {frozen_params:,}")

    params_wd  = [p for n, p in model.named_parameters()
                  if p.requires_grad and "bias" not in n and "scale_weight" not in n]
    params_nwd = [p for n, p in model.named_parameters()
                  if p.requires_grad and ("bias" in n or "scale_weight" in n)]
    optimizer = torch.optim.AdamW(
        [
            {"params": params_wd,  "weight_decay": args.weight_decay},
            {"params": params_nwd, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )

    scaler = torch.amp.GradScaler(enabled=not args.no_amp)

    start_epoch = 1
    best_map    = 0.0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_map    = ckpt.get("best_map", 0.0)
        print(f"  Resuming from epoch {start_epoch}  (best mAP so far: {best_map:.4f})")

    print(f"\nTraining for {args.epochs} epochs  |  LR steps: {args.lr_steps}  |  AMP: {'off' if args.no_amp else 'on'}")
    print(f"{'Ep':>6}  {'lr':>8}  {'loss':>7}  {'loc':>7}  {'cls':>7}  {'mAP':>8}  time")
    print("-" * 72)

    for epoch in range(start_epoch, args.epochs + 1):
        lr = get_lr(epoch, args)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        mAP = 0
        is_best = False
        train_stats = train_one_epoch(model, optimizer, train_loader, scaler, device)
        if epoch % 20 == 0:
            mAP = evaluate(model, val_loader, device)

            is_best = mAP > best_map
            if is_best:
                best_map = mAP

            best_marker = " ★" if is_best else ""
            print(
                f"Ep {epoch:3d}/{args.epochs}  lr={lr:.2e}  "
                f"loss={train_stats['loss']:.3f}  loc={train_stats['loc']:.3f}  cls={train_stats['cls']:.3f}  "
                f"mAP={mAP * 100:.2f}%{best_marker}  "
                f"({train_stats['time']:.0f}s  {train_stats['img_s']:.0f} img/s)"
            )
        else:
            print(
                f"Ep {epoch:3d}/{args.epochs}  lr={lr:.2e}  "
                f"loss={train_stats['loss']:.3f}  loc={train_stats['loc']:.3f}  cls={train_stats['cls']:.3f}  "
                f"mAP=-"
                f"({train_stats['time']:.0f}s  {train_stats['img_s']:.0f} img/s)"
            )

        ckpt = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler":    scaler.state_dict(),
            "mAP":       mAP,
            "best_map":  best_map,
        }
        ckpt_path = os.path.join(args.output_dir, f"ssd_ep{epoch:03d}.pth")
        torch.save(ckpt, ckpt_path)
        if is_best:
            torch.save(ckpt, os.path.join(args.output_dir, "ssd_best.pth"))

    print(f"\nTraining complete.  Best mAP = {best_map * 100:.2f}%")


if __name__ == "__main__":
    main()
