from typing import Dict, List, Optional, Tuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.ops as tv_ops
import xml.etree.ElementTree as ET
from PIL import Image, ImageEnhance
import numpy as np
import torch.utils.data as data
import os
import torchvision.transforms.functional as TF


VOC_CLASSES = [
    "__background__",
    "aeroplane", "bicycle", "bird",   "boat",        "bottle",
    "bus",       "car",     "cat",    "chair",        "cow",
    "diningtable", "dog",   "horse",  "motorbike",    "person",
    "pottedplant", "sheep", "sofa",   "train",        "tvmonitor",
]
NUM_CLASSES = len(VOC_CLASSES)          # 21  (background + 20 VOC classes)
CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}

def box_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """
    Pairwise Jaccard (IoU) overlap between two sets of boxes.

    Args:
        boxes_a: [N, 4]  in [x1, y1, x2, y2]
        boxes_b: [M, 4]  in [x1, y1, x2, y2]
    Returns:
        iou: [N, M]
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """Greedy NMS backed by torchvision's CUDA/C++ kernel."""
    return tv_ops.nms(boxes, scores, iou_threshold)


def batched_nms(
    boxes: Tensor, scores: Tensor, labels: Tensor, iou_threshold: float
) -> Tensor:
    """Per-class NMS backed by torchvision's CUDA/C++ kernel."""
    return tv_ops.batched_nms(boxes, scores, labels, iou_threshold)


def clip_boxes_to_image(boxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
    """Clamp box coordinates to [0, W] × [0, H]."""
    h, w = image_shape
    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
    return boxes


def _xavier_init(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)


class PhotometricDistort:
    """
    Random brightness, contrast, saturation, hue (paper Appendix A).
    Operates on a uint8 PIL Image, returns PIL Image.
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        # Brightness
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))
        # Contrast – sometimes before colour, sometimes after
        apply_contrast_first = random.random() < 0.5
        if apply_contrast_first and random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.5))
        # Saturation
        if random.random() < 0.5:
            img = ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.5))
        # Contrast (second slot)
        if not apply_contrast_first and random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.5))
        return img


class RandomExpand:
    """
    Zoom out: place image in a larger canvas filled with the dataset mean.
    Updates box coordinates accordingly.
    """
    # ImageNet-approximate RGB mean in [0, 255]
    MEAN = (123, 117, 104)

    def __call__(self, img: np.ndarray, boxes: np.ndarray):
        if random.random() < 0.5 or len(boxes) == 0:
            return img, boxes

        h, w = img.shape[:2]
        ratio = random.uniform(1.0, 4.0)
        new_w, new_h = int(w * ratio), int(h * ratio)
        left  = random.randint(0, new_w - w)
        top   = random.randint(0, new_h - h)

        canvas = np.full((new_h, new_w, 3), self.MEAN, dtype=img.dtype)
        canvas[top : top + h, left : left + w] = img

        shifted = boxes.copy()
        shifted[:, [0, 2]] += left
        shifted[:, [1, 3]] += top
        return canvas, shifted


def _box_iou_np(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of boxes (numpy). Returns [N, M]."""
    inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


class RandomSampleCrop:
    """
    Sample a crop with minimum Jaccard overlap in {0.1, 0.3, 0.5, 0.7, 0.9}
    with at least one GT box.  Keeps only boxes whose centre lies inside crop.
    """
    _MIN_IOUS = (None, 0.1, 0.3, 0.5, 0.7, 0.9)

    def __call__(self, img: np.ndarray, boxes: np.ndarray, labels: np.ndarray):
        h, w = img.shape[:2]

        for _ in range(50):                          # outer: pick overlap mode
            min_iou = random.choice(self._MIN_IOUS)
            if min_iou is None:
                return img, boxes, labels            # keep full image

            for _ in range(50):                      # inner: sample a patch
                cw = random.uniform(0.3 * w, w)
                ch = random.uniform(0.3 * h, h)
                if not (0.5 <= ch / cw <= 2.0):      # aspect ratio guard
                    continue

                left = random.uniform(0, w - cw)
                top  = random.uniform(0, h - ch)
                patch = np.array([[left, top, left + cw, top + ch]])

                if len(boxes) == 0:
                    break

                ious = _box_iou_np(boxes, patch)[:, 0]
                if ious.max() < min_iou:
                    continue

                # Crop image
                cropped = img[int(top) : int(top + ch), int(left) : int(left + cw)]

                # Keep boxes whose centre is inside the patch
                cx = (boxes[:, 0] + boxes[:, 2]) / 2
                cy = (boxes[:, 1] + boxes[:, 3]) / 2
                inside = (cx > left) & (cx < left + cw) & \
                         (cy > top)  & (cy < top  + ch)
                if not inside.any():
                    continue

                boxes = boxes[inside].copy()
                labels = labels[inside].copy()

                # Clip + shift coordinates to patch origin
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], left, left + cw) - left
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], top,  top  + ch) - top
                return cropped, boxes, labels

        return img, boxes, labels


class SSDAugmentation:
    """
    Full SSD training augmentation pipeline (paper Sec. 4, Appendix A).

    Input:  PIL Image, list of [x1,y1,x2,y2] boxes, list of int labels
    Output: PIL Image (variable size), updated boxes and labels
    """

    def __init__(self):
        self.photometric = PhotometricDistort()
        self.expand      = RandomExpand()
        self.crop        = RandomSampleCrop()

    def __call__(self, image: Image.Image, boxes: list, labels: list):
        # ── photometric distortions (PIL) ─────────────────────────────────
        image = self.photometric(image)

        # ── geometric transforms (numpy) ──────────────────────────────────
        img_np = np.array(image, dtype=np.float32)
        boxes_np  = np.array(boxes,  dtype=np.float32).reshape(-1, 4)
        labels_np = np.array(labels, dtype=np.int64)

        img_np, boxes_np = self.expand(img_np, boxes_np)
        img_np, boxes_np, labels_np = self.crop(img_np, boxes_np, labels_np)

        # ── random horizontal flip ────────────────────────────────────────
        if random.random() < 0.5:
            h, w = img_np.shape[:2]
            img_np = img_np[:, ::-1].copy()
            if len(boxes_np):
                boxes_np[:, [0, 2]] = w - boxes_np[:, [2, 0]]

        image = Image.fromarray(img_np.clip(0, 255).astype(np.uint8))
        return image, boxes_np.tolist(), labels_np.tolist()

class VOC2007Dataset(data.Dataset):

    def __init__(self, root: str, split: str = "trainval", transform=None):
        self.root      = root
        self.transform = transform

        ids_file = os.path.join(root, "ImageSets", "Main", f"{split}.txt")
        with open(ids_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]

        img = Image.open(
            os.path.join(self.root, "JPEGImages", f"{img_id}.jpg")
        ).convert("RGB")

        boxes, labels = self._parse_xml(
            os.path.join(self.root, "Annotations", f"{img_id}.xml")
        )

        if self.transform is not None:
            img, boxes, labels = self.transform(img, boxes, labels)

        img_tensor = TF.to_tensor(img)   # [C, H, W], float32, [0, 1]

        target = {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img_tensor, target

    def _parse_xml(self, path: str):
        root = ET.parse(path).getroot()
        boxes, labels = [], []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()
            if name not in CLASS_TO_IDX:
                continue
            bb = obj.find("bndbox")
            x1 = float(bb.find("xmin").text)
            y1 = float(bb.find("ymin").text)
            x2 = float(bb.find("xmax").text)
            y2 = float(bb.find("ymax").text)
            # sanity: skip degenerate boxes
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(CLASS_TO_IDX[name])
        return boxes, labels

class ValTransform:
    """No augmentation for validation - just return image as-is."""
    def __call__(self, image: Image.Image, boxes: list, labels: list):
        return image, boxes, labels


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def compute_voc_map(
    all_preds: list,
    all_gts:   list,
    num_classes: int = NUM_CLASSES,
    iou_threshold: float = 0.5,
) -> float:
    aps = []

    for cls_idx in range(1, num_classes):  
        pred_scores_all = []
        pred_boxes_all  = []
        pred_img_all    = []
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

        if n_gt == 0:
            continue
        if not pred_scores_all:
            aps.append(0.0)
            continue

        all_scores = torch.cat(pred_scores_all)
        all_boxes  = torch.cat(pred_boxes_all)
        all_imgs   = torch.cat(pred_img_all)

        order     = all_scores.argsort(descending=True)
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
            iou_vals          = box_iou(pb.unsqueeze(0), gt_boxes)[0]
            best_iou, best_j  = iou_vals.max(0)
            if best_iou.item() >= iou_threshold and not matched_per_img[img_idx][best_j]:
                tp[k] = 1.0
                matched_per_img[img_idx][best_j] = True

        cum_tp    = tp.cumsum(0)
        cum_fp    = (1.0 - tp).cumsum(0)
        precision = cum_tp / (cum_tp + cum_fp + 1e-9)
        recall    = cum_tp / n_gt

        # 11-point interpolated AP
        ap = 0.0
        for t in torch.linspace(0, 1, 11):
            mask = recall >= t
            ap += precision[mask].max().item() if mask.any() else 0.0
        aps.append(ap / 11.0)

    return float(sum(aps) / len(aps)) if aps else 0.0


def get_lr(epoch: int, args) -> float:

    warmup = getattr(args, "warmup_epochs", 5)
    if epoch <= warmup:
        return args.lr * epoch / warmup
    elif epoch < args.lr_steps[0]:
        return args.lr
    elif epoch < args.lr_steps[1]:
        return args.lr * 0.1
    else:
        return args.lr * 0.01