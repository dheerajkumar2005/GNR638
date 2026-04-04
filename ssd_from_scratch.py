import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from utils import box_iou,clip_boxes_to_image,_xavier_init,batched_nms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import vgg16, VGG16_Weights


class BoxCoder:
    def __init__(
        self,
        weights: Tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0),
    ):
        self.weights = weights

    def encode_single(self, gt_boxes: Tensor, anchors: Tensor) -> Tensor:
        wx, wy, ww, wh = self.weights

        aw = anchors[:, 2] - anchors[:, 0]
        ah = anchors[:, 3] - anchors[:, 1]
        acx = anchors[:, 0] + 0.5 * aw
        acy = anchors[:, 1] + 0.5 * ah

        gw = gt_boxes[:, 2] - gt_boxes[:, 0]
        gh = gt_boxes[:, 3] - gt_boxes[:, 1]
        gcx = gt_boxes[:, 0] + 0.5 * gw
        gcy = gt_boxes[:, 1] + 0.5 * gh

        dx = wx * (gcx - acx) / aw
        dy = wy * (gcy - acy) / ah
        dw = ww * torch.log(gw / aw)
        dh = wh * torch.log(gh / ah)

        return torch.stack([dx, dy, dw, dh], dim=1)

    def decode_single(self, deltas: Tensor, anchors: Tensor) -> Tensor:
        wx, wy, ww, wh = self.weights

        aw = anchors[:, 2] - anchors[:, 0]
        ah = anchors[:, 3] - anchors[:, 1]
        acx = anchors[:, 0] + 0.5 * aw
        acy = anchors[:, 1] + 0.5 * ah

        dx = deltas[:, 0] / wx
        dy = deltas[:, 1] / wy
        dw = (deltas[:, 2] / ww).clamp(max=math.log(1000.0 / 16))
        dh = (deltas[:, 3] / wh).clamp(max=math.log(1000.0 / 16))

        pred_cx = dx * aw + acx
        pred_cy = dy * ah + acy
        pred_w = torch.exp(dw) * aw
        pred_h = torch.exp(dh) * ah

        x1 = pred_cx - 0.5 * pred_w
        y1 = pred_cy - 0.5 * pred_h
        x2 = pred_cx + 0.5 * pred_w
        y2 = pred_cy + 0.5 * pred_h

        return torch.stack([x1, y1, x2, y2], dim=1)

class DefaultBoxGenerator(nn.Module):

    def __init__(
        self,
        aspect_ratios: List[List[int]],
        scales: Optional[List[float]] = None,
        min_ratio: float = 0.2,
        max_ratio: float = 0.9,
        steps: Optional[List[int]] = None,
        clip: bool = True,
    ):
        super().__init__()
        self.aspect_ratios = aspect_ratios
        self.steps = steps
        self.clip = clip

        m = len(aspect_ratios)
        if scales is None:
            scales = [
                min_ratio + (max_ratio - min_ratio) * k / (m - 1) for k in range(m)
            ]
            scales.append(1.0) 
        self.scales = scales
        self._wh_pairs: List[Tensor] = self._compute_wh_pairs(m)

    def _compute_wh_pairs(self, num_feature_maps: int) -> List[Tensor]:

        pairs = []
        for k in range(num_feature_maps):
            s_k = self.scales[k]
            s_k_prime = math.sqrt(s_k * self.scales[k + 1])    # extra ar=1 box

            wh: List[List[float]] = [[s_k, s_k], [s_k_prime, s_k_prime]]
            for ar in self.aspect_ratios[k]:
                sq = math.sqrt(ar)
                wh.append([s_k * sq, s_k / sq])    # aspect ratio ar
                wh.append([s_k / sq, s_k * sq])    # aspect ratio 1/ar

            pairs.append(torch.tensor(wh, dtype=torch.float32))   # [A_k, 2]
        return pairs

    def num_anchors_per_location(self) -> List[int]:
        return [2 + 2 * len(r) for r in self.aspect_ratios]

    def forward(
        self, image_size: Tuple[int, int], feature_maps: List[Tensor]
    ) -> Tensor:

        img_h, img_w = image_size
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        all_boxes: List[Tensor] = []
        for k, fmap in enumerate(feature_maps):
            fh, fw = fmap.shape[-2], fmap.shape[-1]

            # Step size (as a fraction of the image) for grid centers
            if self.steps is not None:
                sx = self.steps[k] / img_w
                sy = self.steps[k] / img_h
            else:
                sx = 1.0 / fw
                sy = 1.0 / fh

            # Normalized center grid: ((i + 0.5)/|f_k|, (j + 0.5)/|f_k|)
            cx = (torch.arange(fw, dtype=dtype, device=device) + 0.5) * sx   # [fw]
            cy = (torch.arange(fh, dtype=dtype, device=device) + 0.5) * sy   # [fh]
            grid_y, grid_x = torch.meshgrid(cy, cx, indexing="ij")           # [fh, fw]
            grid_x = grid_x.reshape(-1)    # [fh·fw]
            grid_y = grid_y.reshape(-1)

            wh = self._wh_pairs[k].to(dtype=dtype, device=device)   # [A_k, 2]
            num_a = wh.shape[0]
            num_loc = grid_x.shape[0]

            # Broadcast to [num_loc, num_a]
            gcx = grid_x[:, None].expand(num_loc, num_a)
            gcy = grid_y[:, None].expand(num_loc, num_a)
            gw  = wh[None, :, 0].expand(num_loc, num_a)
            gh  = wh[None, :, 1].expand(num_loc, num_a)

            # Convert to pixel [x1, y1, x2, y2]
            x1 = (gcx - 0.5 * gw).reshape(-1) * img_w
            y1 = (gcy - 0.5 * gh).reshape(-1) * img_h
            x2 = (gcx + 0.5 * gw).reshape(-1) * img_w
            y2 = (gcy + 0.5 * gh).reshape(-1) * img_h

            all_boxes.append(torch.stack([x1, y1, x2, y2], dim=1))

        anchors = torch.cat(all_boxes, dim=0)   # [total_anchors, 4]

        if self.clip:
            anchors[:, [0, 2]] = anchors[:, [0, 2]].clamp(0.0, float(img_w))
            anchors[:, [1, 3]] = anchors[:, [1, 3]].clamp(0.0, float(img_h))

        return anchors

class SSDMatcher:

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def __call__(self, match_quality_matrix: Tensor) -> Tensor:
 
        if match_quality_matrix.numel() == 0:
            return torch.full(
                (match_quality_matrix.shape[1],), -1,
                dtype=torch.int64, device=match_quality_matrix.device,
            )

        # Step 2: best GT for every anchor
        matched_vals, matches = match_quality_matrix.max(dim=0)  # [num_anchors]
        matches[matched_vals < self.iou_threshold] = -1          # mark background

        # Step 1: best anchor for every GT (ensures no GT is left unmatched)
        best_anchor_per_gt, _ = match_quality_matrix.max(dim=1)          # [num_gt]
        gt_idx, anchor_idx = torch.where(
            match_quality_matrix == best_anchor_per_gt[:, None]
        )
        matches[anchor_idx] = gt_idx

        return matches

class SSDFeatureExtractorVGG(nn.Module):

    out_channels: List[int] = [512, 1024, 512, 256, 256, 256]

    def __init__(self, backbone: nn.Sequential):
        super().__init__()

        # Locate MaxPool layers in the VGG features Sequential
        pool_pos = [i for i, l in enumerate(backbone) if isinstance(l, nn.MaxPool2d)]
        _, _, maxpool3_pos, maxpool4_pos, _ = pool_pos

        # Fix pool3 ceil_mode → output becomes 38*38 instead of 37*37
        backbone[maxpool3_pos].ceil_mode = True

        # Learnable per-channel L2-norm scale (init = 20, as in paper Sec. 3.1)
        self.scale_weight = nn.Parameter(torch.ones(512) * 20)

        # conv1_1 … conv4_3  (indices 0 … maxpool4_pos - 1, i.e. 0…22)
        self.features = nn.Sequential(*backbone[:maxpool4_pos])

        # Modified pool5 + atrous FC6 + FC7
        fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),               # pool5 (stride=1)
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),     # FC6 → atrous conv
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),                           # FC7 → 1*1 conv
            nn.ReLU(inplace=True),
        )
        _xavier_init(fc)

        self.extra = nn.ModuleList([
            # Block 0: pool4 + conv5_1/2/3  +  modified pool5 + FC6 + FC7
            # backbone[maxpool4_pos:-1] = indices 23..29 = pool4, conv5_1..3 with relus
            nn.Sequential(*backbone[maxpool4_pos:-1], fc),

            # Block 1 - conv8_2:  1024 → 256 → 512 (stride 2)
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
                nn.ReLU(inplace=True),
            ),

            # Block 2 - conv9_2:   512 → 128 → 256 (stride 2)
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
                nn.ReLU(inplace=True),
            ),

            # Block 3 - conv10_2:  256 → 128 → 256
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(inplace=True),
            ),

            # Block 4 - conv11_2:  256 → 128 → 256
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(inplace=True),
            ),
        ])

        # Xavier init for the newly added conv layers (blocks 1-4)
        for block in list(self.extra)[1:]:
            _xavier_init(block)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # ── conv4_3 with L2 norm + learnable rescale (paper Sec. 3.1) ────
        x = self.features(x)
        rescaled = self.scale_weight.view(1, -1, 1, 1) * F.normalize(x, p=2, dim=1)
        outputs = [rescaled]

        # ── Extra feature maps ────────────────────────────────────────────
        for block in self.extra:
            x = block(x)
            outputs.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(outputs)])


def _build_vgg_extractor(
    pretrained: bool = True,
    trainable_layers: int = 3,
    weights_path: Optional[str] = None,
) -> SSDFeatureExtractorVGG:
    if pretrained and weights_path is not None:
        # Load from local file – no internet required
        backbone = vgg16(weights=None)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        backbone.features.load_state_dict(state)
    else:
        weights = VGG16_Weights.IMAGENET1K_FEATURES if pretrained else None
        backbone = vgg16(weights=weights)
    backbone = backbone.features

    # Stage start indices: [0 (conv1), 4 (pool1→conv2), 9 (pool2→conv3),
    #                       16 (pool3→conv4), 23 (pool4→conv5)]
    stage_starts = [0] + [
        i for i, b in enumerate(backbone) if isinstance(b, nn.MaxPool2d)
    ][:-1]   # drop pool5 (index 30) since it's moved inside the extractor
    num_stages = len(stage_starts)   # 5
    if pretrained:
        trainable_layers = max(0, min(trainable_layers, num_stages))
        freeze_before = (
            len(backbone) if trainable_layers == 0
            else stage_starts[num_stages - trainable_layers]
        )
        for layer in backbone[:freeze_before]:
            for param in layer.parameters():
                param.requires_grad_(False)

    return SSDFeatureExtractorVGG(backbone)


class SSDScoringHead(nn.Module):

    def __init__(self, module_list: nn.ModuleList, num_columns: int):
        super().__init__()
        self.module_list = module_list
        self.num_columns = num_columns

    def forward(self, feature_maps: List[Tensor]) -> Tensor:
        results: List[Tensor] = []
        for i, x in enumerate(feature_maps):
            out = self.module_list[i](x)              # (N, A·K, H, W)
            N, _, H, W = out.shape
            out = out.view(N, -1, self.num_columns, H, W)
            out = out.permute(0, 3, 4, 1, 2)         # (N, H, W, A, K)
            out = out.reshape(N, -1, self.num_columns)  # (N, H·W·A, K)
            results.append(out)
        return torch.cat(results, dim=1)              # (N, Σ H·W·A, K)


class SSDClassificationHead(SSDScoringHead):
    def __init__(
        self,
        in_channels: List[int],
        num_anchors: List[int],
        num_classes: int,
    ):
        layers = nn.ModuleList([
            nn.Conv2d(c, a * num_classes, kernel_size=3, padding=1)
            for c, a in zip(in_channels, num_anchors)
        ])
        _xavier_init(layers)
        super().__init__(layers, num_classes)


class SSDRegressionHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int]):
        layers = nn.ModuleList([
            nn.Conv2d(c, a * 4, kernel_size=3, padding=1)
            for c, a in zip(in_channels, num_anchors)
        ])
        _xavier_init(layers)
        super().__init__(layers, 4)


class SSDHead(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        num_anchors: List[int],
        num_classes: int,
    ):
        super().__init__()
        self.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head     = SSDRegressionHead(in_channels, num_anchors)

    def forward(self, feature_maps: List[Tensor]) -> Dict[str, Tensor]:
        return {
            "bbox_regression": self.regression_head(feature_maps),
            "cls_logits":      self.classification_head(feature_maps),
        }

class SSD(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        anchor_generator: DefaultBoxGenerator,
        num_classes: int,
        size: Tuple[int, int] = (300, 300),
        image_mean: Optional[List[float]] = None,
        image_std:  Optional[List[float]] = None,
        score_thresh:        float = 0.01,
        nms_thresh:          float = 0.45,
        detections_per_img:  int   = 200,
        iou_thresh:          float = 0.5,
        topk_candidates:     int   = 400,
        neg_to_pos_ratio:    float = 3.0,
    ):
        super().__init__()
        self.backbone         = backbone
        self.anchor_generator = anchor_generator
        self.num_classes      = num_classes
        self.size             = size
        self.image_mean       = image_mean or [0.485, 0.456, 0.406]
        self.image_std        = image_std  or [0.229, 0.224, 0.225]
        self.score_thresh     = score_thresh
        self.nms_thresh       = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates  = topk_candidates
        self.neg_to_pos_ratio = neg_to_pos_ratio

        self.box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.matcher   = SSDMatcher(iou_thresh)

        out_channels = backbone.out_channels
        num_anchors  = anchor_generator.num_anchors_per_location()
        self.head = SSDHead(out_channels, num_anchors, num_classes)


    def _preprocess(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[Tensor, List[Tuple[int, int]], Optional[List[Dict[str, Tensor]]]]:

        orig_sizes: List[Tuple[int, int]] = [
            (img.shape[-2], img.shape[-1]) for img in images
        ]
        target_h, target_w = self.size
        processed: List[Tensor] = []

        for img in images:
            # Resize
            img = F.interpolate(
                img.unsqueeze(0).float(),
                size=self.size, mode="bilinear", align_corners=False,
            ).squeeze(0)
            # Normalise
            mean = torch.as_tensor(
                self.image_mean, dtype=img.dtype, device=img.device
            ).view(-1, 1, 1)
            std = torch.as_tensor(
                self.image_std, dtype=img.dtype, device=img.device
            ).view(-1, 1, 1)
            processed.append((img - mean) / std)

        scaled_targets: Optional[List[Dict[str, Tensor]]] = None
        if targets is not None:
            scaled_targets = []
            for t, (oh, ow) in zip(targets, orig_sizes):
                boxes = t["boxes"].clone().float()
                boxes[:, [0, 2]] *= target_w / ow
                boxes[:, [1, 3]] *= target_h / oh
                scaled_targets.append({"boxes": boxes, "labels": t["labels"]})

        return torch.stack(processed), orig_sizes, scaled_targets

    def compute_loss(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor],
        anchors: Tensor,
        matched_idxs: List[Tensor],
    ) -> Dict[str, Tensor]:

        bbox_regression = head_outputs["bbox_regression"]   # [B, total_a, 4]
        cls_logits      = head_outputs["cls_logits"]        # [B, total_a, C]

        num_foreground = 0
        bbox_loss_list:   List[Tensor] = []
        cls_targets_list: List[Tensor] = []

        for t_i, box_reg_i, cls_log_i, match_i in zip(
            targets, bbox_regression, cls_logits, matched_idxs
        ):
            pos = match_i >= 0                                    # [total_a] bool

            # ── Localisation loss (positives only) ────────────────────────
            gt_boxes    = t_i["boxes"][match_i[pos]]              # [P, 4]
            pos_anchors = anchors[pos]                            # [P, 4]
            encoded     = self.box_coder.encode_single(gt_boxes, pos_anchors)

            bbox_loss_list.append(
                F.smooth_l1_loss(box_reg_i[pos], encoded, reduction="sum")
            )
            num_foreground += int(pos.sum().item())


            gt_cls = torch.zeros(
                cls_log_i.shape[0],
                dtype=t_i["labels"].dtype,
                device=t_i["labels"].device,
            )
            gt_cls[pos] = t_i["labels"][match_i[pos]]
            cls_targets_list.append(gt_cls)

        bbox_loss   = torch.stack(bbox_loss_list)      # [B]
        cls_targets = torch.stack(cls_targets_list)    # [B, total_a]

        num_classes = cls_logits.shape[-1]
        cls_loss = F.cross_entropy(
            cls_logits.view(-1, num_classes),
            cls_targets.view(-1),
            reduction="none",
        ).view(cls_targets.shape)                      # [B, total_a]

        pos_mask    = cls_targets > 0
        num_neg     = (self.neg_to_pos_ratio * pos_mask.sum(1, keepdim=True)).long()
        neg_loss    = cls_loss.clone()
        neg_loss[pos_mask] = -float("inf")             # exclude positives from ranking

        _, idx_desc = neg_loss.sort(1, descending=True)
        rank        = idx_desc.argsort(1)              # rank[i,j] = position of anchor j
        neg_mask    = rank < num_neg                   # top-K negatives per image

        N = max(1, num_foreground)
        return {
            "bbox_regression": bbox_loss.sum() / N,
            "classification":  (cls_loss[pos_mask].sum() + cls_loss[neg_mask].sum()) / N,
        }

    def postprocess_detections(
        self,
        head_outputs: Dict[str, Tensor],
        anchors: Tensor,
        orig_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:

        bbox_regression = head_outputs["bbox_regression"]            # [B, total_a, 4]
        pred_scores     = F.softmax(head_outputs["cls_logits"], dim=-1)  # [B, total_a, C]
        target_h, target_w = self.size

        detections: List[Dict[str, Tensor]] = []
        for boxes_i, scores_i, (oh, ow) in zip(
            bbox_regression, pred_scores, orig_sizes
        ):
            boxes_i = self.box_coder.decode_single(boxes_i, anchors)
            boxes_i = clip_boxes_to_image(boxes_i, (target_h, target_w))

            all_boxes:  List[Tensor] = []
            all_scores: List[Tensor] = []
            all_labels: List[Tensor] = []

            for cls_idx in range(1, self.num_classes):   # skip background (0)
                scores_cls = scores_i[:, cls_idx]

                keep = scores_cls > self.score_thresh
                scores_cls = scores_cls[keep]
                boxes_cls  = boxes_i[keep]

                if scores_cls.numel() > self.topk_candidates:
                    scores_cls, idx = scores_cls.topk(self.topk_candidates)
                    boxes_cls = boxes_cls[idx]

                all_boxes.append(boxes_cls)
                all_scores.append(scores_cls)
                all_labels.append(
                    torch.full_like(scores_cls, cls_idx, dtype=torch.int64)
                )

            image_boxes  = torch.cat(all_boxes,  dim=0)
            image_scores = torch.cat(all_scores, dim=0)
            image_labels = torch.cat(all_labels, dim=0)

            keep = batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]


            final_boxes = image_boxes[keep].clone()
            final_boxes[:, [0, 2]] *= ow / target_w
            final_boxes[:, [1, 3]] *= oh / target_h

            detections.append({
                "boxes":  final_boxes,
                "scores": image_scores[keep],
                "labels": image_labels[keep],
            })

        return detections

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ):

        if self.training and targets is None:
            raise ValueError("targets must be provided in training mode.")

        images_t, orig_sizes, scaled_targets = self._preprocess(images, targets)

        # ── Backbone: 6 feature maps ──────────────────────────────────────
        feature_dict = self.backbone(images_t)
        feature_maps: List[Tensor] = list(feature_dict.values())

        # ── Detection head ────────────────────────────────────────────────
        head_outputs = self.head(feature_maps)   # "bbox_regression", "cls_logits"

        # ── Default boxes (same for every image in the batch) ────────────
        anchors = self.anchor_generator(self.size, feature_maps)   # [total_a, 4]

        if self.training:
            assert scaled_targets is not None
            matched_idxs: List[Tensor] = []
            for t_i in scaled_targets:
                if t_i["boxes"].numel() == 0:
                    matched_idxs.append(
                        torch.full(
                            (anchors.shape[0],), -1,
                            dtype=torch.int64, device=anchors.device,
                        )
                    )
                else:
                    iou = box_iou(t_i["boxes"], anchors)   # [num_gt, total_a]
                    matched_idxs.append(self.matcher(iou))

            return self.compute_loss(scaled_targets, head_outputs, anchors, matched_idxs)

        return self.postprocess_detections(head_outputs, anchors, orig_sizes)

def ssd300_vgg16(
    num_classes: int,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    weights_path: Optional[str] = None,
    **kwargs,
) -> SSD:

    backbone = _build_vgg_extractor(
        pretrained=pretrained_backbone,
        trainable_layers=trainable_backbone_layers,
        weights_path=weights_path,
    )

    anchor_generator = DefaultBoxGenerator(
        aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
        steps=[8, 16, 32, 64, 100, 300],
    )

    defaults: Dict = {
        "image_mean": [0.48235, 0.45882, 0.40784],
        "image_std":  [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],
    }
    defaults.update(kwargs)

    return SSD(
        backbone,
        anchor_generator,
        num_classes,
        size=(300, 300),
        **defaults,
    )

