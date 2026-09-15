from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import nms


class AnchorFreeHead(nn.Module):
    """Single-scale, anchor-free detection head (a YOLOv1-style placeholder,
    not YOLO26's real head): one grid cell is responsible for a ground-truth
    box's center, and predicts objectness, per-class scores, and a box
    directly in normalized image-fraction units (no exp/anchor scaling, so
    training is simple and stable). Exists to give the plug-and-play system
    a working head today; a real multi-scale YOLO26 head is a drop-in
    replacement as long as it keeps this same `compute_loss`/`postprocess`
    contract.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        obj_weight: float = 1.0,
        noobj_weight: float = 0.5,
        coord_weight: float = 5.0,
        cls_weight: float = 1.0,
        conf_thresh: float = 0.25,
        nms_iou: float = 0.45,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_channels, 4 + 1 + num_classes, kernel_size=1)
        self.obj_weight = obj_weight
        self.noobj_weight = noobj_weight
        self.coord_weight = coord_weight
        self.cls_weight = cls_weight
        self.conf_thresh = conf_thresh
        self.nms_iou = nms_iou

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)

    def _decode(
        self, raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, grid_h, grid_w = raw.shape
        box = torch.sigmoid(raw[:, 0:4])
        obj = torch.sigmoid(raw[:, 4])
        cls = torch.sigmoid(raw[:, 5:])

        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h, device=raw.device),
            torch.arange(grid_w, device=raw.device),
            indexing="ij",
        )
        cx = (box[:, 0] + grid_x) / grid_w
        cy = (box[:, 1] + grid_y) / grid_h
        w = box[:, 2]
        h = box[:, 3]
        return cx, cy, w, h, obj, cls

    def compute_loss(
        self, raw: torch.Tensor, targets: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        batch_size, _, grid_h, grid_w = raw.shape
        cx, cy, w, h, obj, cls = self._decode(raw)

        obj_target = torch.zeros(batch_size, grid_h, grid_w, device=raw.device)
        box_target = torch.zeros(batch_size, grid_h, grid_w, 4, device=raw.device)
        cls_target = torch.zeros(batch_size, self.num_classes, grid_h, grid_w, device=raw.device)
        pos_mask = torch.zeros(batch_size, grid_h, grid_w, dtype=torch.bool, device=raw.device)

        for image_index, target in enumerate(targets):
            for box_xy_wh, label in zip(target["boxes"], target["labels"]):
                gx, gy, gw, gh = box_xy_wh.tolist()
                col = min(int(gx * grid_w), grid_w - 1)
                row = min(int(gy * grid_h), grid_h - 1)
                obj_target[image_index, row, col] = 1.0
                box_target[image_index, row, col] = torch.tensor(
                    [gx, gy, gw, gh], device=raw.device
                )
                cls_target[image_index, int(label), row, col] = 1.0
                pos_mask[image_index, row, col] = True

        pred_boxes = torch.stack([cx, cy, w, h], dim=-1)
        neg_mask = ~pos_mask
        if pos_mask.any():
            coord_loss = F.mse_loss(pred_boxes[pos_mask], box_target[pos_mask])
            cls_pred = cls.permute(0, 2, 3, 1)[pos_mask]
            cls_gt = cls_target.permute(0, 2, 3, 1)[pos_mask]
            cls_loss = F.binary_cross_entropy(cls_pred, cls_gt)
            obj_pos_loss = F.binary_cross_entropy(obj[pos_mask], obj_target[pos_mask])
        else:
            coord_loss = raw.new_zeros(())
            cls_loss = raw.new_zeros(())
            obj_pos_loss = raw.new_zeros(())
        obj_neg_loss = (
            F.binary_cross_entropy(obj[neg_mask], obj_target[neg_mask])
            if neg_mask.any()
            else raw.new_zeros(())
        )

        return {
            "loss_coord": self.coord_weight * coord_loss,
            "loss_obj": self.obj_weight * obj_pos_loss + self.noobj_weight * obj_neg_loss,
            "loss_cls": self.cls_weight * cls_loss,
        }

    def postprocess(
        self, raw: torch.Tensor, img_size: tuple[int, int]
    ) -> list[dict[str, torch.Tensor]]:
        img_h, img_w = img_size
        cx, cy, w, h, obj, cls = self._decode(raw)
        batch_size = raw.shape[0]

        results = []
        for image_index in range(batch_size):
            scores_per_class = obj[image_index].unsqueeze(0) * cls[image_index]
            best_score, best_label = scores_per_class.max(dim=0)
            keep = best_score > self.conf_thresh

            if not keep.any():
                results.append(
                    {
                        "boxes": raw.new_zeros((0, 4)),
                        "scores": raw.new_zeros((0,)),
                        "labels": torch.zeros((0,), dtype=torch.long, device=raw.device),
                    }
                )
                continue

            cx_k, cy_k, w_k, h_k = cx[image_index][keep], cy[image_index][keep], w[image_index][keep], h[image_index][keep]
            x1 = (cx_k - w_k / 2) * img_w
            y1 = (cy_k - h_k / 2) * img_h
            x2 = (cx_k + w_k / 2) * img_w
            y2 = (cy_k + h_k / 2) * img_h
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
            scores = best_score[keep]
            labels = best_label[keep]

            keep_idx = nms(boxes_xyxy, scores, self.nms_iou)
            results.append(
                {"boxes": boxes_xyxy[keep_idx], "scores": scores[keep_idx], "labels": labels[keep_idx]}
            )
        return results
