# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Adapted from sam2/sam2_image_predictor.py (Apache-2.0).
# https://github.com/facebookresearch/sam2

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image

from seg_lib.models.sam2.modelling.sam2_base import SAM2Base
from seg_lib.models.sam2.modelling.transforms import SAM2Transforms

logger = logging.getLogger(__name__)


class Sam2Predictor:
    def __init__(
        self,
        sam_model: SAM2Base,
        mask_threshold: float = 0.0,
        max_hole_area: float = 0.0,
        max_sprinkle_area: float = 0.0,
    ) -> None:
        """
        Cache a SAM 2 image embedding, then predict masks from prompts.

        Arguments:
          sam_model: SAM 2 or SAM 2.1 image model.
          mask_threshold: Threshold used to binarize mask logits. Defaults to 0.
          max_hole_area: Unused. Hole filling needs the upstream CUDA extension.
          max_sprinkle_area: Unused. Sprinkle removal needs that same extension.
        """
        self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )
        self.mask_threshold = mask_threshold
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        self.reset_image()

    @torch.no_grad()
    def set_image(self, image: Union[np.ndarray, Image]) -> None:
        """
        Embed one RGB image.

        A numpy image is HWC uint8. A PIL image is RGB.
        """
        self.reset_image()
        if isinstance(image, np.ndarray):
            logger.debug("For numpy array image, we assume (HxWxC) format")
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            width, height = image.size
            self._orig_hw = [(height, width)]
        else:
            raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)
        if input_image.shape[1] != 3:
            raise ValueError(
                f"input_image must be of size 1x3xHxW, got {tuple(input_image.shape)}"
            )

        backbone_out = self.model.forward_image(input_image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True

    @torch.no_grad()
    def set_image_batch(self, image_list: List[np.ndarray]) -> None:
        """Embed a list of RGB HWC numpy images."""
        self.reset_image()
        if not isinstance(image_list, list):
            raise TypeError("image_list must be a list of numpy images")
        self._orig_hw = []
        for image in image_list:
            if not isinstance(image, np.ndarray):
                raise TypeError("Images are expected to be numpy arrays in HWC format")
            self._orig_hw.append(image.shape[:2])

        img_batch = self._transforms.forward_batch(image_list).to(self.device)
        batch_size = img_batch.shape[0]
        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        self._is_batch = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the image set by `set_image`.

        Returns masks (CxHxW), IoU scores (C,), and low-resolution logits (Cx256x256).
        """
        if not self._is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        mask_input, coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )
        masks, iou_predictions, low_res_masks = self._predict(
            coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
        )
        return (
            masks.squeeze(0).float().detach().cpu().numpy(),
            iou_predictions.squeeze(0).float().detach().cpu().numpy(),
            low_res_masks.squeeze(0).float().detach().cpu().numpy(),
        )

    def predict_batch(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        box_batch: List[np.ndarray] = None,
        mask_input_batch: List[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords: bool = True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Predict masks for a batch embedded with `set_image_batch`."""
        if not self._is_batch:
            raise RuntimeError("predict_batch requires set_image_batch.")
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before mask prediction."
            )

        all_masks, all_ious, all_low_res = [], [], []
        num_images = len(self._features["image_embed"])
        for img_idx in range(num_images):
            point_coords = None if point_coords_batch is None else point_coords_batch[img_idx]
            point_labels = None if point_labels_batch is None else point_labels_batch[img_idx]
            box = None if box_batch is None else box_batch[img_idx]
            mask_input = None if mask_input_batch is None else mask_input_batch[img_idx]
            mask_input, coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )
            masks, iou_predictions, low_res_masks = self._predict(
                coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
            )
            all_masks.append(masks.squeeze(0).float().detach().cpu().numpy())
            all_ious.append(iou_predictions.squeeze(0).float().detach().cpu().numpy())
            all_low_res.append(low_res_masks.squeeze(0).float().detach().cpu().numpy())
        return all_masks, all_ious, all_low_res

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):
        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            if point_labels is None:
                raise ValueError("point_labels must be supplied if point_coords is supplied.")
            point_coords = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
        if mask_logits is not None:
            mask_input = torch.as_tensor(mask_logits, dtype=torch.float, device=self.device)
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    @torch.no_grad()
    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        concat_points = None if point_coords is None else (point_coords, point_labels)
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            if concat_points is not None:
                concat_points = (
                    torch.cat([box_coords, concat_points[0]], dim=1),
                    torch.cat([box_labels, concat_points[1]], dim=1),
                )
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )
        batched_mode = concat_points is not None and concat_points[0].shape[0] > 1
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        masks = self._transforms.postprocess_masks(low_res_masks, self._orig_hw[img_idx])
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold
        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        if not self._is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) to generate an embedding.")
        return self._features["image_embed"]

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Drop the cached image embedding."""
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False

    def reset_predictor(self) -> None:
        """Alias of `reset_image`, matching the upstream predictor name."""
        self.reset_image()
