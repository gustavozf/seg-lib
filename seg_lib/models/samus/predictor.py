
import warnings
from copy import deepcopy
from typing import Tuple, Optional

import cv2
import torch
import numpy as np
from torch.nn import functional as F

from seg_lib.dataloaders.image_ops import (
    get_pad_offsets, resize_w_pad, to_grayscale
)
from seg_lib.io.image import read_img
from seg_lib.models.samus.modeling.samus import Samus
from seg_lib.models.sam.transforms import ResizeLongestSide
from seg_lib.models.sam import SamPredictor

class SamusResizeLongestSide(ResizeLongestSide):
    '''Pads the image to a centered square and resizes it to `target_length`,
       matching `resize_w_pad`, which is used by the training datasets.'''
    def apply_image(self, image):
        target_size = (self.target_length, self.target_length)
        image = resize_w_pad(image, target_size, interpolation=cv2.INTER_LINEAR)
        return image.astype('uint8')

    def get_padding(self, original_size: Tuple[int, ...]):
        diff, x_pos, y_pos = get_pad_offsets(*original_size[:2])
        return self.target_length / diff, x_pos, y_pos

    def apply_coords(
            self, coords: np.ndarray, original_size: Tuple[int, ...]
        ) -> np.ndarray:
        scale, x_pos, y_pos = self.get_padding(original_size)
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = (coords[..., 0] + x_pos) * scale
        coords[..., 1] = (coords[..., 1] + y_pos) * scale
        return coords

    def apply_coords_torch(
            self, coords: torch.Tensor, original_size: Tuple[int, ...]
        ) -> torch.Tensor:
        scale, x_pos, y_pos = self.get_padding(original_size)
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = (coords[..., 0] + x_pos) * scale
        coords[..., 1] = (coords[..., 1] + y_pos) * scale
        return coords

class SamusPredictor(SamPredictor):
    def __init__(self, sam_model: Samus) -> None:
        self.model = sam_model
        self.mask_threshold = sam_model.mask_threshold
        self.transform = SamusResizeLongestSide(256)
        self.reset_image()

    def set_image(
        self,
        image: np.ndarray | str,
        image_format: str = "RGB",
    ) -> None:
        if isinstance(image, str):
            image = read_img(image)
            image_format = 'RGB'

        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # SAMUS expects the inputs to be a 3-channel grayscale image
        image = to_grayscale(image)
        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image = torch.as_tensor(input_image, device=self.device)
        input_image = input_image.permute(2, 0, 1).contiguous()[None, :, :, :]

        return self.set_torch_image(input_image, image.shape[:2])

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with SamusResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        img_size = self.model.image_encoder.img_size
        if not (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] in (1, 3)
            and max(*transformed_image.shape[2:]) == img_size
        ):
            raise ValueError(
                "`set_torch_image` input must be BCHW with long side "
                f"{img_size}, but found {transformed_image.shape} instead."
            )
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True
        return input_image

    def postprocess_masks(self, low_res_masks: torch.Tensor) -> torch.Tensor:
        '''Upscales the masks and removes the centered square padding.'''
        h, w = self.original_size[:2]
        diff, x_pos, y_pos = get_pad_offsets(h, w)
        masks = F.interpolate(
            low_res_masks, (diff, diff), mode="bilinear", align_corners=False
        )
        return masks[..., y_pos:y_pos+h, x_pos:x_pos+w]

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_image_set:
            raise RuntimeError(
                'An image must be set with .set_image(...) '
                'before mask prediction.'
            )
        if multimask_output:
            warnings.warn(
                '`multimask_output` not supported by SAMUS. '
                'Returning masks as `multimask_output = False`.'
            )

        points = (
            (point_coords, point_labels) if point_coords is not None else None
        )
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points, boxes=boxes, masks=mask_input
        )
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        masks = self.postprocess_masks(low_res_masks)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks
