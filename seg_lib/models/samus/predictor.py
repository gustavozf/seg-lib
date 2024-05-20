
import warnings
from typing import Tuple, Optional

import cv2
import torch
import numpy as np

from seg_lib.models.samus.modeling.samus import Samus
from seg_lib.models.sam.transforms import ResizeLongestSide
from seg_lib.models.sam import SamPredictor

class SamusResizeLongestSide(ResizeLongestSide):
    def apply_image(self, image):
        target_size = (self.target_length, self.target_length)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        return image.astype('uint8')

class SamusPredictor(SamPredictor):
    def __init__(self, sam_model: Samus) -> None:
        self.model = sam_model
        self.transform = SamusResizeLongestSide(256)
        self.reset_image()

    def to_grayscale(self, image: np.ndarray):
        gs_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        image = np.zeros((*gs_image.shape, 3), dtype=np.uint8)
        image[:, :, 0] = gs_image
        image[:, :, 1] = gs_image
        image[:, :, 2] = gs_image
        del gs_image

        return image

    def set_image(
        self,
        image: np.ndarray | str,
        image_format: str = "RGB",
    ) -> None:
        if isinstance(image, str):
            image = cv2.imread(image, 1)[:, :, ::-1](image)
            image_format = 'RGB'

        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # SAMUS expects the inputs to be a 3-channel grayscale image
        image = self.to_grayscale(image)
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
            1x3xHxW, which has been transformed with ResizeLongestSide.
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
        if multimask_output:
            warnings.warn(
                '`multimask_output` not supported by SAMUS. '
                'Returning masks as `multimask_output = False`.'
            )

        return super().predict_torch(
            point_coords,
            point_labels,
            boxes=boxes,
            mask_input=mask_input,
            multimask_output=False,
            return_logits=return_logits,
        )
