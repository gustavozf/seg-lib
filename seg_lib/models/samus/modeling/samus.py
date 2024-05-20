import warnings
from typing import List, Dict, Any

import torch

from seg_lib.models.sam.modeling import Sam
from seg_lib.models.samus.modeling.image_encoder import ImageEncoderViT
from seg_lib.models.samus.modeling.mask_decoder import MaskDecoder
from seg_lib.models.samus.modeling.prompt_encoder import PromptEncoder

def reshape_iou(iou_predictions):
	''' Reshape the output iou preds to drop an additional unused axis.'''
	iou_predictions = torch.stack(iou_predictions, dim=0)
	return iou_predictions.reshape(iou_predictions.shape[0], -1)

def reshape_masks(masks):
	''' Reshape the output masks to drop an additional unused axis.'''
	masks = torch.stack(masks, dim=0)
	m_shape = masks.shape
	return masks.reshape(m_shape[0], -1, m_shape[3], m_shape[4])

class Samus(Sam):
	def __init__(
		self,
		image_encoder: ImageEncoderViT,
		prompt_encoder: PromptEncoder,
		mask_decoder: MaskDecoder,
		pixel_mean: List[float] = [123.675, 116.28, 103.53],
		pixel_std: List[float] = [58.395, 57.12, 57.375],
	) -> None:
		'''
		SAM predicts object masks from an image and input prompts.

		Arguments:
		  image_encoder (ImageEncoderViT): The backbone used to encode the
			image into image embeddings that allow for efficient mask prediction.
		  prompt_encoder (PromptEncoder): Encodes various types of input prompts.
		  mask_decoder (MaskDecoder): Predicts masks from the image embeddings
			and encoded prompts.
		  pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
		  pixel_std (list(float)): Std values for normalizing pixels in the input image.
		'''
		super().__init__(
		  image_encoder, prompt_encoder, mask_decoder,
		  pixel_mean=pixel_mean, pixel_std=pixel_std
		)

		# freeze the prompt encoder
		for param in self.prompt_encoder.parameters():
			param.requires_grad = False
		# freeze the mask decoder
		for param in self.mask_decoder.parameters():
			param.requires_grad = False
		# partially freeze the image encoder
		for n, value in self.image_encoder.named_parameters():
			if ('cnn_embed' not in n 
				and 'post_pos_embed' not in n
				and 'Adapter' not in n
				and '2.attn.rel_pos' not in n
				and '5.attn.rel_pos' not in n
				and '8.attn.rel_pos' not in n
				and '11.attn.rel_pos' not in n
				and 'upneck' not in n):
				value.requires_grad = False

	def forward(
		self,
		batched_input: Dict[str, Any],
		multimask_output: bool):
		'''
			Inference used during training, i.e. batched dict input containing
			the following keys:
				- image: torch.Tensor
					input images, expected to be resized and normalized with
					SAM's mean and standard deviation (Z-Score)
				- point_coords: torch.Tensor
					point coordinates (prompt)
				- point_labels: torch.Tensor
					labels for each point coordinate
				- boxes: torch.Tensor
					bounding boxes (prompt)
				- mask_inputs: torch.Tensor
					intermediate masks (prompt)
					should have the same dimension as the image embedding!
				- original_size': Tuple[int]
					tuple containing the original size for reshaping
		'''
		if multimask_output:
			warnings.warn(
				'`multimask_output` not supported by SAMUS. '
				'Returning masks as `multimask_output = False`.'
			)

		image_embeddings = self.image_encoder(batched_input['image'])
		points = (
			batched_input.get('point_coords', None),
			batched_input.get('point_labels', None)
		)

		sparse_embeddings, dense_embeddings = self.prompt_encoder(
			points=None if None in points else points,
			boxes=batched_input.get('boxes', None),
			masks=batched_input.get('mask_inputs', None))
		low_res_masks, iou_predictions = self.mask_decoder(
			image_embeddings=image_embeddings,
			image_pe=self.prompt_encoder.get_dense_pe(),
			sparse_prompt_embeddings=sparse_embeddings,
			dense_prompt_embeddings=dense_embeddings,
			multimask_output=False)
		masks = self.postprocess_masks(
			low_res_masks,
			input_size=batched_input['image'].shape[-2:],
			original_size=batched_input['original_size'],
		)
		
		return {
			'masks': masks,
			'iou_predictions': iou_predictions,
			'low_res_logits': low_res_masks
		}
