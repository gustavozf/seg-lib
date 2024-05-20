from typing import Dict, Any

from seg_lib.models.sam.modeling import Sam

class SamBatchInf(Sam):    
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
        multimask_output=multimask_output)
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
