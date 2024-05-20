import numpy as np
from skimage.measure import label

class TrainPromptSampler:
    INDEX_SAMPLERS = {'fixed', 'random'}

    def __init__(self, min_blob_count: int = 20, max_num_prompts: int = 1):
        self.min_blob_count = min_blob_count
        self.max_num_prompts = max_num_prompts

    def __int_split(self, input_int: int, num_bins: int):
        quotient, remainder = divmod(input_int, num_bins)
        return [quotient + 1] * remainder + [quotient] * (num_bins - remainder)

    def fixed_index_sampler(self, num_indices, num_prompts):
        return np.linspace(0, num_indices, num=num_prompts+2, dtype=int)[1:-1]

    def random_index_sampler(self, num_indices, num_prompts):
        return np.random.randint(num_indices, size=num_prompts)

    def index_sampler_selector(self, index_sampler: str):
        if index_sampler not in self.INDEX_SAMPLERS:
            raise ValueError(
                f'Index Sampler "{index_sampler}" not supported. '
                f'Available choices: {self.INDEX_SAMPLERS}'
            )
        
        if index_sampler == 'random':
            return self.random_index_sampler

        return self.fixed_index_sampler

    def __return_background(self, sampler, mask, class_id):
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
        pt_index = sampler(indices.shape[0], self.max_num_prompts)
        return indices[pt_index], np.repeat(0, self.max_num_prompts)

    def __call__(
            self, mask: np.ndarray,
            class_id: int = 1, index_sampler: str = 'fixed'
        ):
        """
            This functions aims to generate `N` prompts from each image conexed
            component. It was developed considering that the random points
            could be generated inside a single component. Such action could
            potentially misguide SAM(US) during training.

            Downsides of the approach: the performance is limited by the
            labelling method (currently `skimage.measure.label`).

            @author: Gustavo Zanoni
            @date : 2023-02-03
        """
        sampler = self.index_sampler_selector(index_sampler)
        mask[mask!=class_id] = 0
        # identify the conexed components of an image
        comp_mask, num_comp = label(mask, return_num=True)

        if num_comp < 1:
            return self.__return_background(sampler, mask, class_id)

        # get only the components that have enough pixels
        valid_comp_idx = [
            np.argwhere(comp_mask == i+1)[:, [1,0]] for i in range(num_comp)
        ]
        valid_comp_idx = [
            idx for idx in valid_comp_idx if idx.shape[0] > self.min_blob_count
        ]
        
        if len(valid_comp_idx) == 0:
            return self.__return_background(sampler, mask, class_id)
        
        # if the quantity of components is greater than the maximum number of
        # prompts, we select the larger ones
        if len(valid_comp_idx) > self.max_num_prompts:
            valid_comp_idx = sorted(
                valid_comp_idx, key=lambda idx : idx.shape[0], reverse=True
            )
            valid_comp_idx = valid_comp_idx[:self.max_num_prompts]

        start_idx = 0
        prompts = np.zeros((self.max_num_prompts, 2))
        prompts_labels = np.zeros((self.max_num_prompts))
        prompts_per_image = self.__int_split(
            self.max_num_prompts, len(valid_comp_idx)
        )
        for num_prompts, indices in zip(prompts_per_image, valid_comp_idx):
            end_idx = start_idx + num_prompts

            pt_index = sampler(indices.shape[0], num_prompts)
            prompts[start_idx:end_idx, :] = indices[pt_index]
            prompts_labels[start_idx:end_idx] = np.repeat(1, num_prompts)
            start_idx = end_idx
            del indices, pt_index, num_prompts

        return prompts, prompts_labels