from typing import Sequence
from torch import cat, multinomial, as_tensor, double as torch_double
from torch.utils.data.sampler import Sampler
import torch

class LabelWeightedSampler(Sampler[int]):

    label_weights: Sequence[float]
    klass_indices: Sequence[Sequence[int]]
    num_samples: int

    def __init__(self, label_weights: Sequence[float], labels: Sequence[int], num_samples: int, replacement: bool = True, generator=None) -> None:
        """

        :param label_weights: list(len=num_classes)[float], weights for each class.
        :param labels: list(len=dataset_len)[int], labels of a dataset.
        :param num_samples: number of samples.
        """
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={num_samples}")
        if not isinstance(replacement, bool):
            raise ValueError(f"replacement should be a boolean value, but got replacement={replacement}")

        super(LabelWeightedSampler, self).__init__(None)

        self.label_weights = torch.as_tensor(label_weights, dtype=torch.float32)
        self.labels        = torch.as_tensor(labels, dtype=torch.int)
        self.num_samples   = num_samples
        self.n_klass       = len(label_weights)
        # list of tensor.
        self.klass_indices = [(self.labels == i_klass).nonzero(as_tuple=True)[0]
                              for i_klass in range(self.n_klass)]
        self.replacement = replacement
        self.generator = generator

    def __iter__(self):
        sample_labels = torch.multinomial(self.label_weights,
                                          num_samples=self.num_samples,
                                          replacement=self.replacement,
                                          generator=self.generator)
        sample_indices = torch.empty_like(sample_labels)
        for i_klass in range(self.n_klass):
            left_inds  = (sample_labels == i_klass).nonzero(as_tuple=True)[0]
            right_inds = torch.randint(len(self.klass_indices[i_klass]), size=(len(left_inds), ))
            sample_indices[left_inds] = self.klass_indices[i_klass][right_inds]

        return iter(sample_indices.tolist())

    def __len__(self):
        return self.num_samples