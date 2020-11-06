import torch
from torch.utils.data import Sampler


class DynamicSampler(Sampler):
    """ Designed after the default `RandomSampler` sampler found in the torch.util.data.sampler file """

    def __init__(self, dataset, replacement=False, num_samples=None):
        self.dataset = dataset
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

        print("It made it here")

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.dataset)
        return self._num_samples

    def __iter__(self):
        n = len(self.dataset)
        if self.replacement:
            rand_tensor = torch.randint(high=n, size=(self._num_samples,), dtype=torch.int64)
            return iter(rand_tensor.tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self._num_samples

    @staticmethod
    def compose_events(event1, event2):
        """ Insert the compositing algorithm here """
        pass
