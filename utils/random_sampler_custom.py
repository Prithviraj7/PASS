import torch.utils.data
import numpy as np


class RandomIntIter:
    def __init__(self, high):
        self.high = high

    def __iter__(self):
        return self

    def __next__(self):
        # by keeing the randomness in numpy everything relies on the same seed
        # return torch.randint(high=self.high, size=(1,), dtype=torch.int64)[0]
        return np.random.randint(low=0, high=self.high, size=(1,), dtype=np.int64)[0]


class RandomWeightedIntIter:
    def __init__(self, weights):
        self.weights = np.array(weights)
        self.weights = self.weights / np.sum(self.weights)
        self.buffer_size = 100000
        self.buffer_idx = 0
        self.buffer = self._init_buffer()

    def _init_buffer(self):
        # pytorch's multinomial is very slow, see https://github.com/pytorch/pytorch/issues/11931
        # buffer = torch.multinomial(self.weights, self.buffer_size, False).tolist()
        buffer = np.random.choice(np.arange(len(self.weights)), self.buffer_size, replace=False, p=self.weights)
        return buffer

    def __iter__(self):
        return self

    def __next__(self):
        if self.buffer_idx < self.buffer_size:
            self.buffer_idx += 1
        else:
            self.buffer = self._init_buffer()
            self.buffer_idx = 1
        return self.buffer[self.buffer_idx - 1]


class RandomSamplerInfinite(torch.utils.data.Sampler):
    r"""Random sampler but effectively never runs out (actually reports size of 1e100 but thats too big to matter)

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        super(RandomSamplerInfinite, self).__init__(data_source)
        self.data_source = data_source

    @property
    def num_samples(self):
        return int(1e100)

    def __iter__(self):
        return RandomIntIter(len(self.data_source))

    def __len__(self):
        return self.num_samples


class RandomWeightedSamplerInfinite(torch.utils.data.Sampler):
    r"""Random weighted sampler but effectively never runs out (actually reports size of 1e100 but thats too big to
    matter)

    Arguments:
        data_source (Dataset): dataset to sample from
        weights (list): sample weights (probabilities of sampling them, don't need to be normalized)
    """

    def __init__(self, data_source, weights):
        super(RandomWeightedSamplerInfinite, self).__init__(data_source)
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.data_source = data_source

    @property
    def num_samples(self):
        return int(1e100)

    def __iter__(self):
        return RandomWeightedIntIter(self.weights)

    def __len__(self):
        return self.num_samples
