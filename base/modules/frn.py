"""
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
from functools import partial
import torch
import torch.nn as nn


class TLU(nn.Module):
    """ Thresholded Linear Unit """
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.tau = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        return torch.max(x, self.tau)

    def extra_repr(self):
        return 'num_features={}'.format(self.num_features)


# NOTE generalized version
class FilterResponseNorm(nn.Module):
    """ Filter Response Normalization """
    def __init__(self, num_features, ndim, eps=None, learnable_eps=False):
        """
        Args:
            num_features
            ndim
            eps: if None is given, use the paper value as default.
                from paper, fixed_eps=1e-6 and learnable_eps_init=1e-4.
            learnable_eps: turn eps to learnable parameter, which is recommended on
                fully-connected or 1x1 activation map.
        """
        super().__init__()
        if eps is None:
            if learnable_eps:
                eps = 1e-4
            else:
                eps = 1e-6

        self.num_features = num_features
        self.init_eps = eps
        self.learnable_eps = learnable_eps
        self.ndim = ndim

        self.mean_dims = list(range(2, 2+ndim))

        self.weight = nn.Parameter(torch.ones([1, num_features] + [1]*ndim))
        self.bias = nn.Parameter(torch.zeros([1, num_features] + [1]*ndim))
        if learnable_eps:
            self.eps = nn.Parameter(torch.as_tensor(eps))
        else:
            self.register_buffer('eps', torch.as_tensor(eps))

    def forward(self, x):
        # normalize
        nu2 = x.pow(2).mean(self.mean_dims, keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        # modulation
        x = x * self.weight + self.bias

        return x

    def extra_repr(self):
        return 'num_features={}, init_eps={}, ndim={}'.format(
                self.num_features, self.init_eps, self.ndim)


FilterResponseNorm1d = partial(FilterResponseNorm, ndim=1, learnable_eps=True)
FilterResponseNorm2d = partial(FilterResponseNorm, ndim=2)
