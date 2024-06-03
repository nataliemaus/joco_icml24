from abc import ABC, abstractmethod
from typing import Any

import torch
from botorch.acquisition.objective import IdentityMCObjective
from botorch.generation.utils import _flip_sub_unique
from botorch.posteriors.gpytorch import GPyTorchPosterior
from torch import Tensor
from torch.nn import Module
from joco.utils.get_random_projection import get_random_projection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Code copied form botorch MaxPosteriorSampling class and edited for JoCo
# https://botorch.org/api/_modules/botorch/generation/sampling.html
class SamplingStrategy(Module, ABC):
    r"""Abstract base class for sampling-based generation strategies."""

    @abstractmethod
    def forward(self, X: Tensor, num_samples: int = 1, **kwargs: Any) -> Tensor:
        r"""Sample according to the SamplingStrategy.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension).
            num_samples: The number of samples to draw.
            kwargs: Additional implementation-specific kwargs.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """

        pass  # pragma: no cover


class MaxPosteriorSamplingJoCo(SamplingStrategy):
    r"""Sample from a set of points according to their max posterior value.

    Example:
        >>> MPS = MaxPosteriorSampling(model)  # model w/ feature dim d=3
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = MPS(X, num_samples=5)
    """

    def __init__(
        self,
        models_list=None,
        objective=None,
        replacement=True,
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The objective. Defaults to `IdentityMCObjective()`.
            replacement: If True, sample with replacement.
        """
        super().__init__()
        self.models_list = models_list
        if objective is None:
            objective = IdentityMCObjective()
        self.objective = objective
        self.replacement = replacement

    def forward(
        self,
        X: Tensor,
        num_samples: int = 1,
        observation_noise: bool = False,
        propegate_uncertainty_y: bool = True,
        propegate_uncertainty_x: bool = True,
        rand_proj_baseline: bool = False,
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """
        if rand_proj_baseline:
            compressed_x_dim = self.models_list[-1].feature_extractor.output_dim
            X_compressed = get_random_projection(X, target_dim=compressed_x_dim).to(device)

        y_model = self.models_list[0]
        x_models_list = self.models_list[1:]
        all_y_samples = []  # compresed y samples
        for model in x_models_list:
            if rand_proj_baseline:
                model.eval()
                model.likelihood.eval()
                # don't use feature extractor bc X has already been dim reduced w random proj
                dist = model.likelihood(model(X_compressed, use_feature_extractor=False))
                posterior = GPyTorchPosterior(mvn=dist)
            else:
                posterior = model.posterior(X, observation_noise=observation_noise)
            if not propegate_uncertainty_x:
                y_samples = posterior.mean
            else:
                y_samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
            all_y_samples.append(y_samples)
        all_y_samples = torch.cat(
            all_y_samples, dim=-1
        )  # (bsz, N, compressed_y_dim) !!

        y_model.eval()  # make sure model is in eval mode
        y_model.likelihood.eval()
        # dist = y_model.likelihood(y_model.forward(X)) XXX MISTAKE XXX
        dist = y_model.likelihood(y_model.forward(all_y_samples))
        posterior = GPyTorchPosterior(mvn=dist)
        # Sample from posterior to get predicted score samples
        # NOTE: bsz === num_samples
        # NOTE: N == n_candidates (see turbo.py)
        if propegate_uncertainty_y:
            if propegate_uncertainty_x:
                n_samps = 1
            else:
                n_samps = num_samples
            samples = posterior.rsample(
                sample_shape=torch.Size([n_samps])
            )  # (1, bsz, N, 1) i.e. samples.shape == torch.Size([1, 10, 200, 1])
            samples = samples.squeeze(
                0
            )  # (bsz, N, 1 ), i.e. torch.Size([10, 200, 1])(N =)
        else:  # no uncertainty in y propagation
            samples = (
                posterior.mean
            )  # torch.Size([10, 200, 1]) (testing w/ N=200, in practice use 1k)
            if not propegate_uncertainty_x:
                # in this case we have samples == torch.Size([1000, 1]) = (N,1)
                samples = samples.unsqueeze(0).repeat(
                    (num_samples, 1, 1), 0
                )  # torch.Size([10, 1000, 1]) = (bsz, N, 1)
        # NOTE: NEED samples.shape == (bsz, N, 1 )
        obj = self.objective(samples, X=X)  # num_samples x batch_shape x N
        if self.replacement:
            # if we allow replacement then things are simple(r)
            idcs = torch.argmax(obj, dim=-1)
        else:
            # if we need to deduplicate we have to do some tensor acrobatics
            # first we get the indices associated w/ the num_samples top samples
            _, idcs_full = torch.topk(obj, num_samples, dim=-1)
            # generate some indices to smartly index into the lower triangle of
            # idcs_full (broadcasting across batch dimensions)
            ridx, cindx = torch.tril_indices(num_samples, num_samples)
            # pick the unique indices in order - since we look at the lower triangle
            # of the index matrix and we don't sort, this achieves deduplication
            sub_idcs = idcs_full[ridx, ..., cindx]
            if sub_idcs.ndim == 1:
                idcs = _flip_sub_unique(sub_idcs, num_samples)
            elif sub_idcs.ndim == 2:
                # TODO: Find a better way to do this
                n_b = sub_idcs.size(-1)
                idcs = torch.stack(
                    [_flip_sub_unique(sub_idcs[:, i], num_samples) for i in range(n_b)],
                    dim=-1,
                )
            else:
                # TODO: Find a general way to do this efficiently.
                raise NotImplementedError(
                    "MaxPosteriorSampling without replacement for more than a single "
                    "batch dimension is not yet implemented."
                )
        # idcs is num_samples x batch_shape, to index into X we need to permute for it
        # to have shape batch_shape x num_samples
        if idcs.ndim > 1:
            idcs = idcs.permute(*range(1, idcs.ndim), 0)
        # in order to use gather, we need to repeat the index tensor d times
        idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
        # now if the model is batched batch_shape will not necessarily be the
        # batch_shape of X, so we expand X to the proper shape
        Xe = X.expand(*obj.shape[1:], X.size(-1))
        # finally we can gather along the N dimension
        return torch.gather(Xe, -2, idcs)  # means use the idcs to index dimension -2
