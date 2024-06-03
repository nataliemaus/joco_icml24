# Code copied from here and modified: https://github.com/pytorch/botorch/blob/main/botorch/acquisition/objective.py
from __future__ import annotations

import warnings
from typing import Optional, Union

import gpytorch
import torch

from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.models.deterministic import DeterministicModel
from botorch.models.model import Model

from botorch.optim.optimize import optimize_acqf
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling import IIDNormalSampler, SobolQMCNormalSampler
from botorch.sampling.base import MCSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor

warnings.filterwarnings("ignore")

NUM_OUTCOME_SAMPLES = 256


class LearnedObjective(MCAcquisitionObjective):
    r"""Learned preference objective constructed from a preference model.
    For input `samples`, it samples each individual sample again from the latent
    preference posterior distribution using `pref_model` and return the posterior mean.
    Example:
        >>> train_X = torch.rand(2, 2)
        >>> train_comps = torch.LongTensor([[0, 1]])
        >>> pref_model = PairwiseGP(train_X, train_comps)
        >>> learned_pref_obj = LearnedObjective(pref_model)
        >>> samples = sampler(posterior)
        >>> objective = learned_pref_obj(samples)
    """

    def __init__(
        self,
        pref_model: Model,
        sampler: Optional[MCSampler] = None,
    ):
        r"""
        Args:
            pref_model: A BoTorch model, which models the latent preference/utility
                function. Given an input tensor of size
                `sample_size x batch_shape x N x d`, its `posterior` method should
                return a `Posterior` object with single outcome representing the
                utility values of the input.
            sampler: Sampler for the preference model to account for uncertainty in
                preferece when calculating the objective; it's not the one used
                in MC acquisition functions. If None,
                it uses `IIDNormalSampler(sample_shape=torch.Size([1]))`.
        """
        super().__init__()
        self.pref_model = pref_model
        if isinstance(pref_model, DeterministicModel):
            assert sampler is None
            self.sampler = None
        else:
            if sampler is None:
                self.sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
            else:
                self.sampler = sampler

        self.pref_model.eval()  # make sure model is in eval mode
        self.pref_model.likelihood.eval()

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Sample each element of samples.
        Args:
            samples: A `sample_size x batch_shape x N x d`-dim Tensors of
                samples from a model posterior.
        Returns:
            A `(sample_size * num_samples) x batch_shape x N`-dim Tensor of
            objective values sampled from utility posterior using `pref_model`.
        """
        # post = self.pref_model.posterior(samples) # ORIGINAL CODE
        # REPLACEMENT CODE (Need to get pref_model / y_model posterior without DKL )
        dist = self.pref_model.likelihood(self.pref_model.forward(samples))
        post = GPyTorchPosterior(mvn=dist)
        # TODO: Modify repo so GP2 (pref_model) doesn't have the deep kernel inside of it,
        #       then we can just just import and use LearnedObjective directly rather than
        #       copying and modifying LearnedObjective

        if isinstance(self.pref_model, DeterministicModel):
            # return preference posterior mean
            return post.mean.squeeze(-1)
        else:
            # return preference posterior sample mean
            samples = self.sampler(post).squeeze(-1)  # == torch.Size([64, 256, 1])
            return samples.reshape(
                -1, *samples.shape[2:]
            )  # batch_shape x N, == torch.Size([16384, 1])


class qExpectedImprovementModelList(MCAcquisitionFunction):
    r"""MC-based batch Expected Improvement.
    This computes qEI by
    (1) sampling the joint posterior over q points
    (2) evaluating the improvement over the current best for each sample
    (3) maximizing over q
    (4) averaging over the samples
    `qEI(X) = E(max(max Y - best_f, 0)), Y ~ f(X), where X = (x_1,...,x_q)`
    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> best_f = train_Y.max()[0]
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qEI = qExpectedImprovement(model, best_f, sampler)
        >>> qei = qEI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""q-Expected Improvement.
        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed noiseless). Can be
                a `batch_shape`-shaped tensor, which in case of a batched model
                specifies potentially different values for each element of the batch.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending:  A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
                Concatenated into X upon forward call. Copied and set to have no
                gradient.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.register_buffer("best_f", torch.as_tensor(best_f, dtype=float))

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qExpectedImprovement on the candidate set `X`.
        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.
        Returns:
            A `batch_shape'`-dim Tensor of Expected Improvement values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        # JOCO modificaitons to get samples from model list
        all_y_samples = []
        for model in self.model.models:
            posterior = model.posterior(X)
            y_samples = posterior.rsample(
                sample_shape=torch.Size([NUM_OUTCOME_SAMPLES])
            )
            all_y_samples.append(y_samples)
        samples = torch.cat(all_y_samples, dim=-1)  # torch.Size([256, 256, 1, 2])
        # OG BOTORCH CODE:
        # posterior = self.model.posterior(
        #     X=X, posterior_transform=self.posterior_transform
        # )
        # samples = self.get_posterior_samples(posterior)

        obj = self.objective(samples, X=X)
        obj = (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
        q_ei = obj.max(dim=-1)[0].mean(dim=0)
        return q_ei


def get_ei_candidates_joco(
    models_list,
    batch_size,
    raw_samples,
    num_restarts,
    tr_lb,
    tr_ub,
    best_f,  # Y.max().cuda()
):
    # pref_model = fit_pref_model(train_Y, train_comps)
    # pref_model = models_list[0]
    y_model = models_list[0]  # pref model / Y to score model
    x_models_list = models_list[1:]
    sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([NUM_OUTCOME_SAMPLES])
    )  # NUM_OUTCOME_SAMPLES
    objective = LearnedObjective(pref_model=y_model, sampler=sampler)

    # outcome_model = None # X to Y
    # botorch.models.model_list_gp_regression.ModelListGP
    # outcome_model = botorch.models.model_list_gp_regression.ModelListGP(*x_models_list)
    outcome_model = gpytorch.models.IndependentModelList(*x_models_list)
    # EI BOTORCH: https://botorch.org/api/acquisition.html
    # acq_func = qNoisyExpectedImprovement(
    #     model=outcome_model,
    #     objective=objective,
    #     X_baseline=X_observed,
    #     sampler=sampler,
    # )
    acq_func = qExpectedImprovementModelList(
        model=outcome_model.cuda(),
        best_f=best_f,
        sampler=sampler,
        objective=objective,
    )
    X_next, _ = optimize_acqf(
        acq_function=acq_func,
        q=batch_size,
        bounds=torch.stack([tr_lb, tr_ub]).cuda(),
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        # options={"batch_limit": BATCH_LIMIT},
        sequential=True,
    )
    return X_next
