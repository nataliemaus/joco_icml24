import math
from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.generation import MaxPosteriorSampling
from botorch.optim import optimize_acqf

from joco.utils.ei_joco import get_ei_candidates_joco
from joco.utils.max_posterior_sampling_joco import MaxPosteriorSamplingJoCo
from torch.quasirandom import SobolEngine

RAW_SAMPLES = 256


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 32
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    restart_triggered: bool = False
    best_constraint_values: torch.Tensor = (
        torch.ones(
            2,
        )
        * torch.inf
    )


def update_tr_length(state):
    # Update the length of the trust region according to
    # success and failure counters
    # (Just as in original TuRBO paper)
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:  # Restart when trust region becomes too small
        state.restart_triggered = True

    return state


def update_state_constrained(state, Y_next, C_next):
    """Method used to update the TuRBO state after each
    step of optimization.

    Success and failure counters are updated accoding to
    the objective values (Y_next) and constraint values (C_next)
    of the batch of candidate points evaluated on the optimization step.

    As in the original TuRBO paper, a success is counted whenver
    any one of the new candidate points imporves upon the incumbent
    best point. The key difference for SCBO is that we only compare points
    by their objective values when both points are valid (meet all constraints).
    If exactly one of the two points beinc compared voliates a constraint, the
    other valid point is automatically considered to be better. If both points
    violate some constraints, we compare them inated by their constraint values.
    The better point in this case is the one with minimum total constraint violation
    (the minimum sum over constraint values)"""

    # Determine which candidates meet the constraints (are valid)
    bool_tensor = C_next <= 0
    bool_tensor = torch.all(bool_tensor, dim=-1)
    Valid_Y_next = Y_next[bool_tensor]
    Valid_C_next = C_next[bool_tensor]
    if Valid_Y_next.numel() == 0:  # if none of the candidates are valid
        # pick the point with minimum violation
        sum_violation = C_next.sum(dim=-1)
        min_violation = sum_violation.min()
        # if the minimum voilation candidate is smaller than the violation of the incumbent
        if min_violation < state.best_constraint_values.sum():
            # count a success and update the current best point and constraint values
            state.success_counter += 1
            state.failure_counter = 0
            # new best is min violator
            state.best_value = Y_next[sum_violation.argmin()].item()
            state.best_constraint_values = C_next[sum_violation.argmin()]
        else:
            # otherwise, count a failure
            state.success_counter = 0
            state.failure_counter += 1
    else:  # if at least one valid candidate was suggested,
        # throw out all invalid candidates
        # (a valid candidate is always better than an invalid one)

        # Case 1: if best valid candidate found has a higher obj value that incumbent best
        # count a success, the obj valuse has been improved
        imporved_obj = max(Valid_Y_next) > state.best_value + 1e-3 * math.fabs(
            state.best_value
        )
        # Case 2: if incumbent best violates constraints
        # count a success, we now have suggested a point which is valid and therfore better
        obtained_validity = torch.all(state.best_constraint_values > 0)
        if imporved_obj or obtained_validity:  # If Case 1 or Case 2
            # count a success and update the best value and constraint values
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = max(Valid_Y_next).item()
            state.best_constraint_values = Valid_C_next[Valid_Y_next.argmax()]
        else:
            # otherwise, count a fialure
            state.success_counter = 0
            state.failure_counter += 1

    # Finally, update the length of the trust region according to the
    # updated success and failure counts
    state = update_tr_length(state)

    return state


def update_state(state, Y_next, C_next):
    if C_next is None:
        return update_state_unconstrained(state, Y_next)
    else:
        return update_state_constrained(state, Y_next, C_next)


def update_state_unconstrained(state, Y_next):
    if Y_next.max().item() > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, Y_next.max().item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    device,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=RAW_SAMPLES,
    acqf="ts",  # "ei" or "ts"
    dtype=torch.float32,
    absolute_bounds=None,
    constraint_model_list=None,
    use_turbo=True,
):
    if constraint_model_list is not None:
        assert acqf == "ts"  # SCBO only works with ts
        constrained = True
    else:
        constrained = False
    assert torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(1000, max(2000, 200 * X.shape[-1]))  # 5000

    x_center = X[Y.argmax(), :].clone()
    weights = torch.ones_like(x_center)

    if not use_turbo:
        state.length = 2  # effectively makes search space full region within bounds

    if absolute_bounds is None:
        weights = weights * 8
        tr_lb = x_center - weights * state.length / 2.0
        tr_ub = x_center + weights * state.length / 2.0
    else:
        lb, ub = absolute_bounds
        weights = weights * (ub - lb)
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, lb, ub)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, lb, ub)

    if acqf == "ei":
        ei = qExpectedImprovement(model.cuda(), Y.max().cuda())
        X_next, _ = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]).cuda(),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    elif acqf == "ts":
        dim = X.shape[-1]
        tr_lb = tr_lb.to(device)
        tr_ub = tr_ub.to(device)
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype)
        pert = pert.to(device)
        pert = tr_lb + (tr_ub - tr_lb) * pert
        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1
        mask = mask.to(device)
        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand = X_cand.to(device)
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        assert not constrained
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():
            X_next = thompson_sampling(X_cand, num_samples=batch_size)
    else:
        assert 0, "acqf must be one of 'ei' or 'ts'"

    return X_next


def generate_batch_joco(
    state,
    models_list,  # GP model
    objective,
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Intermediated high dim output values
    S,  # Function values/ rewards
    batch_size,
    device,
    n_candidates=None,  # Number of candidates for Thompson sampling
    acqf="ts",
    dtype=torch.float32,
    absolute_bounds=None,
    use_turbo=True,
    num_restarts=10,
    raw_samples=RAW_SAMPLES,
    propegate_uncertainty_x=True,
    propegate_uncertainty_y=True,
    rand_proj_baseline=False,
):
    assert torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(1000, max(2000, 200 * X.shape[-1]))  # 5000

    x_center = X[S.argmax(), :].clone()
    weights = torch.ones_like(x_center)

    if not use_turbo:
        state.length = 2  # effectively makes search space full region within bounds

    if absolute_bounds is None:
        weights = weights * 8
        tr_lb = x_center - weights * state.length / 2.0
        tr_ub = x_center + weights * state.length / 2.0
    else:
        lb, ub = absolute_bounds
        weights = weights * (ub - lb)
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, lb, ub)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, lb, ub)

    if acqf == "ei":
        if rand_proj_baseline:
            assert 0, "ei not yet implemented for rand_proj_baseline"
        X_next = get_ei_candidates_joco(
            models_list=models_list,
            batch_size=batch_size,
            raw_samples=raw_samples,
            num_restarts=num_restarts,
            tr_lb=tr_lb,
            tr_ub=tr_ub,
            best_f=Y.max().cuda(),
        )
    elif acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype)
        tr_lb = tr_lb.to(device)
        tr_ub = tr_ub.to(device)
        pert = pert.to(device)
        pert = tr_lb + (tr_ub - tr_lb) * pert
        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1
        mask = mask.to(device)

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand = X_cand.to(device)
        X_cand[mask] = pert[mask]  # torch.Size([1000, 20]) (N, D)

        # Sample candidates according to models of intermediate outputs
        thompson_sampling = MaxPosteriorSamplingJoCo(
            models_list=models_list,
            replacement=False,
        )
        X_cand = X_cand.to(device)
        with torch.no_grad():
            X_next = thompson_sampling(
                X=X_cand,
                num_samples=batch_size,
                propegate_uncertainty_x=propegate_uncertainty_x,
                propegate_uncertainty_y=propegate_uncertainty_y,
                rand_proj_baseline=rand_proj_baseline,
            )
    else:
        assert 0

    return X_next
