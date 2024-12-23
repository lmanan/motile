from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structsvm as ssvm

if TYPE_CHECKING:
    from motile.solver import Solver


def fit_weights(
    solver: Solver,
    gt_attribute: str,
    regularizer_weight: float,
    max_iterations: int | None,
    eps: float,
    ground_truth: np.ndarray,
    mask: np.ndarray,
    fit_subset_weights: bool,
) -> np.ndarray:
    """Return the optimal weights for the given solver.

    This uses `structsvm.BundleMethod` to fit the weights.

    Args:
        solver (Solver):
            The solver to fit the weights for.
        gt_attribute (str):
            Node/edge attribute that marks the ground truth for fitting.
            `gt_attribute` is expected to be set to `1` for objects labeled as
            ground truth, `0` for objects explicitly labeled as not part of the
            ground truth, and `None` or not set for unlabeled objects.
        regularizer_weight (float):
            The weight of the quadratic regularizer.
        max_iterations (int):
            Maximum number of gradient steps in the structured SVM.
        eps (float):
            Convergence threshold.
        ground_truth (np.ndarray):
            Set to 1 when True, else 0.
        mask (np.ndarray):
            Set to 1 when annotation is available, else 0.
        fit_subset_weights (bool):
            If set to True, then appearance and disappearance weights are not
            fit.

    Returns:
        np.ndarray:
            The optimal weights for the given solver.
    """
    features = solver.features.to_ndarray()
    mask = mask.astype(np.bool_)
    features[~mask] = 0

    loss = ssvm.SoftMarginLoss(
        solver.constraints,
        features.T,  # note, now 8 x N
        ground_truth,
        ssvm.HammingCosts(ground_truth, mask, solver.num_variables),
    )

    def masked_loss(w):
        all_w = np.concatenate(
            (w, np.array([0, 1, 0, 1])), axis=0
        )  # this assumes appear/disappear are the last four weights, adjust accordingly
        value, gradient = loss.value_and_gradient(all_w)
        return (
            value,
            gradient[:-4],
        )  # this assumes appear/disappear are the last four weights, adjust accordingly

    if fit_subset_weights:
        bundle_method = ssvm.BundleMethod(
            masked_loss,
            dims=features.shape[1] - 4,  # 8 = ignore appearance and disappearance
            regularizer_weight=regularizer_weight,
            eps=eps,
        )
    else:
        bundle_method = ssvm.BundleMethod(
            loss.value_and_gradient,
            dims=features.shape[1],  # 8 =position and attrackt
            regularizer_weight=regularizer_weight,
            eps=eps,
        )

    return bundle_method.optimize(max_iterations)
