from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structsvm as ssvm

from .variables import EdgeSelected, NodeAppear, NodeDisappear, NodeSelected

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

    Returns:
        np.ndarray:
            The optimal weights for the given solver.
    """
    features = solver.features.to_ndarray()
    for node, index in solver.get_variables(NodeSelected).items():
        if mask[index] == 0.0:
            features[index] = (
                0  # we lack information about whether this node is selected.
            )

    for edge, index in solver.get_variables(EdgeSelected).items():
        if mask[index] == 0.0:
            features[index] = (
                0  # we lack information about whether this edge is selected.
            )

    for node, index in solver.get_variables(NodeAppear).items():
        if mask[index] == 0.0:  # we lack information about whether this node appeared
            features[index] = 0

    for node, index in solver.get_variables(NodeDisappear).items():
        if (
            mask[index] == 0.0
        ):  # we lack information about whether this node disappeared
            features[index] = 0

    loss = ssvm.SoftMarginLoss(
        solver.constraints,
        features.T,  # note, now 8 x N
        ground_truth,
        ssvm.HammingCosts(ground_truth, mask, solver.num_variables),
    )

    bundle_method = ssvm.BundleMethod(
        loss.value_and_gradient,
        dims=features.shape[1],  # 8 =position and attrackt
        regularizer_weight=regularizer_weight,
        eps=eps,
    )

    return bundle_method.optimize(max_iterations)
