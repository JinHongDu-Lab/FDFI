"""
Per-sample loss functions for feature-importance estimation.

DFI/CPI feature importance is defined through a loss ``L(y, ŷ)``.  For the
squared-error loss the importance reduces to the classic difference of L2
residuals; this module generalises that definition to arbitrary regression
and (binary) classification losses.

A loss is any callable ``loss(y_true, y_pred) -> np.ndarray`` that returns the
**per-sample** loss without reduction.  Inputs broadcast with NumPy semantics,
so a loss also accepts ``y_pred`` of shape ``(B, n)`` (Monte-Carlo replicates)
and ``y_true`` of shape ``(1, n)``, returning ``(B, n)``.

Built-in losses can be selected by string key via :func:`resolve_loss`, or a
custom callable (e.g. ``functools.partial(huber, delta=2.0)``) can be supplied
directly.
"""

from typing import Callable, Optional, Union

import numpy as np

LossFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Squared-error (L2) loss: ``(y_true - y_pred) ** 2``."""
    return (np.asarray(y_true) - np.asarray(y_pred)) ** 2


def absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Absolute-error (L1) loss: ``|y_true - y_pred|``."""
    return np.abs(np.asarray(y_true) - np.asarray(y_pred))


def huber(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """
    Huber loss with threshold ``delta``.

    Quadratic for residuals ``|r| <= delta`` and linear beyond, giving
    robustness to outliers while remaining smooth at the origin.
    """
    r = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return np.where(r <= delta, 0.5 * r ** 2, delta * (r - 0.5 * delta))


def pinball(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.5) -> np.ndarray:
    """
    Pinball (quantile) loss for the given ``quantile`` in ``(0, 1)``.

    At ``quantile=0.5`` this is half the absolute error.
    """
    r = np.asarray(y_true) - np.asarray(y_pred)
    return np.maximum(quantile * r, (quantile - 1.0) * r)


def log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Binary cross-entropy (log loss).

    ``y_true`` are labels in ``{0, 1}`` and ``y_pred`` are predicted
    probabilities ``P(y = 1)``; probabilities are clipped to
    ``[eps, 1 - eps]`` for numerical stability.
    """
    p = np.clip(np.asarray(y_pred), eps, 1.0 - eps)
    y = np.asarray(y_true)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def brier(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Brier score for binary probabilities: ``(y_true - y_pred) ** 2``.

    Numerically identical to :func:`squared_error` but kept separate to signal
    a classification (probability) use case.
    """
    return (np.asarray(y_true) - np.asarray(y_pred)) ** 2


def zero_one(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    0-1 misclassification loss for binary predictions.

    ``y_pred`` is thresholded at ``threshold`` to obtain a hard label, which is
    compared with ``y_true``; returns ``1.0`` for a mistake and ``0.0``
    otherwise.  Non-differentiable — intended for reporting, not optimisation.
    """
    y = np.asarray(y_true)
    pred_label = (np.asarray(y_pred) >= threshold).astype(y.dtype)
    return (y != pred_label).astype(float)


_LOSS_REGISTRY: dict = {
    "squared_error": squared_error,
    "l2": squared_error,
    "mse": squared_error,
    "absolute_error": absolute_error,
    "l1": absolute_error,
    "mae": absolute_error,
    "huber": huber,
    "pinball": pinball,
    "quantile": pinball,
    "log_loss": log_loss,
    "cross_entropy": log_loss,
    "bce": log_loss,
    "brier": brier,
    "zero_one": zero_one,
}


def available_losses() -> list:
    """Return the sorted list of built-in loss keys accepted by :func:`resolve_loss`."""
    return sorted(_LOSS_REGISTRY)


def resolve_loss(loss: Optional[Union[str, LossFn]]) -> LossFn:
    """
    Resolve a loss specifier to a callable ``loss(y_true, y_pred) -> ndarray``.

    Parameters
    ----------
    loss : str, callable, or None
        - ``None`` → :func:`squared_error` (the DFI default).
        - ``str`` → a built-in key (see :func:`available_losses`); aliases such
          as ``'l2'``, ``'mse'``, ``'bce'`` are accepted (case-insensitive).
        - ``callable`` → returned unchanged.  Must accept ``(y_true, y_pred)``
          and return the per-sample loss.

    Returns
    -------
    callable
        The resolved per-sample loss function.

    Raises
    ------
    ValueError
        If ``loss`` is an unknown string key.
    """
    if loss is None:
        return squared_error
    if callable(loss):
        return loss
    try:
        return _LOSS_REGISTRY[loss.lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"Unknown loss {loss!r}; choose a callable or one of "
            f"{available_losses()}."
        )
