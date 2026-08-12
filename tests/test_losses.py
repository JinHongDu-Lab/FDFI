"""
Tests for the per-sample loss registry in :mod:`fdfi.losses`.
"""

import functools

import numpy as np
import pytest

from fdfi.losses import (
    resolve_loss,
    available_losses,
    squared_error,
    absolute_error,
    huber,
    pinball,
    log_loss,
    brier,
    zero_one,
)


class TestResolveLoss:
    def test_none_returns_squared_error(self):
        assert resolve_loss(None) is squared_error

    @pytest.mark.parametrize(
        "key,fn",
        [
            ("squared_error", squared_error),
            ("l2", squared_error),
            ("mse", squared_error),
            ("absolute_error", absolute_error),
            ("l1", absolute_error),
            ("mae", absolute_error),
            ("huber", huber),
            ("pinball", pinball),
            ("quantile", pinball),
            ("log_loss", log_loss),
            ("cross_entropy", log_loss),
            ("bce", log_loss),
            ("brier", brier),
            ("zero_one", zero_one),
        ],
    )
    def test_string_keys_and_aliases(self, key, fn):
        assert resolve_loss(key) is fn

    def test_case_insensitive(self):
        assert resolve_loss("L2") is squared_error
        assert resolve_loss("Log_Loss") is log_loss

    def test_callable_passthrough(self):
        def custom(y_true, y_pred):
            return np.abs(y_true - y_pred) ** 3

        assert resolve_loss(custom) is custom

    def test_partial_callable(self):
        loss = functools.partial(huber, delta=2.0)
        assert callable(loss)
        # partial is returned unchanged
        assert resolve_loss(loss) is loss

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            resolve_loss("not_a_loss")

    def test_available_losses_sorted_and_complete(self):
        losses = available_losses()
        assert losses == sorted(losses)
        assert "squared_error" in losses
        assert "log_loss" in losses


class TestBuiltinLosses:
    def test_shapes_no_reduction(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.0, 2.0])
        for fn in (squared_error, absolute_error, huber, pinball, brier):
            out = fn(y_true, y_pred)
            assert out.shape == (3,)

    def test_squared_error_values(self):
        assert np.allclose(
            squared_error(np.array([1.0, 0.0]), np.array([0.0, 0.0])),
            [1.0, 0.0],
        )

    def test_absolute_error_values(self):
        assert np.allclose(
            absolute_error(np.array([1.0, -2.0]), np.array([0.0, 0.0])),
            [1.0, 2.0],
        )

    def test_huber_quadratic_and_linear_regions(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([0.5, 5.0])  # |r| = 0.5 (quad), 5.0 (linear), delta=1
        out = huber(y_true, y_pred, delta=1.0)
        assert np.isclose(out[0], 0.5 * 0.5 ** 2)
        assert np.isclose(out[1], 1.0 * (5.0 - 0.5))

    def test_pinball_asymmetry(self):
        # under-prediction (r>0) penalised by q, over-prediction by (1-q)
        q = 0.9
        under = pinball(np.array([1.0]), np.array([0.0]), quantile=q)  # r=1
        over = pinball(np.array([0.0]), np.array([1.0]), quantile=q)   # r=-1
        assert np.isclose(under[0], q * 1.0)
        assert np.isclose(over[0], (1.0 - q) * 1.0)

    def test_log_loss_value(self):
        # -log(0.5) = ln(2)
        assert np.isclose(log_loss(np.array([1.0]), np.array([0.5]))[0], np.log(2))

    def test_log_loss_clips_extremes(self):
        # perfect-but-wrong prediction stays finite due to clipping
        out = log_loss(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        assert np.all(np.isfinite(out))

    def test_zero_one(self):
        y_true = np.array([1.0, 0.0, 1.0])
        y_pred = np.array([0.9, 0.8, 0.2])  # -> 1, 1, 0
        out = zero_one(y_true, y_pred, threshold=0.5)
        assert np.allclose(out, [0.0, 1.0, 1.0])

    def test_broadcasting_over_replicates(self):
        # (1, n) y_true broadcasts against (B, n) y_pred -> (B, n)
        y_true = np.array([[1.0, 0.0]])
        y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])
        out = log_loss(y_true, y_pred)
        assert out.shape == (2, 2)
