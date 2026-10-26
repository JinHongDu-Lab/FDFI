"""
Flow-based DFI explainer using normalizing flows.
"""

import numpy as np
from typing import Optional, Union, Callable, Any, Tuple
from .base import Explainer
from ..results import ExplanationResults


class FlowExplainer(Explainer):
    """
    Flow-based DFI explainer using normalizing flows.
    
    Implements CPI (Conditional Permutation Importance) and 
    SCPI (Sobol-CPI) methods. Both measure feature importance in Z-space:

    - CPI: one half the average per-resample loss difference
    - SCPI: loss difference after averaging counterfactual predictions
    
    For L2 loss with independent (disentangled) features, CPI and SCPI give
    similar results. SCPI is related to the Sobol total-order sensitivity index.
    
    Z-space importance is transformed to X-space using the Jacobian of the flow
    ``phi_X[l] = sum_k H[l,k]^2 * phi_Z[k]`` where H = dX/dZ is the Jacobian
    of the decoder transformation.
    
    Parameters
    ----------
    model : callable
        The model to explain. Should take (n, d) array and return (n,) predictions.
    data : numpy.ndarray
        Background data for fitting flow and resampling. Shape (n, d).
    flow_model : object, optional
        Pre-trained flow model. If None, will create default FlowMatchingModel.
    fit_flow : bool, default=True
        Whether to fit flow model during initialization.
    nsamples : int, default=50
        Number of Monte Carlo samples per feature.
    sampling_method : str, default='resample'
        Method for generating counterfactual Z values:
        - 'resample': Sample from encoded background data
        - 'permutation': Permute within test set
        - 'normal': Sample from standard normal
        - 'condperm': Conditional permutation (regress Z_j | Z_{-j})
    permuter : object, optional
        Regressor for conditional permutation method. Defaults to LinearRegression.
    method : str, default='cpi'
        Which importance method to use:
        - 'cpi': normalized CPI - half the average loss difference
        - 'scpi': Sobol-CPI - average predictions before applying the loss
        - 'both': Compute both CPI and SCPI
    random_state : int, optional
        Seed for the counterfactual draws (``None`` means 0) and the default
        flow training seed.  Each call draws from a new stream spawned from
        it, so repeated calls (e.g. on chunks of a test set) are independent
        and the sequence of calls is reproducible.
    verbose : bool or str, default='final'
        Controls training output:
        - True or 'all': Show full progress bar
        - 'final': Only print final step status (default)
        - False: Silent
    compute_diagnostics : bool, default=True
        Whether to compute disentanglement diagnostics at setup time.
    flow_solver_rtol : float, default=1e-3
        Relative tolerance for default ODE integration in flow encode/decode.
    flow_solver_atol : float, default=1e-5
        Absolute tolerance for default ODE integration in flow encode/decode.
    diagnostics_solver_rtol : float, default=1e-6
        Relative tolerance for diagnostics round-trip integration.
    diagnostics_solver_atol : float, default=1e-8
        Absolute tolerance for diagnostics round-trip integration.
    **kwargs : dict
        Additional arguments passed to FlowMatchingModel if creating default.
    
    Attributes
    ----------
    flow_model : object
        The fitted normalizing flow model.
    Z_full : numpy.ndarray
        Encoded background data in latent space.
    method : str
        The importance method being used ('cpi', 'scpi', or 'both').
    
    Examples
    --------
    >>> import numpy as np
    >>> from fdfi.explainers import FlowExplainer
    >>> 
    >>> # Define a simple model
    >>> def model(x):
    ...     return x[:, 0] + 2 * x[:, 1]
    >>> 
    >>> # Create background data
    >>> X_train = np.random.randn(200, 5)
    >>> X_test = np.random.randn(50, 5)
    >>> 
    >>> # CPI only (default)
    >>> explainer = FlowExplainer(model, X_train, method='cpi')
    >>> results = explainer(X_test)
    >>> 
    >>> # SCPI (Sobol-CPI - different averaging order)
    >>> explainer = FlowExplainer(model, X_train, method='scpi')
    >>> results = explainer(X_test)
    """
    
    def _repr_lines(self) -> list:
        lines = super()._repr_lines()
        fitted = getattr(self, "flow_model", None) is not None
        lines.insert(
            2,
            "  Flow model: " + ("fitted" if fitted else "not fitted (call fit_flow())"),
        )
        return lines

    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        flow_model: Optional[Any] = None,
        fit_flow: bool = True,
        nsamples: int = 50,
        sampling_method: str = "resample",
        permuter: Optional[Any] = None,
        method: str = "cpi",
        random_state: Optional[int] = None,
        verbose: Union[bool, str] = "final",
        compute_diagnostics: bool = True,
        **kwargs: Any
    ):
        """Initialize the FlowExplainer."""
        # Don't fit flow in base class
        super().__init__(
            model,
            data,
            fit_flow=False,
            verbose=verbose,
            compute_diagnostics=compute_diagnostics,
            **kwargs,
        )
        
        self.nsamples = nsamples
        self.sampling_method = sampling_method
        self._set_random_state(random_state)
        self.method = method.lower()
        self.verbose = verbose
        self.kwargs = kwargs
        self.flow_solver_method = kwargs.get("flow_solver_method", "dopri5")
        self.flow_solver_rtol = kwargs.get("flow_solver_rtol", 1e-3)
        self.flow_solver_atol = kwargs.get("flow_solver_atol", 1e-5)
        self.diagnostics_solver_method = kwargs.get(
            "diagnostics_solver_method", self.flow_solver_method
        )
        self.diagnostics_solver_rtol = kwargs.get("diagnostics_solver_rtol", 1e-6)
        self.diagnostics_solver_atol = kwargs.get("diagnostics_solver_atol", 1e-8)
        self.flow_training_seed = kwargs.get("flow_training_seed", self.random_state)
        
        if self.method not in ("cpi", "scpi", "both"):
            raise ValueError(f"method must be 'cpi', 'scpi', or 'both', got '{method}'")
        
        # Set up permuter for condperm method
        if permuter is None:
            from sklearn.linear_model import LinearRegression
            self.permuter = LinearRegression()
        else:
            self.permuter = permuter
        
        # Flow model setup
        self.flow_model = flow_model
        self.Z_full = None
        
        if flow_model is not None:
            # Use provided flow model.
            # When compute_diagnostics=False (e.g. inside Crossfitting._crossfit),
            # skip both the full-data encoding and the expensive high-precision
            # diagnostic encoding — Z_full will be set externally by _crossfit.
            if self.compute_diagnostics:
                self._encode_background()
                if self.Z_full is not None:
                    self._refresh_flow_diagnostics()
        elif fit_flow and data is not None:
            # Create and fit default flow model
            self.fit_flow(num_steps=kwargs.get("num_steps", 5000), verbose=self.verbose)
    
    def fit_flow(
        self,
        X: Optional[np.ndarray] = None,
        num_steps: int = 5000,
        verbose: Union[bool, str] = None,
        **kwargs
    ) -> "FlowExplainer":
        """
        Fit the flow model on data.
        
        Can be called after initialization with fit_flow=False,
        or to refit on new data.
        
        Parameters
        ----------
        X : numpy.ndarray, optional
            Data to fit on. If None, uses self.data.
        num_steps : int, default=5000
            Number of training steps.
        verbose : bool or str, optional
            Controls training output. If None, uses self.verbose.
            - True or 'all': Show full progress bar
            - 'final': Only print final step status (default)
            - False: Silent
        **kwargs
            Additional arguments passed to flow_model.fit().
        
        Returns
        -------
        self
            For method chaining.
        """
        if X is not None:
            self.data = X
        
        if self.data is None:
            raise ValueError("No data provided for flow fitting.")
        
        try:
            from ..models import FlowMatchingModel
        except ImportError as exc:
            raise ImportError(
                "Flow matching requires torch; install it with: pip install torch torchdiffeq"
            ) from exc

        import torch
        if self.flow_training_seed is not None:
            torch.manual_seed(int(self.flow_training_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.flow_training_seed))
            np.random.seed(int(self.flow_training_seed))
        
        d = self.data.shape[1]
        self.flow_model = FlowMatchingModel(
            X=self.data,
            dim=d,
            device=self.kwargs.get("device"),
            hidden_dim=self.kwargs.get("hidden_dim", 64),
            time_embed_dim=self.kwargs.get("time_embed_dim", 32),
            num_blocks=self.kwargs.get("num_blocks", 1),
            use_bn=self.kwargs.get("use_bn", False),
        )
        
        _verbose = verbose if verbose is not None else getattr(self, 'verbose', 'final')
        self.verbose = _verbose
        if _verbose is True or _verbose == 'all' or _verbose == 'final':
            self._log("Training flow model...", level="info")
        _dequantize_noise = kwargs.pop("dequantize_noise", self.kwargs.get("dequantize_noise", 0.0))
        self.flow_model.fit(num_steps=num_steps, verbose=_verbose,
                            dequantize_noise=_dequantize_noise, **kwargs)
        
        # Encode background data
        self._encode_background()
        if self.Z_full is not None:
            self._refresh_flow_diagnostics()
        
        return self
    
    def set_flow(self, flow_model: Any) -> "FlowExplainer":
        """
        Set a user-provided flow model.
        
        The flow model must have a sample_batch(x, t_span) method where:
        - t_span=(1, 0) encodes X to Z
        - t_span=(0, 1) decodes Z to X
        
        Parameters
        ----------
        flow_model : object
            A flow model with sample_batch(x, t_span) method.
        
        Returns
        -------
        self
            For method chaining.
        """
        self.flow_model = flow_model
        self._encode_background()
        if self.Z_full is not None:
            self._refresh_flow_diagnostics()
        return self
    
    def _encode_background(self) -> None:
        """Encode background data to Z space."""
        if self.data is not None and self.flow_model is not None:
            self._set_pool(self.data)

    def _set_pool(self, X: np.ndarray) -> None:
        self.Z_full = self._encode_to_Z(X)
    
    def _encode_to_Z(
        self,
        X: np.ndarray,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """Transform X to latent space Z."""
        if self.flow_model is None:
            raise ValueError("Flow model not set. Call fit_flow() or set_flow() first.")
        
        _rtol = self.flow_solver_rtol if rtol is None else rtol
        _atol = self.flow_solver_atol if atol is None else atol
        _method = self.flow_solver_method if method is None else method

        import torch
        with torch.no_grad():
            Z = self.flow_model.sample_batch(
                X,
                t_span=(1, 0),
                rtol=_rtol,
                atol=_atol,
                method=_method,
            )
            if isinstance(Z, torch.Tensor):
                Z = Z.cpu().numpy()
        return Z
    
    def _decode_to_X(
        self,
        Z: np.ndarray,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """Transform Z back to X space."""
        if self.flow_model is None:
            raise ValueError("Flow model not set. Call fit_flow() or set_flow() first.")
        
        _rtol = self.flow_solver_rtol if rtol is None else rtol
        _atol = self.flow_solver_atol if atol is None else atol
        _method = self.flow_solver_method if method is None else method

        import torch
        with torch.no_grad():
            X_hat = self.flow_model.sample_batch(
                Z,
                t_span=(0, 1),
                rtol=_rtol,
                atol=_atol,
                method=_method,
            )
            if isinstance(X_hat, torch.Tensor):
                X_hat = X_hat.cpu().numpy()
        return X_hat

    def _decode_from_Z(self, Z: np.ndarray) -> np.ndarray:
        """
        Decode Z to X for diagnostics using tighter ODE tolerances.

        This keeps reconstruction-fidelity diagnostics focused on model fit
        rather than numerical integration error.
        """
        return self._decode_to_X(
            Z,
            rtol=self.diagnostics_solver_rtol,
            atol=self.diagnostics_solver_atol,
            method=self.diagnostics_solver_method,
        )

    def _refresh_flow_diagnostics(self) -> None:
        """Compute diagnostics from high-precision X -> Z -> X round-trip."""
        if self.data is None or self.flow_model is None:
            return
        Z_diag = self._encode_to_Z(
            self.data,
            rtol=self.diagnostics_solver_rtol,
            atol=self.diagnostics_solver_atol,
            method=self.diagnostics_solver_method,
        )
        self._compute_diagnostics(
            X_orig=self.data,
            Z_full=Z_diag,
            report_title="Flow Model",
        )
    
    def _compute_jacobian(self, Z: np.ndarray, batch_size: int = 50) -> np.ndarray:
        """
        Compute the Jacobian dX/dZ averaged over samples.
        
        For the flow decoder T: Z -> X, computes H = dX/dZ at each sample point
        and returns the average H matrix. This is used to transform Z-space
        importance to X-space importance via:
            phi_X[l] = sum_k H[l,k]^2 * phi_Z[k]
        
        Parameters
        ----------
        Z : numpy.ndarray
            Latent space data, shape (n, d).
        batch_size : int, default=50
            Batch size for Jacobian computation.
        
        Returns
        -------
        H : numpy.ndarray
            Average Jacobian matrix of shape (d, d), where H[l, k] = dX_l/dZ_k.
        """
        import torch
        from torch.autograd.functional import jacobian
        
        if self.flow_model is None:
            raise ValueError("Flow model not set. Call fit_flow() or set_flow() first.")
        
        n, d = Z.shape
        
        # Get device from flow model
        if hasattr(self.flow_model, 'device'):
            device = self.flow_model.device
        elif hasattr(self.flow_model, 'model'):
            device = next(self.flow_model.model.parameters()).device
        else:
            device = torch.device('cpu')
        
        # Define decoder function for jacobian computation
        def decoder_fn(z_single):
            """Decode a single Z vector to X."""
            z_batch = z_single.unsqueeze(0)  # (1, d)
            x_batch = self.flow_model.sample_batch(
                z_batch,
                t_span=(0, 1),
                rtol=self.flow_solver_rtol,
                atol=self.flow_solver_atol,
                method=self.flow_solver_method,
            )
            if not isinstance(x_batch, torch.Tensor):
                x_batch = torch.tensor(x_batch, dtype=torch.float32, device=device)
            return x_batch.squeeze(0)  # (d,)
        
        # Compute Jacobian for a subset of samples (for efficiency)
        n_samples = min(n, batch_size)
        indices = np.linspace(0, n - 1, n_samples, dtype=int)
        
        H_sum = np.zeros((d, d))
        
        for idx in indices:
            z_single = torch.tensor(Z[idx], dtype=torch.float32, device=device)
            # jacobian returns (output_dim, input_dim) = (d, d)
            jac = jacobian(decoder_fn, z_single)
            H_sum += jac.cpu().numpy()
        
        H_avg = H_sum / n_samples
        return H_avg
    
    def _phi_Z(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        key: int,
        y_true: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Compute CPI and SCPI importance in Z-space through ``self._loss``.

        Both scores use the counterfactual predictions for feature ``j`` but
        differ in the averaging order:

        - CPI: apply the loss per Monte-Carlo replicate, average the loss
          differences, and multiply by one half
          ``0.5 * E_b[L(r, Ỹ) - L(r, y)]``.
        - SCPI: average the prediction first, then apply the loss difference
          ``L(r, E_b[Ỹ]) - L(r, y)``.

        Here ``r`` is ``y_true`` when supplied (loss-difference / DFI form) or
        the baseline prediction ``y`` otherwise (prediction-shift form, defined
        only for the squared-error loss).

        Parameters
        ----------
        Z : numpy.ndarray
            Latent space data, shape (n, d).
        y : numpy.ndarray
            Baseline model predictions, shape (n,).
        key : int
            Seed of this call's draws; feature ``j`` uses the stream
            ``[key, j]`` (see :meth:`_call_key`).
        y_true : numpy.ndarray, optional
            True outcomes, shape (n,).  When provided, uses the DFI loss-difference
            form; when omitted, the label-free divergence form is used.

        Returns
        -------
        ueifs_cpi : numpy.ndarray
            CPI per-sample importance, shape (n, d).
        ueifs_scpi : numpy.ndarray
            SCPI per-sample importance, shape (n, d).
        """
        n, d = Z.shape
        ueifs_cpi = np.zeros((n, d))
        ueifs_scpi = np.zeros((n, d))
        
        for j in range(d):
            rng = np.random.default_rng([key, j])

            # Create B copies of Z: shape (B, n, d)
            Z_tilde = np.tile(Z[None, :, :], (self.nsamples, 1, 1))
            
            # Replace j-th component based on sampling method
            if self.sampling_method == "resample":
                if self.Z_full is None:
                    raise ValueError("Z_full is None. Call fit_flow() first.")
                resample_idx = rng.choice(
                    self.Z_full.shape[0], size=(self.nsamples, n), replace=True
                )
                Z_tilde[:, :, j] = self.Z_full[resample_idx, j]
            elif self.sampling_method == "permutation":
                perm_idx = np.array([rng.permutation(n) for _ in range(self.nsamples)])
                Z_tilde[:, :, j] = Z[perm_idx, j]
            elif self.sampling_method == "normal":
                Z_tilde[:, :, j] = rng.normal(0.0, 1.0, size=(self.nsamples, n))
            elif self.sampling_method == "condperm":
                # Conditional permutation using permuter
                from sklearn.base import clone
                Z_minus_j = np.delete(Z, j, axis=1)
                z_j = Z[:, j]
                rg = clone(self.permuter)
                rg.fit(Z_minus_j, z_j)
                z_j_hat = rg.predict(Z_minus_j)
                eps_z = z_j - z_j_hat
                for b in range(self.nsamples):
                    Z_tilde[b, :, j] = z_j_hat + rng.permutation(eps_z)
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")
            
            # Decode to X space
            Z_tilde_flat = Z_tilde.reshape(-1, d)
            X_tilde_flat = self._decode_to_X(Z_tilde_flat)
            
            # Evaluate model: shape (nsamples, n)
            y_tilde_flat = self.model(X_tilde_flat)
            y_tilde = y_tilde_flat.reshape(self.nsamples, n)
            
            # Normalized CPI: average per-resample loss differences, then / 2.
            #   φ_j^CPI = 0.5 E_b[L(r, Ỹ) - L(r, y)]
            ueifs_cpi[:, j] = self._ueif_from_counterfactuals(
                y, y_tilde, method="cpi", y_true=y_true
            )
            
            # SCPI (Sobol-CPI): average predictions before applying the loss.
            #   φ_j^SCPI = L(r, E_b[Ỹ]) - L(r, y)
            ueifs_scpi[:, j] = self._ueif_from_counterfactuals(
                y, y_tilde, method="scpi", y_true=y_true
            )
        
        return ueifs_cpi, ueifs_scpi
    
    def _ueifs_by_method(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> dict:
        """Per-sample ``(ueifs_X, ueifs_Z)`` of CPI and SCPI, keyed by method."""
        Z = self._encode_to_Z(X)
        # Baseline predictions on the decoded X, for consistency with the flow.
        y_pred = self.model(self._decode_to_X(Z))
        ueifs_cpi, ueifs_scpi = self._phi_Z(Z, y_pred, self._call_key(), y_true=y)

        # Map to X-space via the Jacobian H = dX/dZ:
        # ueifs_X[i, l] = sum_k H[l, k]^2 * ueifs_Z[i, k].
        jacobian_mode = self.kwargs.get("jacobian_mode", "average")
        if jacobian_mode == "per_sample":
            H_sq_batch = self.flow_model.Jacobi_Batch(Z) ** 2  # (n, d, d)

            def to_X(U: np.ndarray) -> np.ndarray:
                return np.einsum("ilk,ik->il", H_sq_batch, U)

        else:
            if jacobian_mode == "avg_sq":
                # Average of squared Jacobians, E[H_i^2]: avoids the
                # cancellation in E[H_i]^2 for sign-flipping (binary) features.
                n_est = min(Z.shape[0], self.kwargs.get("jacobian_n_samples", 100))
                H_sq = (self.flow_model.Jacobi_Batch(Z[:n_est]) ** 2).mean(axis=0)
            else:
                H_sq = self._compute_jacobian(Z) ** 2

            def to_X(U: np.ndarray) -> np.ndarray:
                return U @ H_sq.T

        return {
            "cpi": (to_X(ueifs_cpi), ueifs_cpi),
            "scpi": (to_X(ueifs_scpi), ueifs_scpi),
        }

    def _ueifs(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-sample UEIFs ``(ueifs_X, ueifs_Z)`` of ``self.method`` (CPI for
        ``'both'``); shared with :class:`Crossfitting`."""
        return self._ueifs_by_method(X, y)["scpi" if self.method == "scpi" else "cpi"]

    def __call__(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any
    ) -> ExplanationResults:
        """
        Compute feature importance.

        Parameters
        ----------
        X : numpy.ndarray
            Input data to explain. Shape (n_samples, n_features).
        y : numpy.ndarray, optional
            True outcomes, shape (n_samples,). When provided, uses the DFI
            loss-difference form; when omitted, the label-free divergence form
            is used.
        **kwargs : dict
            Additional parameters (unused, for API compatibility).

        Returns
        -------
        ExplanationResults
            Dict-compatible result with attribute access, ``importance()``,
            ``to_frame()`` and ``top_features()``. Contains:
            - phi_Z: Z-space importance (d,) - CPI or SCPI depending on method
            - std_Z: Standard deviation (d,)
            - se_Z: Standard error (d,)
            - phi_X: X-space importance (d,) - transformed via Jacobian
            - std_X: Standard deviation (d,)
            - se_X: Standard error (d,)
            When method='both', these are CPI and the result also includes
            the SCPI fields with suffix ``_scpi``.
        """
        if self.method != "both":
            return self._explain(X, y)
        ueifs = self._ueifs_by_method(X, y)
        self.ueifs_X, self.ueifs_Z = ueifs["cpi"]
        results = self._moments(*ueifs["cpi"])
        scpi = self._moments(*ueifs["scpi"])
        results.update({f"{k}_scpi": v for k, v in scpi.items()})
        n = X.shape[0]
        self._cache_results(results, n)
        return ExplanationResults(results, explainer_name=type(self).__name__, n=n)
