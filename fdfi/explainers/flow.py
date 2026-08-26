"""
Flow-based DFI explainer using normalizing flows.
"""

import numpy as np
from typing import Optional, Union, Callable, Any
from .base import Explainer


class FlowExplainer(Explainer):
    """
    Flow-based DFI explainer using normalizing flows.
    
    Implements CPI (Conditional Permutation Importance) and 
    SCPI (Sobol-CPI) methods. Both measure feature importance in Z-space:

    - CPI: Squared difference after averaging predictions: (Y - E[f(X_tilde)])^2
    - SCPI: Conditional variance of predictions: Var[f(X_tilde)]
    
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
        - 'cpi': Conditional Permutation Importance - average predictions first
        - 'scpi': Sobol-CPI - average squared differences
        - 'both': Compute both CPI and SCPI
    random_state : int, optional
        Random seed for reproducibility.
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
        self.random_state = random_state if random_state is not None else 0
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
            self.Z_full = self._encode_to_Z(self.data)
    
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
    
    def _phi_Z(self, Z: np.ndarray, y: np.ndarray, y_true: Optional[np.ndarray] = None) -> tuple:
        """
        Compute CPI and SCPI importance in Z-space through ``self._loss``.

        Both scores use the counterfactual predictions for feature ``j`` but
        differ in the averaging order:

        - CPI: average the prediction first, then apply the loss
          ``L(r, E_b[Ỹ])``.
        - SCPI: apply the loss per Monte-Carlo replicate first, then average
          ``E_b[L(r, Ỹ)]``.

        Here ``r`` is ``y_true`` when supplied (loss-difference / DFI form) or
        the baseline prediction ``y`` otherwise (prediction-shift form, defined
        only for the squared-error loss).

        Parameters
        ----------
        Z : numpy.ndarray
            Latent space data, shape (n, d).
        y : numpy.ndarray
            Baseline model predictions, shape (n,).
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
            rng = np.random.default_rng(self.random_state + j)
            
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
                from sklearn.utils import shuffle
                Z_minus_j = np.delete(Z, j, axis=1)
                z_j = Z[:, j]
                rg = clone(self.permuter)
                rg.fit(Z_minus_j, z_j)
                z_j_hat = rg.predict(Z_minus_j)
                eps_z = z_j - z_j_hat
                for b in range(self.nsamples):
                    eps_perm = shuffle(eps_z, random_state=self.random_state + j * 1000 + b)
                    Z_tilde[b, :, j] = z_j_hat + eps_perm
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")
            
            # Decode to X space
            Z_tilde_flat = Z_tilde.reshape(-1, d)
            X_tilde_flat = self._decode_to_X(Z_tilde_flat)
            
            # Evaluate model: shape (nsamples, n)
            y_tilde_flat = self.model(X_tilde_flat)
            y_tilde = y_tilde_flat.reshape(self.nsamples, n)
            
            # CPI: average the prediction first, then apply the loss.
            #   φ_j^CPI = L(r, E_b[Ỹ])   with r = y_true or the baseline prediction
            ueifs_cpi[:, j] = self._ueif_from_counterfactuals(
                y, y_tilde, method="cpi", y_true=y_true
            )
            
            # SCPI (Sobol-CPI): apply the loss per replicate, then average.
            #   φ_j^SCPI = E_b[L(r, Ỹ)]
            # For squared error this equals CPI + Var_b(Ỹ), matching the
            # documented SCPI = E_b[(Y - f(X̃_b))²].
            ueifs_scpi[:, j] = self._ueif_from_counterfactuals(
                y, y_tilde, method="scpi", y_true=y_true
            )
        
        return ueifs_cpi, ueifs_scpi
    
    def __call__(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> dict:
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
        dict
            Dictionary containing:
            - phi_Z: Z-space importance (d,) - CPI or SCPI depending on method
            - std_Z: Standard deviation (d,)
            - se_Z: Standard error (d,)
            - phi_X: X-space importance (d,) - transformed via Jacobian
            - std_X: Standard deviation (d,)
            - se_X: Standard error (d,)
            When method='both', also includes phi_Z_scpi, std_Z_scpi, se_Z_scpi.
        """
        n, d = X.shape
        
        # Encode to Z space
        Z = self._encode_to_Z(X)
        
        # Get model predictions on decoded X (for consistency with flow)
        X_hat = self._decode_to_X(Z)
        y_pred = self.model(X_hat)
        
        # Compute both CPI and SCPI in Z-space (per-sample UEIFs)
        ueifs_cpi, ueifs_scpi = self._phi_Z(Z, y_pred, y_true=y)
        
        # Transform per-sample UEIFs from Z-space to X-space via Jacobian H = dX/dZ
        # ueifs_X[i, l] = sum_k H[l,k]^2 * ueifs_Z[i, k]
        jacobian_mode = self.kwargs.get("jacobian_mode", "average")
        n_jac = self.kwargs.get("jacobian_n_samples", 100)
        if jacobian_mode == "per_sample":
            # Per-sample Jacobians: H_batch[i] = dX/dZ at Z[i]
            H_batch = self.flow_model.Jacobi_Batch(Z)   # (n, d, d)
            H_sq_batch = H_batch ** 2
            ueifs_cpi_X  = np.einsum("ilk,ik->il", H_sq_batch, ueifs_cpi)
            ueifs_scpi_X = np.einsum("ilk,ik->il", H_sq_batch, ueifs_scpi)
        elif jacobian_mode == "avg_sq":
            # Average of squared Jacobians: E[H_i^2] — correct for binary data.
            # Avoids the cancellation in E[H_i]^2 for sign-flipping features.
            n_est = min(n, n_jac)
            H_batch = self.flow_model.Jacobi_Batch(Z[:n_est])  # (n_est, d, d)
            H_sq_avg = (H_batch ** 2).mean(axis=0)              # (d, d)
            ueifs_cpi_X  = ueifs_cpi  @ H_sq_avg.T
            ueifs_scpi_X = ueifs_scpi @ H_sq_avg.T
        else:  # "average" — existing behaviour, backward-compatible
            H = self._compute_jacobian(Z)
            H_sq = H ** 2
            ueifs_cpi_X  = ueifs_cpi  @ H_sq.T
            ueifs_scpi_X = ueifs_scpi @ H_sq.T
        
        # Store per-sample UEIFs for group_importance()
        if self.method == "scpi":
            self.ueifs_Z = ueifs_scpi
            self.ueifs_X = ueifs_scpi_X
        else:
            self.ueifs_Z = ueifs_cpi
            self.ueifs_X = ueifs_cpi_X
        
        # Aggregate statistics
        ddof = 1 if n > 1 else 0
        results = {}
        
        if self.method == "cpi":
            # Z-space (CPI)
            phi_Z = np.mean(ueifs_cpi, axis=0)
            std_Z = np.std(ueifs_cpi, axis=0)
            se_Z = np.std(ueifs_cpi, axis=0, ddof=ddof) / np.sqrt(n)
            # X-space (transformed via Jacobian)
            phi_X = np.mean(ueifs_cpi_X, axis=0)
            std_X = np.std(ueifs_cpi_X, axis=0)
            se_X = np.std(ueifs_cpi_X, axis=0, ddof=ddof) / np.sqrt(n)
            results.update({
                "phi_Z": phi_Z,
                "std_Z": std_Z,
                "se_Z": se_Z,
                "phi_X": phi_X,
                "std_X": std_X,
                "se_X": se_X,
            })
        elif self.method == "scpi":
            # Z-space (SCPI)
            phi_Z = np.mean(ueifs_scpi, axis=0)
            std_Z = np.std(ueifs_scpi, axis=0)
            se_Z = np.std(ueifs_scpi, axis=0, ddof=ddof) / np.sqrt(n)
            # X-space (transformed via Jacobian)
            phi_X = np.mean(ueifs_scpi_X, axis=0)
            std_X = np.std(ueifs_scpi_X, axis=0)
            se_X = np.std(ueifs_scpi_X, axis=0, ddof=ddof) / np.sqrt(n)
            results.update({
                "phi_Z": phi_Z,
                "std_Z": std_Z,
                "se_Z": se_Z,
                "phi_X": phi_X,
                "std_X": std_X,
                "se_X": se_X,
            })
        else:  # method == "both"
            # CPI (Z-space and X-space)
            phi_Z_cpi = np.mean(ueifs_cpi, axis=0)
            std_Z_cpi = np.std(ueifs_cpi, axis=0)
            se_Z_cpi = np.std(ueifs_cpi, axis=0, ddof=ddof) / np.sqrt(n)
            phi_X_cpi = np.mean(ueifs_cpi_X, axis=0)
            std_X_cpi = np.std(ueifs_cpi_X, axis=0)
            se_X_cpi = np.std(ueifs_cpi_X, axis=0, ddof=ddof) / np.sqrt(n)
            # SCPI (Z-space and X-space)
            phi_Z_scpi = np.mean(ueifs_scpi, axis=0)
            std_Z_scpi = np.std(ueifs_scpi, axis=0)
            se_Z_scpi = np.std(ueifs_scpi, axis=0, ddof=ddof) / np.sqrt(n)
            phi_X_scpi = np.mean(ueifs_scpi_X, axis=0)
            std_X_scpi = np.std(ueifs_scpi_X, axis=0)
            se_X_scpi = np.std(ueifs_scpi_X, axis=0, ddof=ddof) / np.sqrt(n)
            results.update({
                # Default to CPI for phi_Z/phi_X
                "phi_Z": phi_Z_cpi,
                "std_Z": std_Z_cpi,
                "se_Z": se_Z_cpi,
                "phi_X": phi_X_cpi,
                "std_X": std_X_cpi,
                "se_X": se_X_cpi,
                # SCPI with suffix
                "phi_Z_scpi": phi_Z_scpi,
                "std_Z_scpi": std_Z_scpi,
                "se_Z_scpi": se_Z_scpi,
                "phi_X_scpi": phi_X_scpi,
                "std_X_scpi": std_X_scpi,
                "se_X_scpi": se_X_scpi,
            })
        
        self._cache_results(results, n)
        return results
