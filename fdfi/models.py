import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torchdiffeq import odeint
from torch.func import jacrev



class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, scale: float = 1.0, use_bn: bool = False):
        super().__init__()
        self.use_bn = use_bn
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.scale = scale
        if use_bn:
            self.bn1 = nn.BatchNorm1d(dim)
            self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        out = self.fc1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = F.silu(out)
        out = self.fc2(out)
        if self.use_bn:
            out = self.bn2(out)
        return x + self.scale * out


class FlowModelResNet(nn.Module):
    def __init__(self, input_dim=2, time_embed_dim=64, hidden_dim=256, num_blocks=4, use_bn=True):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        in_feat = input_dim + time_embed_dim
        layers = [nn.Linear(in_feat, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_blocks):
            layers.append(ResidualMLPBlock(hidden_dim, use_bn=use_bn))

        layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, input_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        t_embed = self.time_embed(t)
        xt = torch.cat([x, t_embed], dim=-1)
        return self.net(xt)


class FlowMatchingModel:

    def __init__(self, X, dim=10, sigma_min=0.01, device=None,
                 hidden_dim=64, time_embed_dim=32, num_blocks=1, use_bn=False):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = dim
        self.sigma_min = sigma_min
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.num_blocks = num_blocks
        self.use_bn = use_bn

        self.set_data(X)

        self.model = FlowModelResNet(
            input_dim=dim,
            time_embed_dim=time_embed_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            use_bn=use_bn
        ).to(self.device)

    def set_data(self, X_np):
        if isinstance(X_np, torch.Tensor):
            self.X = X_np.to(self.device).float()
        else:
            self.X = torch.from_numpy(np.asarray(X_np)).float().to(self.device)

    def _sample_source(self, batch_size):
        return torch.randn(batch_size, self.dim, device=self.device)

    def _sample_target(self, batch_size):
        idx = torch.randint(0, self.X.size(0), (batch_size,), device=self.device)
        return self.X[idx]

    def _loss_fn(self, x0, x1, t):
        """
        Conditional Flow Matching Loss (Lipman et al. 2022).
        Calculates the MSE between predicted velocity and target vector field.
        """
        coeff_x0 = 1 - (1 - self.sigma_min) * t
        xt = coeff_x0 * x0 + t * x1
        v_target = x1 - (1 - self.sigma_min) * x0
        v_pred = self.model(xt, t)
        return F.mse_loss(v_pred, v_target)

    def fit(self, X=None, num_steps=20000, batch_size=512, lr=5e-4, show_plot=False, verbose=True, dequantize_noise=0.0):
        """
        Train the flow matching model.
        
        Parameters
        ----------
        X : array-like, optional
            Training data. If None, uses data set in constructor.
        num_steps : int, default=20000
            Number of training steps.
        batch_size : int, default=512
            Batch size for training.
        lr : float, default=5e-4
            Learning rate.
        show_plot : bool, default=False
            Whether to show the loss curve after training.
        verbose : bool or str, default=True
            Controls training output:
            - True or 'all': Show full progress bar (default)
            - 'final': Only print final step status
            - False or 0: Silent
        """
        if X is not None:
            self.set_data(X)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        losses = []
        
        # Determine verbosity mode
        show_progress = verbose is True or verbose == 'all'
        show_final = verbose == 'final'
        
        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Training", ncols=100)

        for step in iterator:
            x0 = self._sample_source(batch_size)
            x1 = self._sample_target(batch_size)
            if dequantize_noise > 0:
                x1 = x1 + torch.randn_like(x1) * dequantize_noise
            t = torch.rand(batch_size, 1, device=self.device)

            loss = self._loss_fn(x0, x1, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if show_progress:
                iterator.set_postfix(loss=f"{loss.item():.4f}")
        
        if show_final:
            print(f"Training complete: {num_steps} steps, final loss={losses[-1]:.4f}")

        if show_plot:
            plt.plot(losses)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Loss Curve")
            plt.grid(True)
            plt.show()

    def sample_batch(
        self,
        x0,
        t_span=(0, 1),
        rtol=1e-3,
        atol=1e-5,
        method="dopri5",
    ):
        self.model.eval()

        if isinstance(x0, np.ndarray):
            x0 = torch.tensor(x0, dtype=torch.float32)
        x0 = x0.to(self.device)

        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)

        t = torch.tensor(t_span, dtype=torch.float32).to(self.device)

        def odefunc(t, x):
            t_expand = torch.ones(x.size(0), 1, device=self.device) * t
            return self.model(x, t_expand)

        out = odeint(odefunc, x0, t, rtol=rtol, atol=atol, method=method)
        return out[-1]  
    
    def Jacobi_Batch(self, x_batch, t_span=(0, 1)):
        """
        Per-sample Jacobians dX/dZ for a batch of latent-space points.

        Each sample is processed independently via the augmented ODE
        (Jacobi_N). Samples are evaluated sequentially; peak memory per
        call is O(d²) for a single augmented ODE state.

        Parameters
        ----------
        x_batch : array-like, shape (n, d)
            Latent-space (Z) starting points.
        t_span : tuple, default=(0, 1)
            Integration interval (0,1) decodes Z→X.

        Returns
        -------
        H_batch : numpy.ndarray, shape (n, d, d)
            H_batch[i] = dX/dZ at x_batch[i].
        """
        import torch
        self.model.eval()
        if isinstance(x_batch, torch.Tensor):
            x_batch = x_batch.detach().cpu().numpy()
        else:
            x_batch = np.asarray(x_batch)
        Hs = [self.Jacobi_N(x_batch[i], t_span) for i in range(len(x_batch))]
        return np.stack(Hs, axis=0)   # (n, d, d)

    def Jacobi_N(self, y0, t_span=(0, 1)):
        self.model.eval()

        if isinstance(y0, torch.Tensor):
            y0 = y0.detach().cpu().numpy()

        x0 = torch.tensor(y0[:self.dim], dtype=torch.float32, device=self.device)
        J0 = torch.eye(self.dim, dtype=torch.float32, device=self.device).flatten()
        y0_torch = torch.cat([x0, J0])  

        def odefunc_aug(t, y_aug):
            x = y_aug[:self.dim]
            J = y_aug[self.dim:].reshape(self.dim, self.dim)

            x = x.detach().requires_grad_(True)
            t_tensor = torch.tensor([[t]], dtype=torch.float32, device=self.device)

            v = self.model(x.unsqueeze(0), t_tensor).squeeze(0)

        
            A = jacrev(lambda x_: self.model(x_, t_tensor).squeeze(0))(x.unsqueeze(0)).squeeze(0)

            dxdt = v
            dJdt = A @ J

            return torch.cat([dxdt, dJdt.reshape(-1)])

        t = torch.tensor(t_span, dtype=torch.float32, device=self.device)
        y_aug_out = odeint(odefunc_aug, y0_torch, t, rtol=1e-3, atol=1e-5, method='dopri5')
        y1 = y_aug_out[-1]
        J1 = y1[self.dim:].reshape(self.dim, self.dim).detach().cpu().numpy()
        return J1


