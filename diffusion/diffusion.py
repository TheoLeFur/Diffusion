import torch
import torch.nn as nn
import numpy as np


def extract(value: torch.Tensor, t: torch.Tensor, x0_shape: torch.Tensor) -> torch.Tensor:
    """
    We make a method that reshapes val in [batch_size, 1,1, ...]
    This is done for broadcasting purposes
    """
    device = t.device
    out = torch.gather(value, index=t, dim=0).float().to(device)

    return out.view([t.shape[0]] + [1] * (len(x0_shape) - 1))


class DiffusionTrainer(nn.Module):

    def __init__(self, model, beta1, betaT, T):

        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer("betas", torch.linspace(beta1, betaT, T).double())
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.register_buffer("sqrt_alpha", torch.sqrt(self.alpha_bar))
        self.register_buffer("one_minus_sqrt_alpha",
                             torch.sqrt(1 - self.alpha_bar))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        t = torch.randint(0, self.T, size=(x.shape[0], ), device=x.device)
        eps = torch.randn_like(x)

        xt = extract(self.sqrt_alpha, t, x.shape) * x + \
            extract(self.one_minus_sqrt_alpha, t, x.shape) * eps
        loss = nn.functional.mse_loss(self.model(xt, t), eps)
        return loss


class DiffusionSampler(nn.Module):

    def __init__(self, model, beta1, betaT, T):

        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer("betas", torch.linspace(beta1, betaT, T).double())
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alpha_bar", torch.cumprod(self.alphas))
        self.register_buffer("alpha_bar_prev", nn.functional.pad(
            self.alpha_bar, [1, 0], value=1)[:self.T])
        self.register_buffer("sqrt_alpha", torch.sqrt(self.alphas))
        self.register_buffer("sqrt_one_minus_alpha_bar",
                             torch.sqrt(1 - self.alpha_bar))

        self.register_buffer("post_variance", self.betas *
                             (1 - self.alpha_bar_prev)/(1 - self.alpha_bar))
        self.register_buffer("mean_coeff1", 1./torch.sqrt(self.alphas))
        self.register_buffer("mean_coeff2", self.mean_coeff1 *
                             self.betas / self.sqrt_one_minus_alpha_bar)

    def p_mean_variance(self, xt: torch.Tensor, t: torch.Tensor):

        var = torch.cat(self.pos_variance[1:2], self.betas[:1])
        var = extract(var, t, xt.shape)

        eps_pred = self.model(xt, t)
        mean = extract(self.mean_coeff1, t, xt.shape) * xt - \
            extract(self.mean_coeff2, t, xt.shape) * eps_pred

        return eps_pred

    def forward(self, xT: torch.Tensor) -> torch.Tensor:

        xt = xT

        for t_step in reversed(range(self.T)):

            t = xt.new_ones([xT.shape,], dtype=torch.long) * t_step
            noise = torch.randn(xt) if t_step > 1 else 0
            mean, log_var = self.p_mean_variance(xt, t)
            xt = mean + torch.exp(0.5 * log_var) * noise

        x0 = xt
        return torch.clip(x0, -1, 1)
