import numpy as np
from sklearn.cluster import KMeans
from scipy.special import ive
import torch
from torch import nn
from torch.nn import functional as F


class _LogC(torch.autograd.Function):

    @staticmethod
    def forward(ctx, kappa, dimension):
        k = kappa.detach().cpu().double().numpy()
        order = dimension / 2 - 1
        scaled = ive(order, k)
        ratio = ive(order + 1, k) / scaled
        value = order * np.log(k) - dimension / 2 * np.log(2 * np.pi) - np.log(scaled) - k
        if not np.isfinite(value).all() or not np.isfinite(ratio).all():
            raise FloatingPointError("vMF Bessel evaluation failed")
        ctx.save_for_backward(torch.as_tensor(ratio, device=kappa.device, dtype=kappa.dtype))
        return torch.as_tensor(value, device=kappa.device, dtype=kappa.dtype)

    @staticmethod
    def backward(ctx, grad):
        return (-grad * ctx.saved_tensors[0], None)


def log_normalizer(kappa, dimension):
    return _LogC.apply(kappa, dimension)


def sample_vmf(mu, kappa):
    dimension = mu.shape[-1]
    kappa = kappa.squeeze(-1) if kappa.ndim == mu.ndim else kappa
    m = dimension - 1
    b = m / (2 * kappa + torch.sqrt(4 * kappa.square() + m * m))
    x = (1 - b) / (1 + b)
    c = kappa * x + m * torch.log1p(-x.square())
    w = torch.zeros_like(kappa)
    pending = torch.ones_like(kappa, dtype=torch.bool)
    beta = torch.distributions.Beta(kappa.new_tensor(m / 2), kappa.new_tensor(m / 2))
    for _ in range(10000):
        proposal = beta.sample(kappa.shape)
        candidate = (1 - (1 + b) * proposal) / (1 - (1 - b) * proposal)
        accepted = (
            kappa * candidate + m * torch.log1p(-x * candidate) - c >= torch.rand_like(kappa).log()
        )
        take = pending & accepted
        w = torch.where(take, candidate, w)
        pending = pending & ~accepted
        if not pending.any():
            break
    else:
        raise RuntimeError("vMF rejection sampler did not converge")
    tangent = F.normalize(
        torch.randn(*mu.shape[:-1], dimension - 1, device=mu.device, dtype=mu.dtype), dim=-1
    )
    north_sample = torch.cat(
        (w.unsqueeze(-1), torch.sqrt((1 - w.square()).clamp_min(1e-12)).unsqueeze(-1) * tangent), -1
    )
    north = torch.zeros_like(mu)
    north[..., 0] = 1
    reflection = F.normalize(north - mu, dim=-1)
    return F.normalize(
        north_sample - 2 * (north_sample * reflection).sum(-1, keepdim=True) * reflection, dim=-1
    )


class VMFMM(nn.Module):

    def __init__(self, n_cluster=10, n_features=10, eps=1e-06):
        super().__init__()
        if n_cluster < 1 or n_features < 2:
            raise ValueError("Need positive clusters and dimension >= 2")
        self.n_cluster, self.n_features, self.eps = (n_cluster, n_features, eps)
        self.pi_ = nn.Parameter(torch.zeros(n_cluster))
        self.mu_ = nn.Parameter(F.normalize(torch.randn(n_cluster, n_features), dim=1))
        self.k_ = nn.Parameter(torch.full((n_cluster,), 10.0))

    def get_parameters(self):
        return (self.pi_.softmax(0), F.normalize(self.mu_, dim=1), F.softplus(self.k_) + self.eps)

    @torch.no_grad()
    def project_parameters_(self):
        if not torch.isfinite(self.mu_).all():
            raise FloatingPointError("VMFMM raw directions contain NaN/Inf")
        norms = self.mu_.norm(dim=1, keepdim=True)
        if (norms <= self.eps).any():
            raise FloatingPointError(
                "VMFMM raw direction norm collapsed before spherical projection"
            )
        self.mu_.div_(norms)

    @torch.no_grad()
    def initialize_from_parameters(self, pi, mu_c, k_c):
        pi, mu, k = [
            torch.as_tensor(v, device=self.pi_.device, dtype=self.pi_.dtype)
            for v in (pi, mu_c, k_c)
        ]
        if pi.shape != self.pi_.shape or mu.shape != self.mu_.shape or k.shape != self.k_.shape:
            raise ValueError("Invalid mixture parameter shapes")
        if (
            not all((torch.isfinite(v).all() for v in (pi, mu, k)))
            or (pi <= 0).any()
            or (k <= 0).any()
            or (mu.norm(dim=1) == 0).any()
        ):
            raise ValueError("Invalid mixture parameters")
        self.pi_.copy_((pi / pi.sum()).log())
        self.mu_.copy_(F.normalize(mu, dim=1))
        value = (k - self.eps).clamp_min(self.eps)
        self.k_.copy_(value + torch.log(-torch.expm1(-value)))

    def vmfmm_pdfs_log(self, z, directions, concentrations):
        return log_normalizer(concentrations, self.n_features) + z @ directions.T * concentrations

    def log_prob(self, z):
        if z.ndim != 2 or z.shape[1] != self.n_features:
            raise ValueError(f"Expected Z shape (batch, {self.n_features}), got {tuple(z.shape)}")
        if not torch.isfinite(z).all():
            raise FloatingPointError("VMFMM input Z contains NaN/Inf")
        norms = z.norm(dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=0.002, rtol=0):
            raise ValueError("VMFMM input Z must contain unit directions")
        _, mu, k = self.get_parameters()
        log_components = self.pi_.log_softmax(0).unsqueeze(0) + self.vmfmm_pdfs_log(z, mu, k)
        return torch.logsumexp(log_components, dim=1)

    def negative_log_likelihood(self, z):
        return -self.log_prob(z).mean()

    @torch.no_grad()
    def predict(self, z):
        _, mu, k = self.get_parameters()
        return (self.pi_.log_softmax(0) + self.vmfmm_pdfs_log(z, mu, k)).argmax(1).cpu().numpy()


class VMFMixture:

    def __init__(self, n_cluster, max_iter=100):
        self.n_cluster, self.max_iter = (n_cluster, max_iter)

    def fit(self, data):
        data = np.asarray(data, dtype=np.float64)
        if (
            data.ndim != 2
            or len(data) < self.n_cluster
            or data.shape[1] < 2
            or (not np.isfinite(data).all())
            or (not np.allclose(np.linalg.norm(data, axis=1), 1.0, atol=0.002))
        ):
            raise ValueError("Mixture fitting requires finite unit directions and enough samples")
        fitted = KMeans(
            n_clusters=self.n_cluster, n_init=10, max_iter=self.max_iter, random_state=0
        ).fit(data)
        centers = fitted.cluster_centers_.copy()
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        missing = norms[:, 0] <= 1e-12
        if missing.any():
            centers[missing] = data[np.flatnonzero(missing) % len(data)]
            norms = np.linalg.norm(centers, axis=1, keepdims=True)
        self.xi = centers / norms
        counts = np.bincount(fitted.labels_, minlength=self.n_cluster).astype(np.float64).clip(1.0)
        self.pi = counts / counts.sum()
        self.k = np.full(self.n_cluster, 10.0)
        return self
