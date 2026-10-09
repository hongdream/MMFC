import math
import torch
from torch import nn
from networks import MultimodalSphericalVAE, SphericalVAEEncoder
from vmf import VMFMM, VMFMixture


def check_finite(value, name):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name}: NaN/Inf")


def check_unit(value, name):
    check_finite(value, name)
    norms = value.norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=0.002, rtol=0):
        raise ValueError(f"{name}: expected unit directions, norms={norms.detach().cpu()}")


def check_mixture(mixture, name="VMFMM"):
    pi, mu, kappa = mixture.get_parameters()
    for value, suffix in ((pi, "pi"), (mu, "mu_c"), (kappa, "kappa_c")):
        check_finite(value, f"{name} {suffix}")
    check_unit(mu, f"{name} mu_c")
    if (pi < 0).any() or not torch.isclose(pi.sum(), pi.new_tensor(1.0), atol=1e-05):
        raise ValueError(f"{name}: mixture weights must sum to one")
    if not (kappa > 0).all():
        raise ValueError(f"{name}: concentrations must be positive")


def build_spherical_vae_encoder(
    text_dim, speech_dim, latent_dim=20, encoder_hidden=256, encoder_layers=(4, 2, 1)
):
    return SphericalVAEEncoder(
        text_dim, speech_dim, latent_dim, encoder_hidden, encoder_layers=encoder_layers
    )


def build_spherical_vae(
    text_dim,
    speech_dim,
    latent_dim=20,
    encoder_hidden=256,
    decoder_hidden=256,
    encoder_layers=(4, 2, 1),
    decoder_layers=None,
):
    return MultimodalSphericalVAE(
        text_dim,
        speech_dim,
        latent_dim,
        encoder_hidden,
        decoder_hidden,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
    )


class FederatedSphericalVAEClient(nn.Module):

    def __init__(self, spherical_vae, n_clusters=None):
        super().__init__()
        self.spherical_vae = spherical_vae
        self.vmfmm = (
            VMFMM(n_cluster=n_clusters, n_features=spherical_vae.latent_dim)
            if n_clusters is not None
            else None
        )

    @property
    def shared_model(self):
        return self.spherical_vae

    def forward(
        self,
        text_bow,
        speech_boaw,
        lambda_text=1.0,
        lambda_speech=1.0,
        lambda_cluster=0.0,
        sample=True,
    ):
        for value, name in (
            (lambda_text, "lambda_text"),
            (lambda_speech, "lambda_speech"),
            (lambda_cluster, "lambda_cluster"),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        out = self.spherical_vae(text_bow, speech_boaw, sample=sample)
        out["reconstruction_loss"] = (
            lambda_text * out["text_reconstruction_loss"]
            + lambda_speech * out["speech_reconstruction_loss"]
        )
        out["cluster_loss"] = out["reconstruction_loss"].new_zeros(())
        if lambda_cluster:
            if self.vmfmm is None:
                raise ValueError("lambda_cluster requires a client mixture")
            check_mixture(self.vmfmm, "client mixture")
            out["cluster_loss"] = self.vmfmm.negative_log_likelihood(out["mu"])
            check_finite(out["cluster_loss"], "cluster negative log likelihood")
        out["total_loss"] = out["reconstruction_loss"] + lambda_cluster * out["cluster_loss"]
        check_finite(out["total_loss"], "client total loss")
        return out

    def encode_posterior(self, text_bow, speech_boaw):
        return self.spherical_vae.encode(text_bow, speech_boaw)

    def encode(self, text_bow, speech_boaw, sample=False):
        posterior = self.encode_posterior(text_bow, speech_boaw)
        return self.spherical_vae.sample_posterior(posterior) if sample else posterior["mu"]

    @torch.no_grad()
    def predict_cluster(self, text_bow, speech_boaw):
        if self.vmfmm is None:
            raise ValueError("Cluster prediction requires a client VMFMM")
        was_training = self.training
        self.eval()
        try:
            z = self.encode(text_bow, speech_boaw)
            check_mixture(self.vmfmm, "spherical VAE client VMFMM")
            return self.vmfmm.predict(z)
        finally:
            self.train(was_training)


def build_spherical_vae_client(
    text_dim,
    speech_dim,
    latent_dim=20,
    encoder_hidden=256,
    decoder_hidden=256,
    n_clusters=None,
    encoder_layers=(4, 2, 1),
    decoder_layers=None,
):
    return FederatedSphericalVAEClient(
        build_spherical_vae(
            text_dim,
            speech_dim,
            latent_dim,
            encoder_hidden,
            decoder_hidden,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
        ),
        n_clusters,
    )
