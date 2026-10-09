import torch
from torch import nn
from torch.nn import functional as F
from vmf import sample_vmf


def build_mlp(widths, activation=nn.GELU):
    layers = []
    for left, right in zip(widths[:-1], widths[1:]):
        layers.extend((nn.Linear(left, right), activation()))
    return nn.Sequential(*layers)


def layer_widths(hidden_dim, layers):
    if not layers or any((not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in layers)):
        raise ValueError("Layer multipliers must be positive integers")
    return tuple((hidden_dim * n for n in layers))


def normalized_cross_entropy(target, log_probabilities, name):
    if target.shape != log_probabilities.shape or target.ndim != 2:
        raise ValueError(
            f"{name}: target/output shapes must match, got {tuple(target.shape)} and {tuple(log_probabilities.shape)}"
        )
    if (target < 0).any():
        raise ValueError(f"{name}: reconstruction targets must be nonnegative")
    mass = target.sum(dim=1, keepdim=True)
    valid = mass.squeeze(1) > 0
    if not valid.any():
        raise ValueError(f"{name}: batch contains no nonzero target rows")
    normalized_target = target[valid] / mass[valid]
    return -(normalized_target * log_probabilities[valid]).sum(dim=1).mean()


class SphericalVAEEncoder(nn.Module):

    def __init__(
        self,
        text_dim,
        speech_dim,
        latent_dim,
        hidden_dim=256,
        concentration_floor=0.0001,
        encoder_layers=(4, 2, 1),
    ):
        super().__init__()
        if min(text_dim, speech_dim, latent_dim, hidden_dim, concentration_floor) <= 0:
            raise ValueError("Input, hidden and latent dimensions must be positive")
        branch_widths = layer_widths(hidden_dim, encoder_layers)
        fusion_hidden = max(32, 2 * latent_dim)
        self.text_encoder = build_mlp([text_dim, *branch_widths], nn.GELU)
        self.speech_encoder = build_mlp([speech_dim, *branch_widths], nn.GELU)
        self.fusion = nn.Sequential(nn.Linear(2 * branch_widths[-1], fusion_hidden), nn.GELU())
        self.direction_head = nn.Linear(fusion_hidden, latent_dim)
        self.concentration_head = nn.Linear(fusion_hidden, 1)
        self.text_dim = text_dim
        self.speech_dim = speech_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.concentration_floor = concentration_floor

    def forward(self, text_bow, speech_boaw):
        if text_bow.ndim != 2 or text_bow.shape[1] != self.text_dim:
            raise ValueError(f"Expected text BoW shape (batch, {self.text_dim})")
        if speech_boaw.ndim != 2 or speech_boaw.shape[1] != self.speech_dim:
            raise ValueError(f"Expected speech BoAW shape (batch, {self.speech_dim})")
        if len(text_bow) != len(speech_boaw):
            raise ValueError("Text and speech batch sizes differ")
        encoded = torch.cat((self.text_encoder(text_bow), self.speech_encoder(speech_boaw)), dim=1)
        joint = self.fusion(encoded)
        mu = F.normalize(self.direction_head(joint), dim=1)
        kappa = F.softplus(self.concentration_head(joint)).squeeze(1) + self.concentration_floor
        return dict(mu=mu, kappa=kappa)


class SphericalVAEDecoder(nn.Module):

    def __init__(self, text_dim, speech_dim, latent_dim, hidden_dim=256, decoder_layers=(1, 2, 4)):
        super().__init__()
        if min(text_dim, speech_dim, latent_dim, hidden_dim) < 1:
            raise ValueError("Decoder dimensions must be positive")
        widths = layer_widths(hidden_dim, decoder_layers)
        self.text_decoder = nn.Sequential(
            *build_mlp([latent_dim, *widths]), nn.Linear(widths[-1], text_dim)
        )
        self.speech_decoder = nn.Sequential(
            *build_mlp([latent_dim, *widths]), nn.Linear(widths[-1], speech_dim)
        )

    def forward(self, z):
        if z.ndim != 2:
            raise ValueError(f"Expected 2D latent input, got {tuple(z.shape)}")
        return (
            F.log_softmax(self.text_decoder(z), dim=1),
            F.log_softmax(self.speech_decoder(z), dim=1),
        )


class MultimodalSphericalVAE(nn.Module):

    def __init__(
        self,
        text_dim,
        speech_dim,
        latent_dim=20,
        encoder_hidden=256,
        decoder_hidden=256,
        encoder_layers=(4, 2, 1),
        decoder_layers=None,
    ):
        super().__init__()
        encoder_layers = tuple(encoder_layers)
        decoder_layers = (
            tuple(reversed(encoder_layers)) if decoder_layers is None else tuple(decoder_layers)
        )
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder = SphericalVAEEncoder(
            text_dim, speech_dim, latent_dim, encoder_hidden, encoder_layers=encoder_layers
        )
        self.decoder = SphericalVAEDecoder(
            text_dim, speech_dim, latent_dim, decoder_hidden, decoder_layers=decoder_layers
        )
        self.text_dim = text_dim
        self.speech_dim = speech_dim
        self.latent_dim = latent_dim
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden

    def encode(self, text_bow, speech_boaw):
        return self.encoder(text_bow, speech_boaw)

    def sample_posterior(self, posterior):
        return sample_vmf(posterior["mu"], posterior["kappa"])

    def decode(self, z):
        return self.decoder(z)

    def forward(self, text_bow, speech_boaw, sample=True):
        posterior = self.encode(text_bow, speech_boaw)
        z = self.sample_posterior(posterior) if sample else posterior["mu"]
        text_log_prob, speech_log_prob = self.decode(z)
        return dict(
            **posterior,
            z=z,
            text_log_prob=text_log_prob,
            speech_log_prob=speech_log_prob,
            text_reconstruction_loss=normalized_cross_entropy(text_bow, text_log_prob, "text BoW"),
            speech_reconstruction_loss=normalized_cross_entropy(
                speech_boaw, speech_log_prob, "speech BoAW"
            ),
        )
