import logging
from time import perf_counter
import numpy as np
import torch
from model import check_finite, check_mixture, check_unit, VMFMixture

LOGGER = logging.getLogger("stfed")


@torch.no_grad()
def federated_average(states, sizes):
    weights = np.asarray(sizes, dtype=np.float64)
    if (
        not states
        or len(states) != len(weights)
        or (not np.isfinite(weights).all())
        or (weights <= 0).any()
    ):
        raise ValueError("Provide a positive sample count for each client")
    if any((set(state) != set(states[0]) for state in states)):
        raise ValueError("Client parameter names differ")
    weights /= weights.sum()
    averaged = {}
    for name, reference in states[0].items():
        value = torch.zeros_like(reference)
        for state, weight in zip(states, weights):
            if state[name].shape != reference.shape:
                raise ValueError(f"Client parameter shape differs: {name}")
            value.add_(state[name].detach(), alpha=float(weight))
        averaged[name] = value
    return averaged


def broadcast(server, clients):
    shared = server.shared_model.state_dict()
    for client in clients:
        client.shared_model.load_state_dict(shared)


def aggregate(server, clients, sizes):
    states = [dict(client.shared_model.named_parameters()) for client in clients]
    averaged = federated_average(states, sizes)
    if set(averaged) != set(states[0]):
        raise RuntimeError("Aggregation skipped an unexpected shared parameter")
    state = server.shared_model.state_dict()
    state.update(averaged)
    server.shared_model.load_state_dict(state)


def optimization_step(model, optimizer, loss, scope):
    check_finite(loss, f"{scope} total loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            check_finite(parameter.grad, f"{scope} gradient {name}")
    optimizer.step()
    for module in model.modules():
        project = getattr(module, "project_parameters_", None)
        if project is not None:
            project()
    for name, parameter in model.named_parameters():
        check_finite(parameter, f"{scope} parameter {name}")


def train_spherical_vae_local(
    client, loader, epochs, lr, lambda_text, lambda_speech, device, lambda_cluster=0.0, sample=True
):
    client.train()
    optimizer = torch.optim.Adam(client.parameters(), lr=lr, weight_decay=1.2e-06)
    total, count = (0.0, 0)
    for _ in range(epochs):
        for text, speech in loader:
            out = client(
                text.to(device),
                speech.to(device),
                lambda_text,
                lambda_speech,
                lambda_cluster,
                sample=sample,
            )
            optimization_step(client, optimizer, out["total_loss"], "local")
            total += out["total_loss"].item() * len(text)
            count += len(text)
    if not count:
        raise ValueError("Cannot train on an empty client loader")
    return total / count


def spherical_vae_client_round(
    server, clients, loaders, sizes, args, lambda_cluster=0.0, sample=True, lr=None
):
    LOGGER.info(
        "[FEDAVG] broadcasting shared VAE to %d clients | cluster_weight=%.3f | sample=%s",
        len(clients),
        lambda_cluster,
        sample,
    )
    broadcast(server, clients)
    learning_rate = args.lr if lr is None else lr
    losses = []
    for index, (client, loader, size) in enumerate(zip(clients, loaders, sizes), start=1):
        started = perf_counter()
        LOGGER.info(
            "[CLIENT] %d/%d START | samples=%d | local_epochs=%d",
            index,
            len(clients),
            size,
            args.local_epochs,
        )
        loss = train_spherical_vae_local(
            client,
            loader,
            args.local_epochs,
            learning_rate,
            args.lambda_text,
            args.lambda_speech,
            args.device,
            lambda_cluster,
            sample=sample,
        )
        losses.append(loss)
        LOGGER.info(
            "[CLIENT] %d/%d DONE | finite=OK | average_loss=%.6f | elapsed=%.1fs",
            index,
            len(clients),
            loss,
            perf_counter() - started,
        )
    LOGGER.info("[FEDAVG] aggregating shared VAE parameters by client sample count")
    aggregate(server, clients, sizes)
    weighted_loss = float(np.average(losses, weights=sizes))
    LOGGER.info("[FEDAVG] aggregation complete | weighted_loss=%.6f", weighted_loss)
    return weighted_loss


@torch.no_grad()
def collect_z(client, loader, device):
    client.eval()
    values = [client.encode(text.to(device), speech.to(device)).cpu() for text, speech in loader]
    if not values:
        raise ValueError("Cannot collect Z from an empty client loader")
    z = torch.cat(values)
    check_unit(z, "spherical VAE Z")
    return z


def initialize_mixture(mixture, directions, max_iter):
    check_unit(directions, "VMFMixture initialization mu")
    fitted = VMFMixture(n_cluster=mixture.n_cluster, max_iter=max_iter)
    fitted.fit(directions.detach().cpu().double().numpy())
    pi, mu, kappa = [torch.as_tensor(x) for x in (fitted.pi, fitted.xi, fitted.k)]
    for value, name in ((pi, "pi"), (mu, "mu"), (kappa, "kappa")):
        check_finite(value, f"VMFMixture initialization {name}")
    check_unit(mu, "VMFMixture initialization mu_c")
    if (pi < 0).any() or not torch.isclose(pi.sum(), pi.new_tensor(1.0), atol=1e-05):
        raise ValueError("VMFMixture initialization: invalid mixture weights")
    if not (kappa > 0).all():
        raise ValueError("VMFMixture initialization: non-positive concentration")
    mixture.initialize_from_parameters(pi, mu, kappa)
    check_mixture(mixture, "initialized VMFMM")


@torch.no_grad()
def update_global_mixture(server, clients, max_iter):
    if not clients:
        raise ValueError("At least one client is required")
    centers = []
    for client in clients:
        check_mixture(client.vmfmm, "client mixture")
        centers.append(client.vmfmm.get_parameters()[1].detach().cpu())
    initialize_mixture(server.vmfmm, torch.cat(centers), max_iter)
    LOGGER.info(
        "[GLOBAL] mixture updated from %d client centers", sum((len(value) for value in centers))
    )
