"""CVAE utilities for writer/domain/session-conditioned CTC-NAT."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CVAEOutput:
    latent: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor
    kl: torch.Tensor
    film_conditioning: list[tuple[torch.Tensor, torch.Tensor]]


class PosteriorEncoder(nn.Module):
    """BiGRU posterior encoder q(z | x, y)."""

    def __init__(self, hidden_size: int, latent_size: int = 64, gru_hidden_size: int = 256):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.to_mean = nn.Linear(gru_hidden_size * 2, latent_size)
        self.to_logvar = nn.Linear(gru_hidden_size * 2, latent_size)

    @torch._dynamo.disable
    def forward(
        self,
        target_embeddings: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, _ = self.gru(target_embeddings)
        if padding_mask is None:
            pooled = outputs.mean(dim=1)
        else:
            valid = (~padding_mask).unsqueeze(-1)
            summed = (outputs * valid).sum(dim=1)
            denom = valid.sum(dim=1).clamp(min=1)
            pooled = summed / denom
        return self.to_mean(pooled), self.to_logvar(pooled)


class LabelPriorEncoder(nn.Module):
    """Simple label-conditioned prior p(z | coarse labels)."""

    def __init__(
        self,
        num_writer_labels: int = 2048,
        num_domain_labels: int = 64,
        num_source_labels: int = 64,
        label_hidden_size: int = 128,
        latent_size: int = 64,
    ) -> None:
        super().__init__()
        self.writer_embedding = nn.Embedding(num_writer_labels, label_hidden_size)
        self.domain_embedding = nn.Embedding(num_domain_labels, label_hidden_size)
        self.source_embedding = nn.Embedding(num_source_labels, label_hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(label_hidden_size * 3, label_hidden_size * 2),
            nn.GELU(),
            nn.Linear(label_hidden_size * 2, latent_size * 2),
        )

    def forward(
        self,
        writer_ids: torch.Tensor | None,
        domain_ids: torch.Tensor | None,
        source_ids: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if writer_ids is None:
            writer_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        if domain_ids is None:
            domain_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        if source_ids is None:
            source_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

        x = torch.cat(
            [
                self.writer_embedding(writer_ids),
                self.domain_embedding(domain_ids),
                self.source_embedding(source_ids),
            ],
            dim=-1,
        )
        stats = self.mlp(x)
        mean, logvar = stats.chunk(2, dim=-1)
        return mean, logvar


class FiLMProjector(nn.Module):
    """Maps latent z to per-layer gamma/beta tensors."""

    def __init__(self, latent_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.to_gamma = nn.ModuleList(
            [nn.Linear(latent_size, hidden_size) for _ in range(num_layers)]
        )
        self.to_beta = nn.ModuleList(
            [nn.Linear(latent_size, hidden_size) for _ in range(num_layers)]
        )

    def forward(self, latent: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        conditioning: list[tuple[torch.Tensor, torch.Tensor]] = []
        for gamma_layer, beta_layer in zip(self.to_gamma, self.to_beta, strict=True):
            gamma = gamma_layer(latent).unsqueeze(1) + 1.0
            beta = beta_layer(latent).unsqueeze(1)
            conditioning.append((gamma, beta))
        return conditioning


class CVAEConditioner(nn.Module):
    """Full latent pathway used by the Phase 3 research prototype."""

    def __init__(
        self,
        hidden_size: int,
        num_decoder_layers: int,
        latent_size: int = 64,
        gru_hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.posterior = PosteriorEncoder(hidden_size, latent_size, gru_hidden_size)
        self.prior = LabelPriorEncoder(latent_size=latent_size)
        self.film = FiLMProjector(latent_size, hidden_size, num_decoder_layers)

    @staticmethod
    def _reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    @staticmethod
    def kl_divergence(
        q_mean: torch.Tensor,
        q_logvar: torch.Tensor,
        p_mean: torch.Tensor,
        p_logvar: torch.Tensor,
    ) -> torch.Tensor:
        q_var = torch.exp(q_logvar)
        p_var = torch.exp(p_logvar)
        kl = 0.5 * (
            p_logvar
            - q_logvar
            + (q_var + (q_mean - p_mean).pow(2)) / p_var.clamp(min=1e-6)
            - 1.0
        )
        return kl.sum(dim=-1).mean()

    def forward(
        self,
        target_embeddings: torch.Tensor | None,
        target_padding_mask: torch.Tensor | None,
        writer_ids: torch.Tensor | None,
        domain_ids: torch.Tensor | None,
        source_ids: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        sample_posterior: bool = True,
    ) -> CVAEOutput:
        prior_mean, prior_logvar = self.prior(
            writer_ids, domain_ids, source_ids, batch_size=batch_size, device=device
        )

        if target_embeddings is None:
            latent = prior_mean
            mean = prior_mean
            logvar = prior_logvar
            kl = torch.zeros((), device=device)
        else:
            post_mean, post_logvar = self.posterior(target_embeddings, target_padding_mask)
            latent = self._reparameterize(post_mean, post_logvar) if sample_posterior else post_mean
            mean = post_mean
            logvar = post_logvar
            kl = self.kl_divergence(post_mean, post_logvar, prior_mean, prior_logvar)

        return CVAEOutput(
            latent=latent,
            mean=mean,
            logvar=logvar,
            kl=kl,
            film_conditioning=self.film(latent),
        )
