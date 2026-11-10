"""Pose-only recurrent physics hybrid for the fresh BOXRR study.

This is a project adaptation, not a reproduction or accuracy claim. MOUNT
(Chen et al., doi:10.1109/TVCG.2022.3228807, Eq.18) combines recurrent pattern
encoding with recent numerical values. Yoon et al. (ICCE2022) motivate the
physics hybrid; Illahi et al. (MMSys2023) evaluate incremental-quaternion LSTM.
Here a GRU sees 50 ordered patches covering all 400 observed poses and a direct
recent-motion bypass. Only quaternion-derived features enter inference.
"""
from __future__ import annotations

import torch
from torch import nn

from .boxrr_v2_deterministic import ForecastBase, MotionFeatures, PhysicsDecoder

MODEL_NAMES = ('gru_v2',)


class PatchGRUEncoder(nn.Module):
    """Keep the eight ordered samples inside each of 50 nonoverlapping patches.

    Flattening before projection preserves within-patch order, unlike a mean
    pool. The mean/max recurrent readouts provide short gradient paths to early
    history; the final state supplies the causal recurrent summary. All patches
    end at or before the observation boundary, and no state persists per user.
    """
    patch_size = 8
    num_patches = 50

    def __init__(self, summary_dim, width=32, latent_dim=128, token_dim=12):
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = max(32, 2*int(width))
        self.patch_projection = nn.Sequential(
            nn.Linear(self.patch_size*self.token_dim, self.hidden_dim), nn.SiLU(),
            nn.LayerNorm(self.hidden_dim))
        self.recurrent = nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=2,
                                batch_first=True, dropout=0., bidirectional=False)
        # Explicit recent angular-increment bypass (last32 observed intervals).
        self.recent = nn.Sequential(nn.Linear(3*32, 64), nn.SiLU())
        self.summary = nn.Sequential(nn.Linear(summary_dim, 64), nn.SiLU())
        self.output = nn.Sequential(
            nn.Linear(3*self.hidden_dim+128, latent_dim), nn.SiLU(),
            nn.LayerNorm(latent_dim))

    def patch_tokens(self, tokens):
        return tokens.transpose(1, 2).reshape(
            tokens.shape[0], self.num_patches, self.patch_size*self.token_dim)

    def forward(self, summary, tokens):
        patches = self.patch_projection(self.patch_tokens(tokens))
        sequence, _ = self.recurrent(patches)
        pattern = torch.cat((sequence[:, -1], sequence.mean(1), sequence.amax(1)), 1)
        recent = self.recent(tokens[:, :3, -32:].flatten(1))
        return self.output(torch.cat((pattern, recent, self.summary(summary)), 1))


class PatchGRUV2(ForecastBase):
    arm = 'gru_v2'

    def __init__(self, width=32, latent_dim=128, use_gravity=True):
        super().__init__()
        if int(width) < 1 or int(latent_dim) < 1:
            raise ValueError('width and latent_dim must be positive')
        self.latent_dim = int(latent_dim)
        self.features = MotionFeatures(use_gravity=use_gravity)
        self.encoder = PatchGRUEncoder(self.features.summary_dim, width,
                                       self.latent_dim, self.features.token_dim)
        # Same signed, scaled, unbounded tangent residual as the other v2 arms;
        # no artificial cap on learnable angle or target filtering is added.
        self.decoder = PhysicsDecoder(self.latent_dim, self.features.num_experts)

    def encode(self, context):
        bundle = self.features(context)
        return self.encoder(bundle.summary, bundle.tokens), bundle


def build_extra_model(name='gru_v2', seed=20261106, **config):
    if name not in MODEL_NAMES:
        raise ValueError(f'Unknown extra arm {name!r}')
    unexpected = set(config)-{'width', 'latent_dim', 'use_gravity'}
    if unexpected:
        raise ValueError(f'Unknown model options: {sorted(unexpected)}')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return PatchGRUV2(**config)


build_model = build_extra_model
