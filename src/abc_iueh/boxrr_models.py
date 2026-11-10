"""Five bounded BOXRR research arms; stateless 400 -> 20 wxyz forecasting.

These are project adaptations, not reproductions of a published EBM/JEPA result.
All learned features and inference depend only on the supplied 400 past poses.
Future poses enter only training targets, including the stop-gradient JEPA teacher.
No dataset I/O, preprocessing fit, checkpoint discovery or user adaptation occurs here.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAMES = ("ordered", "residual_mlp", "tcn", "energy", "jepa")
WIDTHS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 399)


def normalize_quaternion(q):
    scale = q.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    q = q / scale
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-30)


def conjugate_wxyz(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def multiply_wxyz(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw), -1)


def log_so3(q):
    q = normalize_quaternion(q)
    v = q[..., 1:]
    first = torch.where(v[..., :1] != 0, v[..., :1],
                        torch.where(v[..., 1:2] != 0, v[..., 1:2], v[..., 2:3]))
    negative = (q[..., :1] < 0) | ((q[..., :1] == 0) & (first < 0))
    q = q * torch.where(negative, -1., 1.)
    sine = torch.linalg.vector_norm(q[..., 1:], dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sine, q[..., :1].abs())
    factor = torch.where(sine > 0, angle / sine.clamp_min(1e-30), 2.)
    return q[..., 1:] * factor


def exp_so3(v):
    angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    a2 = angle.square()
    factor = torch.where(angle > 1e-4,
                         torch.sin(angle/2) / angle.clamp_min(1e-12),
                         .5 - a2/48 + a2.square()/3840)
    return normalize_quaternion(torch.cat((torch.cos(angle/2), v*factor), -1))


def geodesic_radians(a, b):
    d = multiply_wxyz(normalize_quaternion(a), conjugate_wxyz(normalize_quaternion(b)))
    return 2 * torch.atan2(torch.linalg.vector_norm(d[..., 1:], dim=-1), d[..., 0].abs())


class MotionFeatures(nn.Module):
    """Signed, axis-aware geometry in the newest observed HMD frame.

    210 scalars: last 32 increments, ten mean/std pairs, eight recent/previous
    differences, ten relative orientations. Fixed scaling uses radians/s;
    no scaler is estimated from training or validation. Tokens retain temporal order.
    """
    summary_size = 210

    def forward(self, context):
        if context.ndim != 3 or context.shape[1:] != (400, 4):
            raise ValueError("context must have shape [B,400,4]")
        q = normalize_quaternion(context)
        last = q[:, -1]
        relative = multiply_wxyz(conjugate_wxyz(last[:, None]), q)
        increments = log_so3(multiply_wxyz(relative[:, 1:], conjugate_wxyz(relative[:, :-1])))
        means = [increments[:, -w:].mean(1) for w in WIDTHS]
        stds = [torch.sqrt((increments[:, -w:] - mu[:, None]).square().mean(1) + 1e-10)
                for w, mu in zip(WIDTHS, means)]
        changes = [increments[:, -w:].mean(1) - increments[:, -2*w:-w].mean(1) for w in WIDTHS[:8]]
        anchors = log_so3(relative[:, [-1-w for w in WIDTHS]])
        summary = torch.cat((increments[:, -32:].flatten(1)*100,
                             torch.stack(means, 1).flatten(1)*100,
                             torch.stack(stds, 1).flatten(1)*100,
                             torch.stack(changes, 1).flatten(1)*100,
                             anchors.flatten(1)), 1)
        # A fixed saturation protects extreme tracking discontinuities without a fitted scaler.
        summary = summary.clamp(-25., 25.)
        accel = increments[:, 1:] - increments[:, :-1]
        tokens = torch.cat((increments[:, -128:]*100, accel[:, -128:]*100,
                            log_so3(relative[:, -128:])), -1).transpose(1, 2).clamp(-25., 25.)
        velocities = torch.stack((torch.zeros_like(means[0]), *means[:5]), 1)
        return last, summary, tokens, velocities


class SummaryEncoder(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(210, 192), nn.SiLU(), nn.LayerNorm(192),
                                 nn.Linear(192, width), nn.SiLU())

    def forward(self, summary):
        return self.net(summary)


class CausalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.left = 2*dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        y = self.conv(F.pad(x, (self.left, 0)))
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return x + F.silu(y)


class TokenEncoder(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.stem = nn.Conv1d(9, 32, 1)
        self.blocks = nn.Sequential(*(CausalBlock(32, d) for d in (1, 2, 4, 8, 16, 32)))
        self.project = nn.Linear(96, latent)

    def forward(self, tokens):
        z = self.blocks(F.silu(self.stem(tokens)))
        return self.project(torch.cat((z[:, :, -1], z.mean(-1), z.amax(-1)), 1))


class PoseDecoder(nn.Module):
    """Causal damping mixture of ZV/CAV plus a signed tangent residual per horizon."""
    def __init__(self, latent):
        super().__init__()
        self.gate = nn.Linear(latent, 6)
        self.residual = nn.Linear(latent, 60)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.gate.weight)
        with torch.no_grad():
            self.gate.bias.copy_(torch.tensor([math.log(.5)] + [math.log(.1)]*5))
        self.register_buffer('steps', torch.arange(1, 21).view(1, 20, 1).float())

    def tangent(self, latent, velocities):
        velocity = (self.gate(latent).softmax(-1)[..., None] * velocities).sum(1)
        residual = .5*torch.tanh(self.residual(latent).view(-1, 20, 3))
        return self.steps*velocity[:, None] + (self.steps/20)*residual

    def forward(self, latent, last, velocities):
        return multiply_wxyz(last[:, None], exp_so3(self.tangent(latent, velocities)))


class ResidualMLP(nn.Module):
    arm = 'residual_mlp'

    def __init__(self):
        super().__init__()
        self.features = MotionFeatures()
        self.encoder = SummaryEncoder()
        self.decoder = PoseDecoder(128)

    def encode(self, x):
        last, summary, tokens, velocities = self.features(x)
        return self.encoder(summary), last, velocities

    def forward(self, x):
        z, last, velocities = self.encode(x)
        return self.decoder(z, last, velocities)


class TemporalModel(nn.Module):
    arm = 'tcn'

    def __init__(self):
        super().__init__()
        self.features = MotionFeatures()
        self.summary_encoder = SummaryEncoder(64)
        self.token_encoder = TokenEncoder(64)
        self.fusion = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.LayerNorm(128))
        self.decoder = PoseDecoder(128)

    def encode(self, x):
        last, summary, tokens, velocities = self.features(x)
        online = self.token_encoder(tokens)
        z = self.fusion(torch.cat((online, self.summary_encoder(summary)), -1))
        return z, last, velocities, online

    def forward(self, x):
        z, last, velocities, _ = self.encode(x)
        return self.decoder(z, last, velocities)


class ConditionalEnergy(ResidualMLP):
    """Conditional contrastive EBM with finite-support geodesic-risk inference.

    The neural energy is a bounded correction to a quadratic proposal prior.
    Training contrasts the observed endpoint against local perturbations,
    other batch endpoints, the proposal and analytic candidates. This is
    contrastive ranking, not an exact normalized likelihood claim. Inference
    softmaxes energies on a deterministic context-only candidate set and picks
    the candidate with lowest expected SO(3) distance under that distribution.
    """
    arm = 'energy'

    def __init__(self):
        super().__init__()
        self.energy_head = nn.Sequential(nn.Linear(135, 64), nn.SiLU(), nn.Linear(64, 32),
                                         nn.SiLU(), nn.Linear(32, 1))
        # 19 local candidates; seven context-only analytic candidates are added below.
        offsets = [torch.zeros(3)]
        for scale in (.03, .10, .25):
            for axis in range(3):
                value = torch.zeros(3); value[axis] = scale
                offsets.extend((value, -value))
        self.register_buffer('offsets', torch.stack(offsets))

    def energy(self, z, candidate, proposal):
        delta = (candidate-proposal[:, None])/.15
        norm = delta.square().sum(-1, keepdim=True)
        inputs = torch.cat((z[:, None].expand(-1, candidate.shape[1], -1),
                            candidate/.3, delta, norm.clamp_max(100)), -1)
        return .5*norm[..., 0] + 2*torch.tanh(self.energy_head(inputs)[..., 0])

    def candidates(self, proposal, velocities):
        local = proposal[:, None] + self.offsets[None]
        analytic = 20*velocities
        # Damped CAV1 is useful where active movement changes direction.
        return torch.cat((local, analytic, 10*velocities[:, 1:2]), 1)

    def decode_energy(self, z, tangent, velocities):
        candidate = self.candidates(tangent[:, -1], velocities)
        energies = self.energy(z, candidate, tangent[:, -1])
        weights = (-energies).softmax(-1)
        # Inference is discrete; training the energy uses an explicit contrastive objective.
        with torch.no_grad():
            rotations = exp_so3(candidate.detach())
            distance = geodesic_radians(rotations[:, :, None], rotations[:, None, :])
            risks = (distance * weights.detach()[:, None, :]).sum(-1)
            selected = risks.argmin(-1)
        chosen = candidate[torch.arange(len(candidate), device=candidate.device), selected]
        return tangent + self.decoder.steps/20*(chosen-tangent[:, -1])[:, None]

    def forward(self, x):
        z, last, velocities = self.encode(x)
        tangent = self.decoder.tangent(z, velocities)
        tangent = self.decode_energy(z, tangent, velocities)
        return multiply_wxyz(last[:, None], exp_so3(tangent))


def future_tokens(last, future):
    """Teacher-only future patch; never called by an inference forward pass."""
    rel = multiply_wxyz(conjugate_wxyz(last[:, None]), normalize_quaternion(future))
    identity = torch.zeros_like(rel[:, :1]); identity[..., 0] = 1.
    previous = torch.cat((identity, rel[:, :-1]), 1)
    u = log_so3(multiply_wxyz(rel, conjugate_wxyz(previous)))
    a = torch.cat((torch.zeros_like(u[:, :1]), u[:, 1:] - u[:, :-1]), 1)
    return torch.cat((u*100, a*100, log_so3(rel)), -1).transpose(1, 2).clamp(-25., 25.)


class JointEmbeddingPredictor(TemporalModel):
    """Joint supervised/JEPA adaptation with an EMA future-patch teacher."""
    arm = 'jepa'

    def __init__(self):
        super().__init__()
        self.predictor = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 64))
        self.target_encoder = copy.deepcopy(self.token_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.ema_momentum = .998

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self


def _supervised(prediction, target):
    error = geodesic_radians(prediction, target)
    return error[:, -1].mean() + .2*error.mean(), error[:, -1].mean()


def _variance_covariance(z):
    centered = z-z.mean(0)
    variance = F.relu(1-torch.sqrt(centered.square().mean(0)+1e-4)).mean()
    covariance = centered.T@centered/max(len(z)-1, 1)
    off_diagonal = covariance-torch.diag_embed(torch.diagonal(covariance))
    return variance, off_diagonal.square().sum()/z.shape[1]


def training_loss(model, x, y, step=0, total_steps=20000):
    """Loss and detached diagnostics; call after_optimizer_step after each update."""
    if x.shape[1:] != (400, 4) or y.shape[1:] != (20, 4) or len(x) != len(y):
        raise ValueError('training requires context[B,400,4] and target[B,20,4]')
    if getattr(model, 'arm', 'ordered') == 'ordered':
        endpoint = geodesic_radians(model(x)[:, -1], y[:, -1]).mean()
        return endpoint, {'endpoint_radians': endpoint.detach()}
    if isinstance(model, ConditionalEnergy):
        z, last, velocities = model.encode(x)
        tangent = model.decoder.tangent(z, velocities)
        prediction = multiply_wxyz(last[:, None], exp_so3(tangent))
        supervised, endpoint = _supervised(prediction, y)
        positive = log_so3(multiply_wxyz(conjugate_wxyz(last), y[:, -1]))
        scales = positive.new_tensor([.02, .05, .10, .25, .50]).repeat_interleave(2)
        perturbed = positive[:, None]+torch.randn(len(x), len(scales), 3, device=x.device)*scales[None, :, None]
        # Detached proposals prevent winning ranking by moving the candidate support.
        negative = torch.cat((perturbed, positive.roll(1, 0)[:, None],
                              model.candidates(tangent[:, -1].detach(), velocities).detach()), 1)
        candidates = torch.cat((positive[:, None], negative), 1)
        energy = model.energy(z, candidates, tangent[:, -1].detach())
        contrastive = F.cross_entropy(-energy, torch.zeros(len(x), device=x.device, dtype=torch.long))
        loss = supervised+.03*contrastive
        return loss, {'endpoint_radians': endpoint.detach(), 'energy_contrastive': contrastive.detach()}
    if isinstance(model, JointEmbeddingPredictor):
        z, last, velocities, online = model.encode(x)
        prediction = model.decoder(z, last, velocities)
        supervised, endpoint = _supervised(prediction, y)
        ft = future_tokens(last, y)
        with torch.no_grad():
            teacher = model.target_encoder(ft)
        predicted = model.predictor(z)
        latent = F.l1_loss(F.layer_norm(predicted, (64,)), F.layer_norm(teacher, (64,)))
        online_future = model.token_encoder(ft)
        var1, cov1 = _variance_covariance(online)
        var2, cov2 = _variance_covariance(online_future)
        variance = (var1+var2)/2
        covariance = (cov1+cov2)/2
        ramp = min(1., (step+1)/max(1, min(2000, total_steps//10)))
        loss = supervised+ramp*(.01*latent+.01*variance+.001*covariance)
        return loss, {'endpoint_radians': endpoint.detach(), 'jepa_latent_l1': latent.detach(),
                      'jepa_variance': variance.detach(), 'jepa_covariance': covariance.detach(),
                      'jepa_online_std': online.detach().std(0, unbiased=False).mean(),
                      'jepa_teacher_std': teacher.detach().std(0, unbiased=False).mean()}
    supervised, endpoint = _supervised(model(x), y)
    return supervised, {'endpoint_radians': endpoint.detach()}


@torch.no_grad()
def after_optimizer_step(model):
    if isinstance(model, JointEmbeddingPredictor):
        for target, online in zip(model.target_encoder.parameters(), model.token_encoder.parameters()):
            target.lerp_(online, 1-model.ema_momentum)


def build_model(name, baseline_module=None, seed=20261106):
    if name not in MODEL_NAMES:
        raise ValueError(f'Unknown model: {name}')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if name == 'ordered':
            if baseline_module is None:
                raise ValueError('ordered requires the independently SHA-verified baseline module')
            model = baseline_module.OrderedMotionProduct(seed=seed)
            model.arm = 'ordered'
            return model
        return {'residual_mlp': ResidualMLP, 'tcn': TemporalModel,
                'energy': ConditionalEnergy, 'jepa': JointEmbeddingPredictor}[name]()
