"""Fresh BOXRR-only deterministic research adaptations, 400 wxyz -> 20 wxyz.

The ordered decoder, temporal mixers, multirate causal convolutions and tangent
DESP are adaptations, not reproductions of a paper's reported accuracy. Time is
100 Hz here; endpoint20 is 200 ms by arithmetic, despite the disputed official
100 ms wording. Geometry is shared with the audited v1 implementation.
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from .boxrr_models import (normalize_quaternion, conjugate_wxyz, multiply_wxyz,
                          log_so3, exp_so3, geodesic_radians)

MODEL_NAMES = ('ordered_v2', 'mlp_v2', 'tcn_v2', 'desp_v2')
WIDTHS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 399)


def rotation_6d(q):
    """First two columns, column-major, of the Hamilton rotation matrix."""
    w, x, y, z = normalize_quaternion(q).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y),
                        2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), -1)


def current_gravity(q):
    """World +Y expressed in current HMD coordinates; source-native Unity up.

    This is an explicit convention, not a gravity sensor measurement. Disable
    use_gravity for the controlled relative-only variant or other conventions.
    """
    w, x, y, z = normalize_quaternion(q).unbind(-1)
    return torch.stack((2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)), -1)


@dataclass
class MotionBundle:
    last: torch.Tensor
    tokens: torch.Tensor
    summary: torch.Tensor
    experts: torch.Tensor
    increments: torch.Tensor
    residual_hint: torch.Tensor | None = None


class MotionFeatures(nn.Module):
    """400 observed samples, signed newest-frame increments and continuous6D.

    No fitted scaler, user identity, future observation or external sensor.
    The optional tilt feature deliberately relaxes global-rotation invariance.
    """
    token_dim = 12
    num_experts = 15

    def __init__(self, use_gravity=True):
        super().__init__()
        self.use_gravity = bool(use_gravity)
        self.summary_dim = 240 + (3 if self.use_gravity else 0)
        self.register_buffer('steps', torch.arange(1, 21, dtype=torch.float32))
        self.register_buffer('identity6', torch.tensor([1., 0., 0., 0., 1., 0.]))
        # Causal least-squares fits to recent interval velocities. The latest
        # interval's midpoint is t=0; future interval j is t=j. Ridge only slope.
        fit = torch.zeros(3, 2, 32, dtype=torch.float64)
        for i, width in enumerate((8, 16, 32)):
            age = torch.arange(1-width, 1, dtype=torch.float64)/width
            design = torch.stack((torch.ones_like(age), age), -1)
            weight = torch.exp(age)
            gram = design.T @ (weight[:, None]*design)
            gram[1, 1] += .01
            matrix = torch.linalg.solve(gram, design.T*weight)
            matrix[1] /= width
            fit[i, :, -width:] = matrix
        self.register_buffer('fit_weights', fit.float())
        # Brown double smoothing, initialized S_0=D_0=x_0. Output is anchored
        # at q_0, hence only the extrapolated trend is retained in each expert.
        slopes=[]
        for alpha in (.15, .35, .65):
            remaining = torch.arange(63, -1, -1, dtype=torch.float64)
            single = alpha*(1-alpha)**remaining
            double = alpha**2*(remaining+1)*(1-alpha)**remaining
            single[0] = (1-alpha)**63
            double[0] = (1-alpha)**63*(1+alpha*63)
            slopes.append(alpha/(1-alpha)*(single-double))
        self.register_buffer('desp_slope_weights', torch.stack(slopes).float())

    def build_tokens(self, context, reference=None):
        """Any observed length T, returning [B,12,T]; reference is [B,4].

        For a training-only future teacher, concatenate the observed last pose
        with targets, supply that observed pose as reference, then remove token0.
        """
        q = normalize_quaternion(context)
        anchor = q[:, -1] if reference is None else normalize_quaternion(reference)
        relative = multiply_wxyz(conjugate_wxyz(anchor[:, None]), q)
        u = log_so3(multiply_wxyz(relative[:, 1:], conjugate_wxyz(relative[:, :-1])))
        u = F.pad(u, (0, 0, 1, 0))
        change = F.pad(u[:, 1:]-u[:, :-1], (0, 0, 1, 0))
        pose = rotation_6d(relative)-self.identity6
        # Fixed feature clipping is only numerical outlier protection, not a
        # target filter. No hard bound is imposed on predicted tangent residuals.
        return torch.cat((100*u, 100*change, pose), -1).clamp(-50, 50).transpose(1, 2)

    def forward(self, context):
        if context.ndim != 3 or context.shape[1:] != (400, 4):
            raise ValueError('Expected [B,400,4] observed wxyz quaternions')
        q = normalize_quaternion(context)
        last = q[:, -1]
        relative = multiply_wxyz(conjugate_wxyz(last[:, None]), q)
        increments = log_so3(multiply_wxyz(relative[:, 1:], conjugate_wxyz(relative[:, :-1])))
        means = [increments[:, -w:].mean(1) for w in WIDTHS]
        stds = [(increments[:, -w:]-mu[:, None]).square().mean(1).add(1e-10).sqrt()
                for w, mu in zip(WIDTHS, means)]
        changes = [increments[:, -w:].mean(1)-increments[:, -2*w:-w].mean(1)
                   for w in WIDTHS[:8]]
        anchors = rotation_6d(relative[:, [-1-w for w in WIDTHS]])-self.identity6
        parts = [increments[:, -32:].flatten(1)*100,
                 torch.stack(means, 1).flatten(1)*100,
                 torch.stack(stds, 1).flatten(1)*100,
                 torch.stack(changes, 1).flatten(1)*100, anchors.flatten(1)]
        if self.use_gravity:
            parts.append(current_gravity(last))
        summary = torch.cat(parts, 1).clamp(-50, 50)
        h = self.steps[None, :, None]
        trajectories = [torch.zeros_like(h*means[0][:, None])]
        trajectories += [h*mu[:, None] for mu in means[:6]]
        trajectories += [tau*(1-torch.exp(-h/tau))*means[2][:, None] for tau in (5., 10.)]
        coefficients = torch.einsum('kwt,btc->bkwc', self.fit_weights, increments[:, -32:])
        for i in range(3):
            velocity, acceleration = coefficients[:, i, 0], coefficients[:, i, 1]
            trajectories.append(h*velocity[:, None]+.5*h*(h+1)*acceleration[:, None])
        recent_pose = log_so3(relative[:, -64:])
        desp_vel = torch.einsum('kt,btc->bkc', self.desp_slope_weights, recent_pose)
        trajectories += [h*desp_vel[:, i, None] for i in range(3)]
        return MotionBundle(last, self.build_tokens(q, last), summary,
                            torch.stack(trajectories, 1), increments)


class PhysicsDecoder(nn.Module):
    """Causal expert mixture plus unconstrained signed tangent correction."""
    def __init__(self, latent_dim=128, num_experts=15, prefer_desp=False):
        super().__init__()
        self.gate = nn.Linear(latent_dim, num_experts)
        self.residual = nn.Linear(latent_dim, 60)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        prior = torch.full((num_experts,), .01)
        prior[0] = .15
        prior[3] = .65  # mean of last4 increments; all prior entries normalize.
        if prefer_desp:
            prior[3] = .1
            prior[12:15] = .25
        with torch.no_grad():
            self.gate.bias.copy_(prior.log())
        self.register_buffer('fraction', torch.arange(1, 21).view(1, 20, 1)/20.)

    def base_tangent(self, latent, bundle):
        weights = self.gate(latent).softmax(-1)
        return (weights[:, :, None, None]*bundle.experts).sum(1)

    def tangent(self, latent, bundle):
        residual = .1*self.fraction*self.residual(latent).view(-1, 20, 3)
        if bundle.residual_hint is not None:
            residual = residual+bundle.residual_hint
        return self.base_tangent(latent, bundle)+residual

    def forward(self, latent, bundle):
        return normalize_quaternion(multiply_wxyz(bundle.last[:, None], exp_so3(self.tangent(latent, bundle))))


class OrderedDecoder(PhysicsDecoder):
    """Three ordered exponentials preserve the ordered-product family."""
    def __init__(self, latent_dim, num_experts):
        super().__init__(latent_dim, num_experts)
        self.residual = nn.Linear(latent_dim, 9)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def relative(self, latent, bundle):
        b = .1*self.residual(latent).reshape(-1, 3, 3)
        s = self.fraction
        first = exp_so3(self.base_tangent(latent, bundle)+s*b[:, 0, None])
        return multiply_wxyz(multiply_wxyz(first, exp_so3(s.square()*b[:, 1, None])),
                              exp_so3(s.pow(3)*b[:, 2, None]))

    def tangent(self, latent, bundle):
        return log_so3(self.relative(latent, bundle))

    def forward(self, latent, bundle):
        return normalize_quaternion(multiply_wxyz(bundle.last[:, None], self.relative(latent, bundle)))


class ForecastBase(nn.Module):
    def decode(self, latent, bundle):
        return self.decoder(latent, bundle)

    def forward(self, context):
        return self.decode(*self.encode(context))


class OrderedMotionV2(ForecastBase):
    arm = 'ordered_v2'
    def __init__(self, width=32, latent_dim=128, use_gravity=True, **unused):
        super().__init__()
        self.latent_dim = latent_dim
        self.features = MotionFeatures(use_gravity)
        self.encoder = nn.Sequential(nn.Linear(self.features.summary_dim+21, max(width*2, 64)),
                                     nn.SiLU(), nn.Linear(max(width*2, 64), latent_dim), nn.SiLU())
        self.decoder = OrderedDecoder(latent_dim, self.features.num_experts)

    def encode(self, context):
        bundle = self.features(context)
        areas=[]
        for w in (8, 16, 32, 64, 128, 256, 399):
            u = bundle.increments[:, -w:]
            # .5 sum_(i<j) u_i cross u_j, preserving chronological order.
            prefix = u.cumsum(1)-u
            areas.append(.5*torch.linalg.cross(prefix, u).sum(1))
        features = torch.cat((bundle.summary, 1000*torch.stack(areas, 1).flatten(1)), 1)
        return self.encoder(features), bundle


class MixerBlock(nn.Module):
    def __init__(self, width, time_rank=48):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.time = nn.Sequential(nn.Linear(400, time_rank), nn.SiLU(), nn.Linear(time_rank, 400))
        self.norm2 = nn.LayerNorm(width)
        self.channels = nn.Sequential(nn.Linear(width, 2*width), nn.SiLU(), nn.Linear(2*width, width))

    def forward(self, x):
        x = x+self.time(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        return x+self.channels(self.norm2(x))


class MixerEncoder(nn.Module):
    def __init__(self, summary_dim, width=32, latent_dim=128, depth=2, time_rank=48):
        super().__init__()
        self.stem = nn.Linear(12, width)
        self.blocks = nn.Sequential(*(MixerBlock(width, time_rank) for _ in range(depth)))
        self.summary = nn.Sequential(nn.Linear(summary_dim, 64), nn.SiLU())
        self.output = nn.Sequential(nn.Linear(6*width+64, latent_dim), nn.SiLU(), nn.LayerNorm(latent_dim))

    def forward(self, summary, tokens):
        h = self.blocks(self.stem(tokens.transpose(1, 2)))
        pooled = F.adaptive_avg_pool1d(h.transpose(1, 2), 4).flatten(1)
        return self.output(torch.cat((h[:, -1], h.mean(1), pooled, self.summary(summary)), 1))


class MLPMixerV2(ForecastBase):
    arm = 'mlp_v2'
    def __init__(self, width=32, latent_dim=128, use_gravity=True, depth=2, time_rank=48, **unused):
        super().__init__()
        self.latent_dim = latent_dim
        self.features = MotionFeatures(use_gravity)
        self.encoder = MixerEncoder(self.features.summary_dim, width, latent_dim, depth, time_rank)
        self.linear_skip = nn.Linear(400, 20, bias=False)
        nn.init.zeros_(self.linear_skip.weight)
        self.decoder = PhysicsDecoder(latent_dim, self.features.num_experts)

    def encode(self, context):
        bundle = self.features(context)
        bundle.residual_hint = self.linear_skip(bundle.tokens[:, :3]/100).transpose(1, 2)
        return self.encoder(bundle.summary, bundle.tokens), bundle


class CausalDepthwiseBlock(nn.Module):
    def __init__(self, width, kernel=5, dilation=1):
        super().__init__()
        self.left = (kernel-1)*dilation
        self.depthwise = nn.Conv1d(width, width, kernel, dilation=dilation, groups=width)
        self.norm = nn.LayerNorm(width)
        self.mix = nn.Sequential(nn.Conv1d(width, 2*width, 1), nn.SiLU(), nn.Conv1d(2*width, width, 1))

    def forward(self, x):
        y = self.depthwise(F.pad(x, (self.left, 0)))
        return x+self.mix(self.norm(y.transpose(1, 2)).transpose(1, 2))


class TemporalEncoder(nn.Module):
    """Recent128 at100Hz and all400 at25Hz; endpoint anchored average pooling."""
    def __init__(self, summary_dim, width=32, latent_dim=128, kernel=5):
        super().__init__()
        self.token_dim = width
        self.stem = nn.Conv1d(12, width, 1)
        self.blocks = nn.Sequential(*(CausalDepthwiseBlock(width, kernel, d) for d in (1, 2, 4, 8, 16)))
        self.long_blocks = nn.Sequential(*(CausalDepthwiseBlock(width, kernel, d) for d in (1, 2, 4, 8, 16)))
        self.summary = nn.Sequential(nn.Linear(summary_dim, 64), nn.SiLU())
        self.output = nn.Sequential(nn.Linear(6*width+64, latent_dim), nn.SiLU(), nn.LayerNorm(latent_dim))

    def encode_tokens(self, tokens):
        return self.blocks(F.silu(self.stem(tokens)))

    @staticmethod
    def pool(x):
        return torch.cat((x[:, :, -1], x.mean(-1), x.amax(-1)), 1)

    def encode_with_sequence(self, summary, tokens):
        short = self.encode_tokens(tokens[:, :, -128:])
        # Each4-sample block ends at an observed time; never overlaps the future.
        coarse = F.avg_pool1d(tokens, 4, 4)
        long = self.long_blocks(F.silu(self.stem(coarse)))
        z = self.output(torch.cat((self.pool(short), self.pool(long), self.summary(summary)), 1))
        return z, short

    def forward(self, summary, tokens):
        return self.encode_with_sequence(summary, tokens)[0]


class TemporalV2(ForecastBase):
    arm = 'tcn_v2'
    def __init__(self, width=32, latent_dim=128, use_gravity=True, kernel=5, **unused):
        super().__init__()
        self.latent_dim = latent_dim
        self.features = MotionFeatures(use_gravity)
        self.encoder = TemporalEncoder(self.features.summary_dim, width, latent_dim, kernel)
        self.horizon_projection = nn.Linear(32, 20, bias=False)
        self.horizon_head = nn.Conv1d(width, 3, 1)
        nn.init.zeros_(self.horizon_head.weight)
        nn.init.zeros_(self.horizon_head.bias)
        self.decoder = PhysicsDecoder(latent_dim, self.features.num_experts)

    def encode(self, context):
        bundle = self.features(context)
        z, sequence = self.encoder.encode_with_sequence(bundle.summary, bundle.tokens)
        bundle.residual_hint = .1*self.horizon_head(self.horizon_projection(sequence[:, :, -32:])).transpose(1, 2)
        return z, bundle


class DESPResidualV2(ForecastBase):
    """Causal Brown tangent-trend bank with a three-convolution learned residual."""
    arm = 'desp_v2'
    def __init__(self, width=32, latent_dim=128, use_gravity=True, **unused):
        super().__init__()
        self.latent_dim = latent_dim
        self.features = MotionFeatures(use_gravity)
        self.stem = nn.Conv1d(12, width, 1)
        self.convolutions = nn.Sequential(*(CausalDepthwiseBlock(width, 7, 1) for _ in range(3)))
        self.encoder = nn.Sequential(nn.Linear(3*width+self.features.summary_dim, latent_dim), nn.SiLU(),
                                     nn.Linear(latent_dim, latent_dim), nn.SiLU())
        self.decoder = PhysicsDecoder(latent_dim, self.features.num_experts, prefer_desp=True)

    def encode(self, context):
        bundle = self.features(context)
        h = self.convolutions(F.silu(self.stem(bundle.tokens[:, :, -64:])))
        return self.encoder(torch.cat((TemporalEncoder.pool(h), bundle.summary), 1)), bundle


def build_model(name, seed=20261106, **config):
    constructors = dict(zip(MODEL_NAMES, (OrderedMotionV2, MLPMixerV2, TemporalV2, DESPResidualV2)))
    if name not in constructors:
        raise ValueError(f'Unknown deterministic arm {name!r}')
    allowed = {'width', 'latent_dim', 'use_gravity'}
    if name == 'mlp_v2':
        allowed |= {'depth', 'time_rank'}
    if name == 'tcn_v2':
        allowed.add('kernel')
    unexpected = set(config)-allowed
    if unexpected:
        raise ValueError(f'Unknown model options: {sorted(unexpected)}')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return constructors[name](**config)
