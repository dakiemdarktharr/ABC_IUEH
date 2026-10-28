"""Compact angular-exposure pilot models; no data access or fitting at import.

All arms use the same 1,360-parameter causal encoder and physical-unit tokens.
The ordered-label median is optimal for snapped short-arc targets only, not
for arbitrary SO(3) targets. Training losses use the explicit units below;
evaluation must independently score actual output20 quaternions in degrees.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ARMS = ("nll21", "scalar_angle", "scalar_pinball", "risk21", "vector_angle")
PARAMETER_COUNTS = {"nll21": 1717, "scalar_angle": 1649,
                    "scalar_pinball": 1649, "risk21": 1717, "vector_angle": 1683}
LOSS_SCALING = {
    "nll21": {"unit": "dimensionless", "formula": "categorical NLL with uniform exact-minimum ties"},
    "scalar_angle": {"unit": "radian", "scale": 1.0},
    "scalar_pinball": {"unit": "dimensionless", "formula": "sum_j nearest_probability_j * 0.5*abs(s-j)/20"},
    "risk21": {"unit": "radian_squared", "formula": "mean_j (predicted_cost_j-actual_angle_j)^2"},
    "vector_angle": {"unit": "radian", "scale": 1.0},
}


def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Normalize nonzero wxyz quaternions; norms use a finite zero subgradient."""
    # Inputs are finite unit quaternions by contract. Scale first to keep
    # moderately nonunit intermediates from overflowing their squared norm.
    scale = q.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    scaled = q / scale
    return scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True).clamp_min(1e-30)


def multiply_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack((aw*bw - ax*bx - ay*by - az*bz,
                        aw*bx + ax*bw + ay*bz - az*by,
                        aw*by - ax*bz + ay*bw + az*bx,
                        aw*bz + ax*by - ay*bx + az*bw), dim=-1)


def conjugate_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def log_so3(q: torch.Tensor) -> torch.Tensor:
    """Principal SO(3) log, radians; canonical sign also fixes exact half turns."""
    q = normalize_quaternion(q)
    scalar, vector = q[..., :1], q[..., 1:]
    first = torch.where(vector[..., :1] != 0, vector[..., :1],
                        torch.where(vector[..., 1:2] != 0, vector[..., 1:2], vector[..., 2:3]))
    negative = (scalar < 0) | ((scalar == 0) & (first < 0))
    q = q * torch.where(negative, -1.0, 1.0)
    sine = torch.linalg.vector_norm(q[..., 1:], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sine, q[..., :1].abs())
    factor = torch.where(sine > 0, angle / sine.clamp_min(1e-30), 2.0)
    return q[..., 1:] * factor


def exp_so3(v: torch.Tensor) -> torch.Tensor:
    """SO(3) exponential with finite derivatives at a learned zero residual.

    ``sqrt(sum(v*v))`` has an undefined backward expression at zero in this
    use. vector_norm supplies the zero subgradient and the small-angle series
    avoids dividing by an unprotected zero in either torch.where branch.
    """
    angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    angle2 = angle * angle
    small_factor = 0.5 - angle2 / 48.0 + angle2 * angle2 / 3840.0
    factor = torch.where(angle > 1e-4,
                         torch.sin(angle * 0.5) / angle.clamp_min(1e-12), small_factor)
    q = torch.cat((torch.cos(angle * 0.5), v * factor), dim=-1)
    return normalize_quaternion(q)


def geodesic_radians(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Actual rotation angle, shape without the last quaternion dimension.

    Uses atan2 rather than acos near identical quaternions; the loss is
    nondifferentiable at exact coincidence/half-turn as any angular distance
    is, with finite PyTorch subgradients. No smooth surrogate is substituted.
    """
    delta = multiply_wxyz(normalize_quaternion(a), conjugate_wxyz(normalize_quaternion(b)))
    sine = torch.linalg.vector_norm(delta[..., 1:], dim=-1)
    return 2.0 * torch.atan2(sine, delta[..., 0].abs())


def nearest_probabilities(costs: torch.Tensor) -> torch.Tensor:
    """Uniform over exactly tied finite minimum costs; no tolerance is fitted."""
    tied = (costs == costs.amin(dim=-1, keepdim=True)).to(costs.dtype)
    return tied / tied.sum(dim=-1, keepdim=True)


class CausalBlock(nn.Module):
    def __init__(self, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.depth = nn.Conv1d(16, 16, 5, dilation=dilation, groups=16)
        self.point = nn.Conv1d(16, 16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.depth(F.pad(x, (4 * self.dilation, 0)))
        return F.relu(x + self.point(F.relu(z)))


class ArcEncoder(nn.Module):
    """341-increment receptive field; current-frame anchoring is stateless."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(3, 16, 5)
        self.blocks = nn.ModuleList(CausalBlock(d) for d in (4, 16, 64))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(F.pad(tokens, (4, 0))))
        for block in self.blocks:
            x = block(x)
        return x[:, :, -1]


class ArcModel(nn.Module):
    """Five controlled pilot arms with a shared deterministic encoder init.

    ``logits`` returns the untransformed head, not necessarily probabilities.
    ``forward`` emits unit wxyz [B,20,4]. NLL decoder choices median/mean/map
    share exactly the same fitted probabilities. ``training_loss`` returns a
    scalar or per-example values and never loads data or fits statistics.
    """
    actual_context_observations = 342
    receptive_field_increments = 341

    def __init__(self, arm: str, seed: int = 20261026):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arc arm: {arm!r}")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.arm, self.seed = arm, int(seed)
        # Layer constructors consume torch RNG, but the process's RNG state
        # must not change when constructing this deterministic local model.
        with torch.random.fork_rng(devices=[]):
            self.encoder = ArcEncoder()
            if arm in ("nll21", "risk21"):
                self.head = nn.Linear(16, 21)
            else:
                self.head = nn.Sequential(nn.Linear(16, 16), nn.ReLU(),
                                          nn.Linear(16, 3 if arm == "vector_angle" else 1))
        rng = np.random.default_rng(self.seed)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name.endswith("bias"):
                    p.zero_()
                else:
                    p.copy_(torch.from_numpy(rng.normal(0.0, 0.02, tuple(p.shape)).astype(np.float32)))
        self.register_buffer("candidate_indices", torch.arange(21, dtype=torch.float32).view(1, 21))
        self.register_buffer("horizons", torch.arange(1, 21, dtype=torch.float32).view(1, 20, 1))
        self.register_buffer("cdf_matrix", torch.triu(torch.ones(21, 21, dtype=torch.float32)))

    @staticmethod
    def _validate_context(context: torch.Tensor) -> None:
        if not torch.jit.is_tracing():
            if context.ndim != 3 or context.shape[1:] != (400, 4) or context.dtype != torch.float32:
                raise ValueError("context must be float32 [B,400,4]")

    def encoder_sha256(self) -> str:
        """Host-side metadata only; canonical names, shapes and FP32 bytes."""
        digest = hashlib.sha256()
        for name, value in self.encoder.state_dict().items():
            digest.update(json.dumps([name, list(value.shape)], separators=(",", ":")).encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes())
        return digest.hexdigest()

    def features(self, context: torch.Tensor) -> torch.Tensor:
        self._validate_context(context)
        q = normalize_quaternion(context[:, -342:])
        current = q[:, -1:]
        world = multiply_wxyz(q[:, 1:], conjugate_wxyz(q[:, :-1]))
        body = multiply_wxyz(multiply_wxyz(conjugate_wxyz(current), world), current)
        # Fixed factor100 changes radians/sample to the nominal physical scale;
        # it does not reconcile the disputed organizer horizon clock.
        return (100.0 * log_so3(body)).transpose(1, 2)

    def logits(self, context: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.features(context)))

    @staticmethod
    def _current_velocity(context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = normalize_quaternion(context[:, -2:])
        current = q[:, -1:]
        velocity = log_so3(multiply_wxyz(current, conjugate_wxyz(q[:, :1])))
        return current, velocity

    @staticmethod
    def _short_arc(velocity: torch.Tensor) -> torch.Tensor:
        return 20.0 * torch.linalg.vector_norm(velocity, dim=-1, keepdim=True) < torch.pi

    def endpoint_candidates(self, context: torch.Tensor) -> torch.Tensor:
        self._validate_context(context)
        current, velocity = self._current_velocity(context)
        rotations = exp_so3(self.candidate_indices.unsqueeze(-1) * velocity)
        return normalize_quaternion(multiply_wxyz(rotations, current))

    def candidate_costs(self, context: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing():
            if (target_q.ndim != 2 or target_q.shape != (context.shape[0], 4)
                    or target_q.dtype != torch.float32):
                raise ValueError("target_q must be float32 [B,4]")
        return geodesic_radians(self.endpoint_candidates(context), target_q[:, None, :])

    def decode_exposure(self, context: torch.Tensor, exposure: torch.Tensor) -> torch.Tensor:
        """Decode an already computed [B] exposure in [0,20], with CAV fallback.

        This pure readout lets fixed priors and all NLL decisions share the
        exact same geometry. No network is evaluated and no statistic is fit.
        """
        self._validate_context(context)
        if not torch.jit.is_tracing():
            if exposure.shape != (context.shape[0],) or exposure.dtype != torch.float32:
                raise ValueError("exposure must be float32 [B]")
        current, velocity = self._current_velocity(context)
        exposure = exposure.view(-1, 1, 1)
        exposure = torch.where(self._short_arc(velocity), exposure, torch.full_like(exposure, 20.0))
        rotation = exp_so3(torch.minimum(self.horizons, exposure) * velocity)
        return normalize_quaternion(multiply_wxyz(rotation, current))

    def decode_probabilities(self, context: torch.Tensor, probabilities: torch.Tensor,
                             decode: str = "median") -> torch.Tensor:
        """Decode supplied normalized [B,21] probabilities without the encoder."""
        if not torch.jit.is_tracing():
            if probabilities.shape != (context.shape[0], 21) or probabilities.dtype != torch.float32:
                raise ValueError("probabilities must be float32 [B,21]")
        if decode == "median":
            cdf = probabilities @ self.cdf_matrix
            exposure = (cdf[:, :20] < 0.5).to(probabilities.dtype).sum(dim=-1)
        elif decode == "mean":
            exposure = (probabilities * self.candidate_indices).sum(dim=-1)
        elif decode == "map":
            exposure = probabilities.argmax(dim=-1).to(probabilities.dtype)
        else:
            raise ValueError("nll21 decode must be median, mean or map")
        return self.decode_exposure(context, exposure)

    def exposure(self, output: torch.Tensor, decode: str = "median") -> torch.Tensor:
        """Scalar exposure [B,1,1]; scalar support is continuous [0,20]."""
        if self.arm in ("scalar_angle", "scalar_pinball"):
            return (20.0 * torch.sigmoid(output)).view(-1, 1, 1)
        if self.arm == "risk21":
            # ArgMin is explicit. It is not TopK, but its NPU mapping remains
            # unmeasured; this research module makes no export-speed claim.
            return output.argmin(dim=-1).to(output.dtype).view(-1, 1, 1)
        if self.arm != "nll21":
            raise ValueError("vector_angle has no scalar exposure")
        probabilities = torch.softmax(output, dim=-1)
        if decode == "median":
            cdf = probabilities @ self.cdf_matrix
            result = (cdf[:, :20] < 0.5).to(output.dtype).sum(dim=-1)
        elif decode == "mean":
            result = (probabilities * self.candidate_indices).sum(dim=-1)
        elif decode == "map":
            result = probabilities.argmax(dim=-1).to(output.dtype)
        else:
            raise ValueError("nll21 decode must be median, mean or map")
        return result.view(-1, 1, 1)

    def _prediction(self, context: torch.Tensor, output: torch.Tensor, decode: str) -> torch.Tensor:
        current, velocity = self._current_velocity(context)
        if self.arm == "vector_angle":
            pure_velocity = torch.cat((torch.zeros_like(velocity[..., :1]), velocity), dim=-1)
            body_velocity = multiply_wxyz(multiply_wxyz(conjugate_wxyz(current), pure_velocity), current)[..., 1:]
            tangent = self.horizons * body_velocity + (self.horizons / 20.0) * output[:, None, :]
            prediction = multiply_wxyz(current, exp_so3(tangent))
            cav = multiply_wxyz(exp_so3(self.horizons * velocity), current)
            prediction = torch.where(self._short_arc(velocity), prediction, cav)
        else:
            exposure = self.exposure(output, decode)
            return self.decode_exposure(context, exposure.view(-1))
        return normalize_quaternion(prediction)

    def forward(self, context: torch.Tensor, decode: str = "median") -> torch.Tensor:
        return self._prediction(context, self.logits(context), decode)

    def training_loss(self, context: torch.Tensor, target_q: torch.Tensor,
                      reduction: str = "mean", candidate_costs: torch.Tensor | None = None) -> torch.Tensor:
        """Frozen natural-unit losses; ``none`` returns one value per example.

        Geometry targets/nearest labels are detached training labels. Models
        receive only context. Risk21 regresses true unsnapped angular costs;
        it is not an energy-density or nearest-label classification loss.
        """
        if not torch.jit.is_tracing():
            if target_q.shape != (context.shape[0], 4) or target_q.dtype != torch.float32:
                raise ValueError("target_q must be float32 [B,4]")
            if candidate_costs is not None and (candidate_costs.shape != (context.shape[0], 21)
                    or candidate_costs.dtype != torch.float32 or candidate_costs.device != context.device):
                raise ValueError("cached candidate_costs must be float32 [B,21] on the context device")
        output = self.logits(context)
        if self.arm in ("scalar_angle", "vector_angle"):
            loss = geodesic_radians(self._prediction(context, output, "median")[:, -1], target_q)
        else:
            with torch.no_grad():
                costs = (self.candidate_costs(context, target_q) if candidate_costs is None
                         else candidate_costs.detach())
                probabilities = nearest_probabilities(costs)
            if self.arm == "nll21":
                loss = -(probabilities * F.log_softmax(output, dim=-1)).sum(dim=-1)
            elif self.arm == "risk21":
                loss = ((output - costs) ** 2).mean(dim=-1)
            else:
                s = self.exposure(output).view(-1, 1)
                loss = (probabilities * (0.5 / 20.0) * (s - self.candidate_indices).abs()).sum(dim=-1)
        if reduction == "none":
            return loss
        if reduction == "sum":
            return loss.sum()
        if reduction == "mean":
            return loss.mean()
        raise ValueError("reduction must be none, sum or mean")
