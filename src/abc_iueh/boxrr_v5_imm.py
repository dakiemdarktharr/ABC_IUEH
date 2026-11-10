"""Bounded causal SO(3) CV/CA IMM and an unrestricted neural correction.

All rotations/rates live in the fixed frame of the newest observed pose.  The
error convention is left boxplus: (R,w,a) + e = (Exp(e_R) R,w+e_w,a+e_a).
CV has six active error coordinates; its common nine-coordinate embedding has
exactly zero acceleration and acceleration uncertainty.  No 9x9 inverse is used.
The filter is detached and has only global, serialized scalar hyperparameters.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .boxrr_models import (
    conjugate_wxyz, exp_so3, log_so3, multiply_wxyz, normalize_quaternion,
)


def _observed_unit(q):
    """Identity fallback before normalization for an invalid literal-zero pose."""
    q = q.float()
    identity = torch.zeros_like(q)
    identity[..., 0] = 1.
    return normalize_quaternion(torch.where(q.abs().amax(-1, keepdim=True) > 0, q, identity))


def _skew(v):
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(*v.shape[:-1], 3, 3)


def _so3_matrices(v):
    """Analytic Exp rotation, left Jacobian and inverse left Jacobian."""
    theta2 = v.square().sum(-1, keepdim=True)
    theta = theta2.clamp_min(1e-16).sqrt()
    small = theta2 < 1e-4
    a = torch.where(small, 1-theta2/6+theta2.square()/120,
                    torch.sin(theta)/theta)
    b = torch.where(small, .5-theta2/24+theta2.square()/720,
                    (1-torch.cos(theta))/theta2.clamp_min(1e-16))
    c = torch.where(small, 1/6-theta2/120+theta2.square()/5040,
                    (theta-torch.sin(theta))/(theta*theta2).clamp_min(1e-16))
    d = torch.where(small, 1/12+theta2/720+theta2.square()/30240,
                    (1-.5*theta*torch.cos(theta/2)/torch.sin(theta/2).clamp_min(1e-8))
                    / theta2.clamp_min(1e-16))
    cross = _skew(v)
    square = cross @ cross
    eye = torch.eye(3, dtype=v.dtype, device=v.device)
    return (eye+a[..., None]*cross+b[..., None]*square,
            eye+b[..., None]*cross+c[..., None]*square,
            eye-.5*cross+d[..., None]*square)


def _transport_matrix(orientation):
    eye = torch.eye(9, dtype=orientation.dtype, device=orientation.device)
    result = eye.expand(*orientation.shape[:-2], 9, 9).clone()
    result[..., :3, :3] = orientation
    return result


def _align_to_previous(prediction, anchor):
    previous = torch.cat((anchor[:, None], prediction[:, :-1]), 1)
    signs = torch.where((previous*prediction).sum(-1, keepdim=True) < 0, -1., 1.)
    return prediction*signs.cumprod(1)


class IMMResidual(nn.Module):
    """Stateless float32 [B,400,4] -> [B,20,4] wxyz research candidate.

    ``interaction=False`` keeps independent CV/CA state/covariance recursions
    with the same transition/evidence probability update and identical head.
    ``mode_evidence=False`` instead freezes equal probabilities as a control.
    Noise defaults are engineering initial values, not fitted or validated.
    """
    arm = "imm_so3_neural_residual"

    def __init__(self, width=64, updates=31, warmup_increments=8, dt=.01,
                 interaction=True, measurement_sigma=.01, cv_accel_noise=4.,
                 ca_jerk_noise=40., dwell_cv=.15, dwell_ca=.15,
                 initial_orientation_sigma=.01, initial_velocity_sigma=1.,
                 initial_acceleration_sigma=10., covariance_floor=1e-9,
                 reanchor=True, decoder="mean", mean_iterations=4,
                 risk_iterations=8, mode_evidence=True):
        super().__init__()
        if (width < 1 or int(width) != width or updates < 1 or int(updates) != updates
                or warmup_increments < 1 or int(warmup_increments) != warmup_increments
                or updates+warmup_increments+1 > 400):
            raise ValueError("positive integer width/updates/warmup and support <=400 required")
        if dt != .01:
            raise ValueError("BOXRR filtering requires dt=.01 seconds (100 Hz)")
        positive = (dt, measurement_sigma, cv_accel_noise, ca_jerk_noise, dwell_cv,
                    dwell_ca, initial_orientation_sigma, initial_velocity_sigma,
                    initial_acceleration_sigma, covariance_floor)
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("time, noise, dwell, initial scales and floor must be finite positive")
        if decoder not in ("mean", "argmax", "risk"):
            raise ValueError("decoder must be mean, argmax or risk")
        if (mean_iterations < 1 or int(mean_iterations) != mean_iterations
                or risk_iterations < 1 or int(risk_iterations) != risk_iterations):
            raise ValueError("decoder iteration counts must be positive integers")
        self.width, self.updates = int(width), int(updates)
        self.warmup_increments = int(warmup_increments)
        self.dt = float(dt)
        self.interaction, self.reanchor = bool(interaction), bool(reanchor)
        self.measurement_sigma = float(measurement_sigma)
        self.cv_accel_noise, self.ca_jerk_noise = float(cv_accel_noise), float(ca_jerk_noise)
        self.dwell_cv, self.dwell_ca = float(dwell_cv), float(dwell_ca)
        self.initial_orientation_sigma = float(initial_orientation_sigma)
        self.initial_velocity_sigma = float(initial_velocity_sigma)
        self.initial_acceleration_sigma = float(initial_acceleration_sigma)
        self.covariance_floor = float(covariance_floor)
        self.decoder = decoder
        self.mean_iterations, self.risk_iterations = int(mean_iterations), int(risk_iterations)
        self.mode_evidence = bool(mode_evidence)
        self.support = self.updates+self.warmup_increments+1
        stay_cv, stay_ca = math.exp(-dt/dwell_cv), math.exp(-dt/dwell_ca)
        self.register_buffer("transition", torch.tensor([[stay_cv, 1-stay_cv],
                                                         [1-stay_ca, stay_ca]]))
        self.register_buffer("active", torch.tensor([[1.]*6+[0.]*3, [1.]*9]))
        self.register_buffer("feature_scales", torch.tensor([.1]*3+[1.]*3+[10.]*3))
        # Both controls instantiate exactly these modules and feature dimensions.
        self.raw_encoder = nn.Sequential(nn.Linear((2*self.support-1)*3, width),
                                         nn.SiLU(), nn.LayerNorm(width))
        self.state_encoder = nn.Sequential(nn.Linear(190, width), nn.SiLU(),
                                           nn.LayerNorm(width))
        self.fusion = nn.Sequential(nn.Linear(2*width, width), nn.SiLU())
        self.residual = nn.Linear(width, 60)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def get_config(self):
        names = ("width", "updates", "warmup_increments", "dt", "interaction",
                 "measurement_sigma", "cv_accel_noise", "ca_jerk_noise", "dwell_cv",
                 "dwell_ca", "initial_orientation_sigma", "initial_velocity_sigma",
                 "initial_acceleration_sigma", "covariance_floor", "reanchor",
                 "decoder", "mean_iterations", "risk_iterations", "mode_evidence")
        return {name: getattr(self, name) for name in names}

    def _observations(self, context):
        if context.ndim != 3 or context.shape[1:] != (400, 4):
            raise ValueError("context must have shape [B,400,4]")
        q = _observed_unit(context[:, -self.support:])
        anchor = q[:, -1]
        return anchor, multiply_wxyz(conjugate_wxyz(anchor[:, None]), q)

    def _clean_covariance(self, covariance):
        active = self.active[None]
        covariance = (covariance+covariance.transpose(-1, -2))*.5
        covariance = covariance*active[..., :, None]*active[..., None, :]
        return covariance+torch.diag_embed(active*self.covariance_floor)

    def _interact(self, q, velocity, acceleration, covariance, probabilities):
        prior = probabilities @ self.transition
        if not self.interaction:
            return q, velocity, acceleration, covariance, prior
        weights = probabilities[:, :, None]*self.transition[None]/prior[:, None, :]
        # The two-rotation intrinsic mean has a closed shortest-geodesic form.
        between = log_so3(multiply_wxyz(q[:, 1], conjugate_wxyz(q[:, 0])))
        mean_q = multiply_wxyz(exp_so3(weights[:, 1, :, None]*between[:, None]), q[:, :1])
        mean_v = torch.einsum("bij,bik->bjk", weights, velocity)
        mean_a = torch.einsum("bij,bik->bjk", weights, acceleration)
        orientation_delta = log_so3(multiply_wxyz(q[:, :, None],
                                                   conjugate_wxyz(mean_q[:, None])))
        # D_e Log(Exp(e) Exp(delta))|0 = J_left(delta)^-1.
        transform = _transport_matrix(_so3_matrices(orientation_delta)[2])
        displaced_cov = transform @ covariance[:, :, None] @ transform.transpose(-1, -2)
        delta = torch.cat((orientation_delta, velocity[:, :, None]-mean_v[:, None],
                           acceleration[:, :, None]-mean_a[:, None]), -1)
        mixed_cov = ((displaced_cov+delta[..., :, None]*delta[..., None, :])
                     *weights[..., None, None]).sum(1)
        mean_a = mean_a*self.active[None, :, 6:]
        return mean_q, mean_v, mean_a, self._clean_covariance(mixed_cov), prior

    def _propagate(self, q, velocity, acceleration, covariance):
        dt = self.dt
        acceleration = acceleration*self.active[None, :, 6:]
        u = dt*velocity+.5*dt*dt*acceleration
        rotation, jacobian, _ = _so3_matrices(u)
        eye = torch.eye(3, dtype=q.dtype, device=q.device)
        ca = self.active[:, 6, None, None]
        transition = torch.zeros((*q.shape[:-1], 9, 9), dtype=q.dtype, device=q.device)
        transition[..., :3, :3] = rotation
        transition[..., :3, 3:6] = dt*jacobian
        transition[..., :3, 6:] = .5*dt*dt*jacobian*ca
        transition[..., 3:6, 3:6] = eye
        transition[..., 3:6, 6:] = dt*eye*ca
        transition[..., 6:, 6:] = eye*ca
        # Integrated white angular acceleration (CV), white angular jerk (CA).
        cv = q.new_tensor([[dt**3/3, dt**2/2, 0.], [dt**2/2, dt, 0.], [0., 0., 0.]])
        ca_noise = q.new_tensor([[dt**5/20, dt**4/8, dt**3/6],
                                 [dt**4/8, dt**3/3, dt**2/2],
                                 [dt**3/6, dt**2/2, dt]])
        noise = torch.stack((cv*self.cv_accel_noise**2, ca_noise*self.ca_jerk_noise**2))
        noise = torch.einsum("mij,kl->mikjl", noise, eye).reshape(2, 9, 9)
        transport = _transport_matrix(jacobian)
        noise = transport @ noise[None] @ transport.transpose(-1, -2)
        predicted_cov = transition @ covariance @ transition.transpose(-1, -2)+noise
        return (normalize_quaternion(multiply_wxyz(exp_so3(u), q)),
                velocity+dt*acceleration, acceleration, self._clean_covariance(predicted_cov))

    def _measure(self, q, velocity, acceleration, covariance, observation, prior):
        innovation = log_so3(multiply_wxyz(observation[:, None], conjugate_wxyz(q)))
        eye = torch.eye(3, dtype=q.dtype, device=q.device)
        measurement_cov = eye*self.measurement_sigma**2
        innovation_cov = covariance[..., :3, :3]+measurement_cov
        # Strict Cholesky: numerical failure is surfaced, never target-dependent fallback.
        chol = torch.linalg.cholesky(innovation_cov)
        gain = torch.cholesky_solve(covariance[..., :, :3].transpose(-1, -2), chol).transpose(-1, -2)
        delta = (gain @ innovation[..., None]).squeeze(-1)
        updated_q = normalize_quaternion(multiply_wxyz(exp_so3(delta[..., :3]), q))
        velocity = velocity+delta[..., 3:6]
        acceleration = (acceleration+delta[..., 6:])*self.active[None, :, 6:]
        identity = torch.eye(9, dtype=q.dtype, device=q.device)
        kh = torch.zeros_like(covariance)
        kh[..., :, :3] = gain
        joseph = ((identity-kh) @ covariance @ (identity-kh).transpose(-1, -2)
                  +gain @ measurement_cov @ gain.transpose(-1, -2))
        reset = _transport_matrix(_so3_matrices(delta[..., :3])[1])
        covariance = self._clean_covariance(reset @ joseph @ reset.transpose(-1, -2))
        whitened = torch.cholesky_solve(innovation[..., None], chol)
        quadratic = (innovation[..., None, :3] @ whitened).squeeze(-1).squeeze(-1)
        logdet = 2*torch.log(chol.diagonal(dim1=-2, dim2=-1)).sum(-1)
        likelihood = -.5*(quadratic+logdet+3*math.log(2*math.pi))
        if self.mode_evidence:
            probabilities = torch.softmax(prior.clamp_min(1e-30).log()+likelihood, -1)
        else:
            probabilities = torch.full_like(prior, .5)
        return updated_q, velocity, acceleration, covariance, probabilities, innovation, likelihood

    @torch.no_grad()
    def filter_state(self, context):
        """Return detached posterior diagnostics, replaying only 8+31 increments."""
        # Covariance algebra must remain fp32 even inside a caller's AMP scope.
        with torch.autocast(device_type=context.device.type, enabled=False):
            return self._filter_state(context)

    def _filter_state(self, context):
        anchor, relative = self._observations(context)
        b = context.shape[0]
        n = self.warmup_increments
        increments = log_so3(multiply_wxyz(relative[:, 1:n+1],
                                            conjugate_wxyz(relative[:, :n])))
        velocity = (increments.mean(1)/self.dt)[:, None].expand(-1, 2, -1).clone()
        acceleration = torch.zeros_like(velocity)
        q = relative[:, n:n+1].expand(-1, 2, -1).clone()
        variances = ([self.initial_orientation_sigma**2]*3
                     +[self.initial_velocity_sigma**2]*3
                     +[self.initial_acceleration_sigma**2]*3)
        covariance = torch.diag_embed(q.new_tensor(variances)*self.active)[None].expand(b, -1, -1, -1).clone()
        probabilities = q.new_full((b, 2), .5)
        for i in range(n+1, self.support):
            q, velocity, acceleration, covariance, prior = self._interact(
                q, velocity, acceleration, covariance, probabilities)
            q, velocity, acceleration, covariance = self._propagate(q, velocity, acceleration, covariance)
            (q, velocity, acceleration, covariance, probabilities, innovation,
             likelihood) = self._measure(q, velocity, acceleration, covariance, relative[:, i], prior)
        return {"q": q, "velocity": velocity, "acceleration": acceleration,
                "covariance": covariance, "probabilities": probabilities,
                "innovation": innovation, "log_likelihood": likelihood, "anchor": anchor}

    @torch.no_grad()
    def forecast_states(self, context):
        """Twenty measurement-free steps, holding final mode probabilities fixed."""
        with torch.autocast(device_type=context.device.type, enabled=False):
            return self._forecast_states(context)

    def _forecast_states(self, context):
        state = self.filter_state(context)
        q, velocity, acceleration, covariance = (state[k] for k in ("q", "velocity", "acceleration", "covariance"))
        if self.reanchor:
            displacement = log_so3(q)
            transform = _transport_matrix(_so3_matrices(displacement)[2])
            covariance = self._clean_covariance(transform @ covariance @ transform.transpose(-1, -2))
            q = torch.zeros_like(q)
            q[..., 0] = 1.
        origin_q, origin_cov = q.clone(), covariance.clone()
        rotations, covariances = [], []
        for _ in range(20):
            q, velocity, acceleration, covariance = self._propagate(q, velocity, acceleration, covariance)
            rotations.append(q)
            covariances.append(covariance)
        return {"posterior": state, "q": torch.stack(rotations, 1),
                "covariance": torch.stack(covariances, 1), "origin_q": origin_q,
                "origin_covariance": origin_cov, "probabilities": state["probabilities"]}

    def _decode(self, rotations, covariance, probabilities, decoder):
        q0, q1 = rotations[:, :, 0], rotations[:, :, 1]
        delta = log_so3(multiply_wxyz(q1, conjugate_wxyz(q0)))
        mean = multiply_wxyz(exp_so3(probabilities[:, None, 1:2]*delta), q0)
        if decoder == "mean":
            return mean
        if decoder == "argmax":
            index = probabilities.argmax(-1)[:, None, None, None].expand(-1, 20, 1, 4)
            return rotations.gather(2, index).squeeze(2)
        if decoder != "risk":
            raise ValueError("decoder must be mean, argmax or risk")
        chol = torch.linalg.cholesky(covariance[..., :3, :3])
        columns = 2*chol.transpose(-1, -2)
        offsets = torch.cat((torch.zeros_like(columns[..., :1, :]), columns, -columns), -2)
        support = multiply_wxyz(exp_so3(offsets), rotations[..., None, :]).flatten(2, 3)
        local_weights = rotations.new_tensor([.25]+[.125]*6)
        weights = (probabilities[:, None, :, None]*local_weights).expand(-1, 20, -1, -1).flatten(2, 3)
        current = mean
        for _ in range(self.mean_iterations):
            residuals = log_so3(multiply_wxyz(support, conjugate_wxyz(current[:, :, None])))
            current = multiply_wxyz(exp_so3((weights[..., None]*residuals).sum(2)), current)
        for _ in range(self.risk_iterations):
            residuals = log_so3(multiply_wxyz(support, conjugate_wxyz(current[:, :, None])))
            inv_distance = weights/torch.linalg.vector_norm(residuals, dim=-1).clamp_min(1e-4)
            update = (inv_distance[..., None]*residuals).sum(2)/inv_distance.sum(2, keepdim=True)
            current = multiply_wxyz(exp_so3(update), current)
        return normalize_quaternion(current)

    @torch.no_grad()
    def pure_filter(self, context, decoder=None):
        """Same-state standalone filter baseline; never invokes neural parameters."""
        forecast = self.forecast_states(context)
        relative = self._decode(forecast["q"], forecast["covariance"],
                                forecast["probabilities"], decoder or self.decoder)
        anchor = forecast["posterior"]["anchor"]
        prediction = normalize_quaternion(multiply_wxyz(anchor[:, None], relative))
        return _align_to_previous(prediction, anchor).float()

    def _raw_features(self, context):
        _, relative = self._observations(context)
        increments = log_so3(multiply_wxyz(relative[:, 1:], conjugate_wxyz(relative[:, :-1])))
        return torch.cat((increments.flatten(1)/self.dt, log_so3(relative).flatten(1)/.1), -1)

    def _state_features(self, state):
        covariance = state["covariance"]/self.feature_scales[None, None, :, None]/self.feature_scales[None, None, None, :]
        covariance = torch.sign(covariance)*torch.log1p(covariance.abs())
        likelihood = state["log_likelihood"]
        likelihood = torch.sign(likelihood)*torch.log1p(likelihood.abs())
        return torch.cat((log_so3(state["q"]).flatten(1)/.1,
                          state["velocity"].flatten(1), state["acceleration"].flatten(1)/10,
                          covariance.flatten(1), state["innovation"].flatten(1)/.1,
                          state["probabilities"], likelihood), -1)

    def forward(self, context):
        forecast = self.forecast_states(context)
        state = forecast["posterior"]
        base = self._decode(forecast["q"], forecast["covariance"],
                            forecast["probabilities"], self.decoder).detach()
        raw = self.raw_encoder(self._raw_features(context))
        estimated = self.state_encoder(self._state_features(state))
        correction = self.residual(self.fusion(torch.cat((raw, estimated), -1))).reshape(-1, 20, 3).float()
        relative = exp_so3(log_so3(base)+correction)
        anchor = _observed_unit(context[:, -1])
        prediction = normalize_quaternion(multiply_wxyz(anchor[:, None], relative))
        return _align_to_previous(prediction, anchor).float()
