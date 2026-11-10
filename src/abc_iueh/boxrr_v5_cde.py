"""Causal, fixed-step second-order SG Neural CDE for the BOXRR research round.

This is a PyTorch BOXRR adaptation, not a numerical reproduction of the
ICCV 2025 JAX model.  A polynomial is fitted ONCE to observed orientations;
its analytical first/second tangent derivatives drive both reconstruction and
forecast integration.  No future measured control, persistent inference state,
dataset access, fitted population scaler, or upstream implementation is used.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .boxrr_models import (
    conjugate_wxyz,
    exp_so3,
    log_so3,
    multiply_wxyz,
    normalize_quaternion,
)


def _observed_quaternions(context: torch.Tensor) -> torch.Tensor:
    """Normalize allowed input, with identity fallback for an invalid zero pose.

    Replace zeros BEFORE normalization, so even the unused branch has finite
    derivatives. Valid unit quaternion inputs are the external contract.
    """
    q = context.float()
    identity = torch.zeros_like(q)
    identity[..., 0] = 1.
    return normalize_quaternion(torch.where(q.abs().amax(-1, keepdim=True) > 0, q, identity))


@dataclass(frozen=True)
class PolynomialControl:
    """Past-fitted power coefficients in u=1+2*t/duration, t in seconds.

    `coefficients` are [B,degree+1,3] rotation vectors in the last observed
    body frame. Value evaluation subtracts p(0), so the actual last pose is
    the forecast anchor even when the SG estimate has a nonzero intercept.
    First/second derivatives are tangent coordinate derivatives, NOT angular
    velocity/acceleration after an SO(3) exponential Jacobian transformation.
    """

    coefficients: torch.Tensor
    last: torch.Tensor
    duration: float

    def evaluate(self, times: torch.Tensor, derivative: int = 0) -> torch.Tensor:
        """Return [B,T,3]; extend the same coefficients at arbitrary future t.

        Only time coordinates and cached past coefficients enter this method.
        The scale factors explicitly convert normalized coordinates to SI time.
        """
        if derivative not in (0, 1, 2):
            raise ValueError('derivative must be 0, 1 or 2')
        t = torch.as_tensor(times, device=self.coefficients.device, dtype=torch.float32).reshape(-1)
        scale = 2. / self.duration
        u = 1. + scale*t
        basis = []
        for power in range(self.coefficients.shape[1]):
            if power < derivative:
                term = torch.zeros_like(u)
            elif derivative == 0:
                term = u.pow(power)-1.  # p(t)-p(0), including exact zero at t=0
            elif derivative == 1:
                term = power*u.pow(power-1)*scale
            else:
                term = power*(power-1)*u.pow(power-2)*scale**2
            basis.append(term)
        return torch.einsum('tp,bpc->btc', torch.stack(basis, -1), self.coefficients)


class CausalSGControl(nn.Module):
    """Bounded degree weighted SG regression using only the observed suffix.

    All observation weights are positive and bounded away from zero. A single
    tiny solve is shared across the batch because these trainable weights are
    global parameters, not population estimates or batch-dependent features.
    Normalized [-1,1] coordinates condition the float32 solve; physical sample
    times remain exactly dt=.01 s. No target smoothing is performed.
    """

    def __init__(self, window: int = 32, polynomial_order: int = 3,
                 dt: float = .01, ridge: float = 1e-6):
        super().__init__()
        if not isinstance(window, int) or not 4 <= window <= 32:
            raise ValueError('window must be an integer in [4,32]')
        if polynomial_order not in (2, 3) or polynomial_order >= window:
            raise ValueError('polynomial_order must be 2 or 3 and less than window')
        if dt != .01:
            raise ValueError('BOXRR control requires dt=.01 seconds (100 Hz)')
        if not 0 <= ridge <= .001:
            raise ValueError('ridge must lie in [0,.001]')
        self.window = window
        self.polynomial_order = polynomial_order
        self.dt = dt
        self.ridge = ridge
        self.duration = (window-1)*dt
        self.weight_logits = nn.Parameter(torch.zeros(window))
        u = torch.linspace(-1., 1., window)
        self.register_buffer('design', torch.stack([u.pow(p) for p in range(polynomial_order+1)], -1))
        self.register_buffer('regularizer', torch.eye(polynomial_order+1))

    def observation_weights(self) -> torch.Tensor:
        # Ratio <=20 even for extreme logits, preventing singular learned masks.
        weights = .05 + .95*torch.sigmoid(self.weight_logits)
        return weights / weights.mean()

    def forward(self, observed: torch.Tensor) -> PolynomialControl:
        if observed.ndim != 3 or observed.shape[-1] != 4 or observed.shape[1] < self.window:
            raise ValueError('observed must be [B,N,4] with N >= window')
        q = _observed_quaternions(observed[:, -self.window:])
        last = q[:, -1]
        relative = multiply_wxyz(conjugate_wxyz(last[:, None]), q)
        tangent = log_so3(relative)
        weight = self.observation_weights()
        atw = self.design.T * weight[None]
        gram = atw @ self.design + self.ridge*self.regularizer
        projection = torch.linalg.solve(gram, atw)
        coefficients = torch.einsum('pn,bnc->bpc', projection, tangent)
        return PolynomialControl(coefficients, last, self.duration)


class SecondOrderVectorField(nn.Module):
    """Fused neural time, velocity and acceleration response maps.

    dz/dt = f_time(z) + f_velocity(z) p'(t)
                         + f_acceleration(z) [acceleration_scale p''(t)].

    The two control maps are separate blocks of the learned projection and
    share a trunk to reduce small GPU kernel launches. A nonvanishing time
    channel (dt/dt=1) allows autonomous dynamics when the pose is stationary.
    Bounded response maps stabilize explicit integration; the readout is signed
    and has no amplitude cap. This is second-order CONTROL, not a claim that
    the latent itself satisfies a mechanical second-order state equation.
    """

    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.trunk = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width), nn.Tanh())
        self.control_map = nn.Linear(width, 7*width)

    def forward(self, z: torch.Tensor, control_derivative: torch.Tensor) -> torch.Tensor:
        response = torch.tanh(self.control_map(self.trunk(z))).reshape(-1, self.width, 7)
        return (response*control_derivative[:, None]).sum(-1)


class SGNeuralCDE(nn.Module):
    """Stateless [B,400,4] -> [B,20,4] unit-wxyz forecaster.

    Default integration has exactly 52 knots: 32 observed and 20 future.
    All 20 forecasts share one latent integration, rather than solving each
    horizon separately. `substeps` refines every fixed interval for numerical
    checks. Midpoint RK2 is the default, and classical RK4 is also supported.
    There is no adaptive solver, data-dependent loop or history-length-400 loop.
    """

    arm = 'sg_neural_cde'

    def __init__(self, width: int = 64, window: int = 32, substeps: int = 1,
                 polynomial_order: int = 3, solver: str = 'rk2',
                 dt: float = .01, ridge: float = 1e-6,
                 acceleration_scale: float = .1):
        super().__init__()
        if not isinstance(width, int) or width < 4:
            raise ValueError('width must be an integer >=4')
        if not isinstance(substeps, int) or not 1 <= substeps <= 8:
            raise ValueError('substeps must be an integer in [1,8]')
        if solver not in ('rk2', 'rk4'):
            raise ValueError('solver must be rk2 or rk4')
        if not 0 < acceleration_scale <= 1:
            raise ValueError('acceleration_scale must lie in (0,1] seconds')
        self.width = width
        self.window = window
        self.substeps = substeps
        self.polynomial_order = polynomial_order
        self.solver = solver
        self.dt = dt
        self.ridge = ridge
        self.acceleration_scale = acceleration_scale
        self.sg = CausalSGControl(window, polynomial_order, dt, ridge)
        self.initial = nn.Sequential(nn.Linear(9, width), nn.SiLU(), nn.Linear(width, width), nn.Tanh())
        self.field = SecondOrderVectorField(width)
        self.readout = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3, bias=False))
        # Keep a useful polynomial starting point without blocking gradient flow
        # into the learned dynamics at the first optimizer update.
        nn.init.normal_(self.readout[-1].weight, std=.02)
        self.register_buffer('future_times', torch.arange(1, 21, dtype=torch.float32)*dt)

    def get_config(self) -> dict:
        """Return exactly the constructor arguments needed for a reload."""
        return {key: getattr(self, key) for key in (
            'width', 'window', 'substeps', 'polynomial_order', 'solver', 'dt',
            'ridge', 'acceleration_scale')}

    def fit_control(self, context: torch.Tensor) -> PolynomialControl:
        if context.ndim != 3 or context.shape[1:] != (400, 4):
            raise ValueError('context must have shape [B,400,4]')
        return self.sg(context)

    def integrate_control(self, path: PolynomialControl) -> tuple[torch.Tensor, torch.Tensor]:
        """Return origin latent [B,W] and future latents [B,20,W].

        Analytical controls for every RK stage are evaluated in one vectorized
        operation. The only integration loop is the bounded fixed time grid.
        Exposed for numerical verification and the declared polynomial control.
        """
        intervals = self.window-1+20
        steps = intervals*self.substeps
        step = self.dt/self.substeps
        # Half-grid includes endpoint and midpoint controls for either method.
        stage_times = (torch.arange(2*steps+1, device=path.coefficients.device,
                                    dtype=torch.float32)*(.5*step)-self.sg.duration)
        value = path.evaluate(stage_times)
        velocity = path.evaluate(stage_times, 1)
        acceleration = path.evaluate(stage_times, 2)
        time_channel = torch.ones_like(velocity[..., :1])
        controls = torch.cat((time_channel, velocity,
                              self.acceleration_scale*acceleration), -1)
        z = self.initial(torch.cat((value[:, 0], .1*velocity[:, 0],
                                    .01*acceleration[:, 0]), -1))
        origin_step = (self.window-1)*self.substeps
        origin = None
        future = []
        for index in range(steps):
            k1 = self.field(z, controls[:, 2*index])
            if self.solver == 'rk2':
                z = z+step*self.field(z+.5*step*k1, controls[:, 2*index+1])
            else:
                k2 = self.field(z+.5*step*k1, controls[:, 2*index+1])
                k3 = self.field(z+.5*step*k2, controls[:, 2*index+1])
                k4 = self.field(z+step*k3, controls[:, 2*index+2])
                z = z+(step/6.)*(k1+2*k2+2*k3+k4)
            completed = index+1
            if completed == origin_step:
                origin = z
            elif completed > origin_step and (completed-origin_step) % self.substeps == 0:
                future.append(z)
        return origin, torch.stack(future, 1)

    def polynomial_forecast(self, context: torch.Tensor) -> torch.Tensor:
        """Declared same-SG-without-learned-dynamics control (no fit itself)."""
        with torch.autocast(device_type=context.device.type, enabled=False):
            path = self.fit_control(context)
            return normalize_quaternion(multiply_wxyz(path.last[:, None], exp_so3(path.evaluate(self.future_times)))).float()

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=context.device.type, enabled=False):
            path = self.fit_control(context)
            origin, future = self.integrate_control(path)
            # Subtracting the actual origin readout imposes an exact t=0 anchor.
            # Both terms remain in the graph; no detached hidden shortcuts.
            correction = self.readout(future)-self.readout(origin)[:, None]
            tangent = path.evaluate(self.future_times)+correction
            prediction = multiply_wxyz(path.last[:, None], exp_so3(tangent))
            return normalize_quaternion(prediction).float()
