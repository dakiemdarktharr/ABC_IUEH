"""Round-2 conditional density and predictive-embedding models.

Only official fitting windows may enter training_loss. Inference takes context
alone. The EBM is a tangent-space density pushforward, not a claimed Haar PDF;
its score is trained with proposal-corrected ranking NCE. JEPA has a real
pretraining phase and four future-only EMA patch targets.
"""
from copy import deepcopy
import math
import torch
from torch import nn
import torch.nn.functional as F

from .boxrr_v2_deterministic import TemporalV2
from .boxrr_models import (normalize_quaternion, conjugate_wxyz, multiply_wxyz,
                          log_so3, exp_so3, geodesic_radians)

MODEL_NAMES = ('energy_v2', 'jepa_v2')


def supervised_loss(prediction, target):
    angles = geodesic_radians(prediction, target)
    endpoint = angles[:, -1].mean()
    return endpoint + .1*angles.mean(), endpoint


def mixture_log_prob(value, logits, means, scales):
    """Exact diagonal-Gaussian mixture log PDF on R3.

    value [B,N,3], logits [B,K], means/scales [B,K,3].
    Scales are strictly positive, and samples are never clipped/truncated.
    """
    standardized = (value[:, :, None]-means[:, None])/scales[:, None]
    component = (-.5*standardized.square()-scales[:, None].log()
                 -.5*math.log(2*math.pi)).sum(-1)
    return torch.logsumexp(component + F.log_softmax(logits, -1)[:, None], -1)


def mixture_sample(logits, means, scales, count):
    """Independent ancestral samples; no gradient through proposal sampling."""
    with torch.no_grad():
        indices = torch.multinomial(logits.softmax(-1), count, replacement=True)
        indices = indices[:, :, None].expand(-1, -1, 3)
        selected_mean = means.gather(1, indices)
        selected_scale = scales.gather(1, indices)
        return selected_mean + selected_scale*torch.randn_like(selected_mean)


def weighted_rotation_median(rotations, weights, initial, iterations=4):
    """Fixed-iteration SO3 angular-risk action, with finite-risk safeguard.

    Returns the best of initial, weighted medoid and all Weiszfeld iterates.
    This is a local numerical approximation, not a global optimality claim.
    """
    weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-12)
    distances = geodesic_radians(rotations[:, :, None], rotations[:, None, :])
    risks = (distances*weights[:, None]).sum(-1)
    index = risks.argmin(-1)
    medoid = rotations.gather(1, index[:, None, None].expand(-1, 1, 4))[:, 0]
    medoid_risk = risks.gather(1, index[:, None])[:, 0]
    initial = normalize_quaternion(initial)
    initial_risk = (geodesic_radians(initial[:, None], rotations)*weights).sum(-1)
    best = torch.where((medoid_risk < initial_risk)[:, None], medoid, initial)
    best_risk = torch.minimum(medoid_risk, initial_risk)
    # Starting exactly on a medoid makes inverse-distance iteration nearly
    # stationary. Start the continuous path at a weighted tangent mean while
    # retaining both discrete and proposal actions in the risk safeguard.
    centered = log_so3(multiply_wxyz(conjugate_wxyz(initial[:, None]), rotations))
    current = normalize_quaternion(multiply_wxyz(initial, exp_so3((weights[:, :, None]*centered).sum(1))))
    risk = (geodesic_radians(current[:, None], rotations)*weights).sum(-1)
    better = risk < best_risk
    best = torch.where(better[:, None], current, best)
    best_risk = torch.minimum(best_risk, risk)
    for _ in range(iterations):
        displacement = log_so3(multiply_wxyz(conjugate_wxyz(current[:, None]), rotations))
        distance = displacement.norm(dim=-1).clamp_min(1e-5)
        inverse_distance = weights/distance
        update = (inverse_distance[:, :, None]*displacement).sum(1)/inverse_distance.sum(-1, keepdim=True)
        # Local step safeguard bounds large rotations but never truncates a
        # proposal sample or changes its stated sampling density.
        length = update.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        update = update*torch.clamp(.5/length, max=1.)
        current = normalize_quaternion(multiply_wxyz(current, exp_so3(update)))
        risk = (geodesic_radians(current[:, None], rotations)*weights).sum(-1)
        better = risk < best_risk
        best = torch.where(better[:, None], current, best)
        best_risk = torch.minimum(best_risk, risk)
    return best, best_risk


class ConditionalEnergyV2(TemporalV2):
    """Supervised temporal proposal with an independently trained EBM density.

    q is an exact Gaussian mixture. p(v|x) proportional to q(v|x)*exp(-r(x,v)),
    where the bounded residual r preserves a finite normalizer. Both density
    branches see detached context features and deterministic proposal endpoints;
    they cannot degrade the point forecaster through auxiliary gradients.
    """
    arm = 'energy_v2'

    def __init__(self, width=32, latent_dim=128, use_gravity=True, kernel=5,
                 components=3, negatives=48, quadrature_per_component=16,
                 median_iterations=4, energy_warmup=2000, seed=20261106):
        super().__init__(width, latent_dim, use_gravity, kernel)
        if components < 1 or negatives < 1 or quadrature_per_component < 4:
            raise ValueError('Positive mixture/sample counts required; quadrature >=4')
        self.components = int(components)
        self.negatives = int(negatives)
        self.median_iterations = int(median_iterations)
        self.energy_warmup = int(energy_warmup)
        self.density_head = nn.Sequential(nn.Linear(latent_dim, 64), nn.SiLU(),
                                          nn.Linear(64, components*7))
        final = self.density_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        initial_scales = torch.logspace(math.log10(.02), math.log10(.2), components)
        # scale=.003+softplus(raw); initialise distinct components to avoid
        # identical-mixture symmetry. Components have unbounded Gaussian support.
        with torch.no_grad():
            bias = final.bias.view(components, 7)
            bias[:, 3:6] = torch.log(torch.expm1(initial_scales-.003))[:, None]
        self.energy_head = nn.Sequential(nn.Linear(latent_dim+7, 64), nn.SiLU(),
                                        nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.energy_head[-1].weight)
        nn.init.zeros_(self.energy_head[-1].bias)
        # Fixed, antithetic low-discrepancy normal quadrature, equal number per
        # component. Component probabilities enter integration weights explicitly.
        half = (quadrature_per_component+1)//2
        uniform = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed).draw(half)
        normal = math.sqrt(2.)*torch.erfinv(2*uniform.clamp(1e-6, 1-1e-6)-1)
        normal = torch.cat((normal, -normal), 0)[:quadrature_per_component]
        self.register_buffer('normal_quadrature', normal)
        self.register_buffer('fraction', torch.arange(1, 21).view(1, 20, 1)/20.)

    def distribution(self, latent, endpoint, detach_features=False):
        if detach_features:
            latent, endpoint = latent.detach(), endpoint.detach()
        raw = self.density_head(latent).view(-1, self.components, 7)
        means = endpoint[:, None]+.3*torch.tanh(raw[:, :, :3])
        scales = .003 + F.softplus(raw[:, :, 3:6].clamp(-12, 2))
        return raw[:, :, 6], means, scales

    def residual_energy(self, latent, value, endpoint):
        difference = (value-endpoint[:, None])/.15
        inputs = torch.cat((latent[:, None].expand(-1, value.shape[1], -1),
                            value/.3, difference,
                            difference.square().sum(-1, keepdim=True).clamp(max=100)), -1)
        return 6.*torch.tanh(self.energy_head(inputs).squeeze(-1))

    def energy(self, latent, value, endpoint, distribution):
        return -mixture_log_prob(value, *distribution)+self.residual_energy(latent, value, endpoint)

    def quadrature(self, distribution):
        logits, means, scales = distribution
        points = means[:, :, None]+scales[:, :, None]*self.normal_quadrature[None, None]
        count = self.normal_quadrature.shape[0]
        log_mass = (F.log_softmax(logits, -1)[:, :, None]-math.log(count)).expand(-1, -1, count)
        return points.flatten(1, 2), log_mass.flatten(1, 2)

    def decode_density(self, latent, bundle, use_energy=True):
        tangent = self.decoder.tangent(latent, bundle)
        endpoint = tangent[:, -1]
        distribution = self.distribution(latent, endpoint)
        points, log_mass = self.quadrature(distribution)
        residual = self.residual_energy(latent, points, endpoint) if use_energy else torch.zeros_like(log_mass)
        # Stratified integration under q: pi_k/M per component, then multiply
        # by p/q=exp(-residual). This is not softmax over an arbitrary grid.
        weights = (log_mass-residual).softmax(-1)
        rotations = exp_so3(points)
        initial = exp_so3(endpoint)
        chosen, risk = weighted_rotation_median(rotations, weights, initial, self.median_iterations)
        base_relative = exp_so3(tangent)
        correction = log_so3(multiply_wxyz(conjugate_wxyz(base_relative[:, -1]), chosen))
        corrected = multiply_wxyz(base_relative, exp_so3(self.fraction*correction[:, None]))
        prediction = normalize_quaternion(multiply_wxyz(bundle.last[:, None], corrected))
        diagnostics = {
            'energy_ess': 1./weights.square().sum(-1),
            'energy_max_weight': weights.amax(-1),
            'energy_estimated_risk_radians': risk,
            'energy_endpoint_correction_radians': correction.norm(dim=-1),
        }
        return prediction, diagnostics

    def proposal_prediction(self, context):
        return self.decode(*self.encode(context))

    def forward_with_diagnostics(self, context):
        return self.decode_density(*self.encode(context))

    def forward(self, context):
        return self.forward_with_diagnostics(context)[0]


class JointEmbeddingV2(TemporalV2):
    """Four future-only patches, EMA temporal target and explicit pretraining.

    The online context encoder always receives only the observed 400 samples.
    Future targets occur solely in training_loss. The decoder is unused during
    pretraining; a separate supervised phase and optimizer must follow it.
    """
    arm = 'jepa_v2'

    def __init__(self, width=32, latent_dim=128, use_gravity=True, kernel=5,
                 embedding_dim=64, ema_decay=.998):
        super().__init__(width, latent_dim, use_gravity, kernel)
        self.embedding_dim = int(embedding_dim)
        self.ema_decay = float(ema_decay)
        self.patch_projection = nn.Sequential(nn.Linear(width, embedding_dim),
                                               nn.SiLU(), nn.Linear(embedding_dim, embedding_dim))
        self.horizon_embedding = nn.Parameter(torch.randn(4, 16)*.02)
        self.predictor = nn.Sequential(nn.Linear(latent_dim+16, latent_dim), nn.SiLU(),
                                       nn.Linear(latent_dim, embedding_dim))
        self.descriptor_head = nn.Linear(embedding_dim, 6)
        # Copy only the actually shared causal stack; unused summary/long-path
        # weights would make the teacher count and update misleading.
        self.target_stem = deepcopy(self.encoder.stem)
        self.target_blocks = deepcopy(self.encoder.blocks)
        self.target_projection = deepcopy(self.patch_projection)
        for teacher in self.teacher_modules():
            teacher.requires_grad_(False)
            teacher.eval()

    def teacher_modules(self):
        return (self.target_stem, self.target_blocks, self.target_projection)

    def train(self, mode=True):
        super().train(mode)
        for teacher in self.teacher_modules():
            teacher.eval()
        return self

    def predict_embeddings(self, latent):
        z = latent[:, None].expand(-1, 4, -1)
        positions = self.horizon_embedding[None].expand(latent.shape[0], -1, -1)
        return self.predictor(torch.cat((z, positions), -1))

    def future_tokens(self, last, future, previous=None):
        # Two observed boundary poses give the first future change its actual
        # previous increment. Only the final20 target tokens reach the teacher.
        if previous is not None:
            sequence = torch.cat((previous[:, None], last[:, None], future), 1)
            return self.features.build_tokens(sequence, reference=last)[:, :, -20:]
        sequence = torch.cat((last[:, None], future), 1)
        tokens = self.features.build_tokens(sequence, reference=last)[:, :, 1:]
        tokens = tokens.clone()
        tokens[:, 3:6, 0] = 0.  # Unknown boundary change; never fabricate u-0.
        return tokens

    def target_embeddings(self, tokens, teacher=True):
        if teacher:
            h = self.target_blocks(F.silu(self.target_stem(tokens)))
            projection = self.target_projection
        else:
            h = self.encoder.encode_tokens(tokens)
            projection = self.patch_projection
        patches = h.reshape(h.shape[0], h.shape[1], 4, 5).mean(-1).transpose(1, 2)
        return projection(patches)

    def descriptors(self, last, future):
        relative = multiply_wxyz(conjugate_wxyz(last[:, None]), normalize_quaternion(future))
        displacement = log_so3(relative[:, [4, 9, 14, 19]])/.2
        full = torch.cat((last[:, None], normalize_quaternion(future)), 1)
        reference_relative = multiply_wxyz(conjugate_wxyz(last[:, None]), full)
        increments = log_so3(multiply_wxyz(reference_relative[:, 1:],
                                           conjugate_wxyz(reference_relative[:, :-1])))
        velocity = increments.reshape(-1, 4, 5, 3).mean(2)/.01
        return torch.cat((displacement, velocity), -1)


def variance_covariance(embeddings):
    """VICReg moments across examples, separately for each future patch.

    Pooling patches into the batch could hide collapse behind fixed positions.
    """
    centered = embeddings-embeddings.mean(0, keepdim=True)
    std = (centered.square().mean(0)+1e-4).sqrt()
    variance = F.relu(1.-std).mean()
    denominator = max(embeddings.shape[0]-1, 1)
    covariance = torch.einsum('bpd,bpe->pde', centered, centered)/denominator
    diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1)
    off_diagonal = covariance-torch.diag_embed(diagonal)
    penalty = off_diagonal.square().sum((-1, -2)).mean()/embeddings.shape[-1]
    return variance, penalty, std.mean(), covariance


def _energy_loss(model, x, y, step):
    latent, bundle = model.encode(x)
    prediction = model.decode(latent, bundle)
    supervised, endpoint = supervised_loss(prediction, y)
    point = model.decoder.tangent(latent, bundle)[:, -1].detach()
    features = latent.detach()
    distribution = model.distribution(features, point)
    target = log_so3(multiply_wxyz(conjugate_wxyz(bundle.last), y[:, -1])).detach()
    nll = -mixture_log_prob(target[:, None], *distribution).mean()
    detached = tuple(value.detach() for value in distribution)
    nce = torch.zeros_like(supervised)
    residual_size = torch.zeros_like(supervised)
    if step >= model.energy_warmup:
        negatives = mixture_sample(*detached, model.negatives)
        candidates = torch.cat((target[:, None], negatives), 1)
        log_q = mixture_log_prob(candidates, *detached)
        energy = model.energy(features, candidates, point, detached)
        logits = -energy-log_q
        nce = F.cross_entropy(logits, torch.zeros(x.shape[0], device=x.device, dtype=torch.long))
        residual_size = model.residual_energy(features, candidates, point).square().mean()
    loss = supervised+.02*nll+.03*nce+1e-5*residual_size
    return loss, {'endpoint_radians': endpoint.detach(), 'proposal_endpoint_radians': endpoint.detach(),
                  'density_nll': nll.detach(), 'energy_nce': nce.detach(),
                  'energy_residual_square': residual_size.detach(),
                  'energy_active': float(step >= model.energy_warmup)}


def _jepa_loss(model, x, y, phase, step):
    latent, bundle = model.encode(x)
    tokens = model.future_tokens(bundle.last, y, previous=x[:, -2])
    with torch.no_grad():
        target = model.target_embeddings(tokens, teacher=True)
    online = model.target_embeddings(tokens, teacher=False)
    predicted = model.predict_embeddings(latent)
    normal_pred = F.layer_norm(predicted, (model.embedding_dim,))
    normal_target = F.layer_norm(target, (model.embedding_dim,))
    latent_loss = F.smooth_l1_loss(normal_pred, normal_target)
    var, cov, online_std, covariance = variance_covariance(online)
    prediction_var, _, prediction_std, _ = variance_covariance(predicted)
    descriptors = model.descriptors(bundle.last, y).detach()
    grounding = F.smooth_l1_loss(model.descriptor_head(predicted), descriptors)
    online_grounding = F.smooth_l1_loss(model.descriptor_head(online), descriptors)
    representation = latent_loss+.25*grounding+.1*online_grounding+.1*var+.02*prediction_var+.01*cov
    diagnostics = {
        'jepa_latent': latent_loss.detach(), 'jepa_descriptor': grounding.detach(),
        'jepa_online_descriptor': online_grounding.detach(),
        'jepa_variance': var.detach(), 'jepa_covariance': cov.detach(),
        'jepa_online_std': online_std.detach(), 'jepa_predicted_std': prediction_std.detach(),
        'jepa_teacher_std': target.std(0, unbiased=False).mean().detach(),
    }
    if (step+1) % 200 == 0:
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvalsh(covariance.mean(0)).clamp_min(0)
            trace = eigenvalues.sum()
            probabilities = eigenvalues/trace.clamp_min(torch.finfo(eigenvalues.dtype).tiny)
            entropy = -(probabilities*probabilities.clamp_min(torch.finfo(eigenvalues.dtype).tiny).log()).sum()
            diagnostics['jepa_effective_rank'] = torch.where(trace>0,torch.exp(entropy),trace.new_zeros(()))
    if phase == 'pretrain':
        return representation, diagnostics
    prediction = model.decode(latent, bundle)
    supervised, endpoint = supervised_loss(prediction, y)
    diagnostics['endpoint_radians'] = endpoint.detach()
    return supervised+.02*representation, diagnostics


def training_loss(model, x, y, step=0, total_steps=20000, phase='finetune'):
    if phase not in ('pretrain', 'finetune'):
        raise ValueError("phase must be 'pretrain' or 'finetune'")
    if x.ndim != 3 or x.shape[1:] != (400, 4) or y.shape != (x.shape[0], 20, 4):
        raise ValueError('Training requires [B,400,4] context and [B,20,4] targets')
    if isinstance(model, ConditionalEnergyV2):
        if phase != 'finetune':
            raise ValueError('Energy arm has no JEPA pretraining phase')
        return _energy_loss(model, x, y, int(step))
    if isinstance(model, JointEmbeddingV2):
        return _jepa_loss(model, x, y, phase, int(step))
    raise TypeError('Expected an energy_v2 or jepa_v2 model')


@torch.no_grad()
def after_optimizer_step(model):
    if isinstance(model, JointEmbeddingV2):
        pairs = ((model.target_stem, model.encoder.stem),
                 (model.target_blocks, model.encoder.blocks),
                 (model.target_projection, model.patch_projection))
        for teacher, online in pairs:
            for target, source in zip(teacher.parameters(), online.parameters()):
                target.lerp_(source, 1.-model.ema_decay)
            for target, source in zip(teacher.buffers(), online.buffers()):
                target.copy_(source)


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if isinstance(model, JointEmbeddingV2):
        unused = (model.patch_projection, model.predictor, model.descriptor_head,
                  *model.teacher_modules())
        excluded = sum(p.numel() for module in unused for p in module.parameters())
        excluded += model.horizon_embedding.numel()
        inference = total-excluded
    else:
        inference = total
    return {'total': total, 'trainable': trainable, 'inference': inference}


def build_latent_model(name, seed=20261106, **config):
    constructors = {'energy_v2': ConditionalEnergyV2, 'jepa_v2': JointEmbeddingV2}
    if name not in constructors:
        raise ValueError(f'Unknown latent arm {name!r}')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if name == 'energy_v2':
            return constructors[name](seed=seed, **config)
        return constructors[name](**config)

