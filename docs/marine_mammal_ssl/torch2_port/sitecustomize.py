# Auto-imported at interpreter startup when /home/yarix/a2v is on PYTHONPATH.
# Installs the torch-2.x shims animal2vec/fairseq-0.12.2 need for TRAINING, before any user import.
import os, sys, types

import torch
# (a) torch._six shim — defensive no-op (this fairseq build doesn't reference it, but transitive imports might)
if not hasattr(torch, "_six"):
    _six = types.ModuleType("torch._six")
    _six.string_classes = (str, bytes); _six.int_classes = (int,); _six.inf = float("inf")
    import collections.abc as _abc; _six.container_abcs = _abc
    torch._six = _six; sys.modules["torch._six"] = _six

# (b) compute_mask_indices: swallow newer kwargs (e.g. add_masks) that a2v passes but installed
#     fairseq doesn't accept. Patch the SOURCE before nn.modalities.base does `from ... import`.
try:
    import inspect, fairseq.data.data_utils as _du
    _orig_cmi = _du.compute_mask_indices
    _sig = set(inspect.signature(_orig_cmi).parameters)
    def _cmi(*a, **k):
        return _orig_cmi(*a, **{kk: vv for kk, vv in k.items() if kk in _sig})
    _du.compute_mask_indices = _cmi
    print("[sitecustomize] compute_mask_indices kwarg-swallow installed", flush=True)
except Exception as e:
    print("[sitecustomize] cmi patch skipped:", e, flush=True)

# (c) EMAModuleConfig: a2v passes fork-only kwargs (log_norms) the installed fairseq dataclass lacks.
#     Swallow unknown kwargs into plain attributes so config.log_norms still resolves. EMA is needed
#     for TRAINING (data2vec targets), so we can't skip it like inference did.
try:
    import inspect
    from fairseq.modules import ema_module as _em
    _Cfg = _em.EMAModuleConfig
    _orig_emacfg_init = _Cfg.__init__
    _emacfg_params = set(inspect.signature(_orig_emacfg_init).parameters)
    def _emacfg_init(self, *a, **k):
        known = {kk: vv for kk, vv in k.items() if kk in _emacfg_params}
        extra = {kk: vv for kk, vv in k.items() if kk not in _emacfg_params}
        _orig_emacfg_init(self, *a, **known)
        for kk, vv in extra.items():
            try: object.__setattr__(self, kk, vv)
            except Exception: setattr(self, kk, vv)
    _Cfg.__init__ = _emacfg_init
    print("[sitecustomize] EMAModuleConfig kwarg-swallow installed", flush=True)
except Exception as e:
    print("[sitecustomize] EMA patch skipped:", e, flush=True)

# (d) EMAModule: a2v passes fork-only `copy_model`. copy_model=False is LOAD-BEARING — a2v already built
#     the target via make_target_model(); vanilla's unconditional deepcopy recurses on the task circular
#     ref. Reimplement __init__ to honor copy_model (skip the deepcopy when False).
try:
    import copy as _copy
    from fairseq.modules.ema_module import EMAModule as _EMA
    def _ema_init2(self, model, config, device=None, skip_keys=None, copy_model=True, **_extra):
        self.decay = config.ema_decay
        self.model = _copy.deepcopy(model) if copy_model else model
        self.model.requires_grad_(False)
        self.config = config
        self.skip_keys = skip_keys or set()
        self.fp32_params = {}
        self.logs = {}   # fork EMAModule exposes .logs (norm stats); data2vec2.py:966 iterates it unconditionally
        if device is not None:
            self.model = self.model.to(device=device)
        if getattr(config, "ema_fp32", False):
            self.build_fp32_params()
        self.update_freq_counter = 0
    _EMA.__init__ = _ema_init2
    # set_decay: a2v passes fork-only `weight_decay` (data2vec2.py:406). Vanilla set_decay(decay) only.
    _orig_set_decay = _EMA.set_decay
    def _set_decay(self, decay, weight_decay=None, *_a, **_k):
        return _orig_set_decay(self, decay)
    _EMA.set_decay = _set_decay
    print("[sitecustomize] EMAModule.__init__ copy_model-aware reimpl + set_decay shim installed", flush=True)
except Exception as e:
    print("[sitecustomize] EMAModule patch skipped:", e, flush=True)

# (e) ModelCriterion: a2v's ExpandedModelCriterion passes a fork-only `can_sum` arg to the base
#     ModelCriterion.__init__ (vanilla takes task, loss_weights, log_keys). Accept + store it.
try:
    from fairseq.criterions.model_criterion import ModelCriterion as _MC
    _orig_mc_init = _MC.__init__
    def _mc_init(self, task, loss_weights=None, log_keys=None, can_sum=True, **_extra):
        _orig_mc_init(self, task, loss_weights, log_keys)
        self.can_sum = can_sum
    _MC.__init__ = _mc_init
    print("[sitecustomize] ModelCriterion can_sum-aware init installed", flush=True)
except Exception as e:
    print("[sitecustomize] ModelCriterion patch skipped:", e, flush=True)

# (f) composite optimizer `dynamic_groups` (fork feature): the model tags decoder params with
#     param_group="decoder", but the config defines only groups.default. dynamic_groups clones the
#     default group config for each discovered param_group so vanilla composite's key-match passes.
try:
    import dataclasses, copy as _cp2
    from fairseq.optim import composite as _comp
    from omegaconf import open_dict as _od
    _COC = _comp.CompositeOptimizerConfig
    if "dynamic_groups" not in getattr(_COC, "__dataclass_fields__", {}):
        _COC.__annotations__["dynamic_groups"] = bool
        _f = dataclasses.field(default=False); _f.name = "dynamic_groups"; _f.type = bool
        _f._field_type = dataclasses._FIELD
        _COC.__dataclass_fields__["dynamic_groups"] = _f
        setattr(_COC, "dynamic_groups", False)
    _FCO = _comp.FairseqCompositeOptimizer
    _orig_fco_init = _FCO.__init__
    def _fco_init(self, cfg, params):
        if getattr(cfg, "dynamic_groups", False) and "default" in cfg.groups:
            discovered = set(getattr(p, "param_group", "default") for p in params)
            with _od(cfg.groups):
                for g in discovered:
                    if g not in cfg.groups:
                        cfg.groups[g] = _cp2.deepcopy(cfg.groups["default"])
        _orig_fco_init(self, cfg, params)
    _FCO.__init__ = _fco_init
    print("[sitecustomize] composite dynamic_groups shim installed", flush=True)
except Exception as e:
    print("[sitecustomize] composite patch skipped:", e, flush=True)

print(f"[sitecustomize] torch {torch.__version__} shims installed", flush=True)
