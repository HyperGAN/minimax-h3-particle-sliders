"""Fresh-only scoped initialization, before any optimizer; original restore lineage."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from particlegan.init import Uniform, KEEP
from particle_sliders.initialization import ComponentInitialization, InitializationReceipt, initialize_fresh_components
from .contracts import CORE_REVISION, SOURCE_PROFILE, INITIALIZATION_POLICY, digest, file_hash


@dataclass(frozen=True)
class Construction:
    mode: str
    cohort: str
    options_sha256: str
    initialization_json: str | None = None


def source_identity():
    return {"core_revision": CORE_REVISION, "source_profile": SOURCE_PROFILE,
            "policy": INITIALIZATION_POLICY, "product_initialization_source_sha256": file_hash(__file__)}


def parameter_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{INITIALIZATION_POLICY}:{seed}:{name}".encode()).digest()[:8], "big") % 2**63


def declared_hndl_weight(leaf):
    """Expose one exact ordinary HNDL weight to the existing shared helper.

    HNDL0.10 Linear is an EqualLinear subclass, not an exact nn.Linear leaf.
    With equalized=False its real forward is the original nn.Linear forward.
    A plain holder carries that SAME Parameter with complete explicit PG
    declarations; no layer class, forward, optimizer or shared registry changes.
    """
    from hndl.operators.linear import Linear as HNDLLinear
    if (type(leaf) is not HNDLLinear or getattr(leaf, "equalized", None) is not False
            or leaf.bias is not None or any(leaf.children())
            or nn.utils.parametrize.is_parametrized(leaf)
            or list(leaf.buffers(recurse=False))
            or set(leaf._parameters) != {"weight", "bias"}
            or type(leaf.weight) is not nn.Parameter or not leaf.weight.requires_grad
            or type(leaf.in_features) is not int or type(leaf.out_features) is not int
            or leaf.in_features < 1 or leaf.out_features < 1
            or tuple(leaf.weight.shape) != (leaf.out_features, leaf.in_features)):
        raise ValueError("exact ordinary bias-free non-equalized HNDL Linear weight required")
    holder = nn.Module()
    holder.register_parameter("weight", leaf.weight)
    return holder


def policies(adapter, critic, seed):
    """QR only declared fresh down matrices at their explicit host RMS.

    Original H3 down constructor is Kaiming(a=sqrt5), not Flux a=1.
    Original zero UP and HNDL critic constructor values are kept exactly.
    No bank exists and no R2 table is invented.
    """
    scopes={}
    for index,branch in enumerate(adapter.loras):
        for node in ("down","up"):
            leaf=branch.graph[node];holder=declared_hndl_weight(leaf)
            key=f"adapter.branch{index}.{node}"
            law=Uniform(-1/math.sqrt(leaf.in_features),1/math.sqrt(leaf.in_features)) if node=="down" else KEEP
            scopes[key]=ComponentInitialization(holder,"deterministic_orthogonal",
                {"weight":parameter_seed(seed,key+".weight")},distributions={"weight":law})
    # KEEP is explicit: preserve the pinned HNDL critic's real host law, including
    # all bias Parameters; do not silently replace it with a guessed QR scale.
    scopes["critic.host_law"]=ComponentInitialization(critic,"keep",
        {})
    actual={id(p) for root in(adapter,critic) for p in root.parameters() if p.requires_grad}
    owned={id(p) for scope in scopes.values() for p in scope.module.parameters() if p.requires_grad}
    if actual!=owned:raise ValueError("fresh scopes must cover exactly owned H3 trainables")
    return scopes


def validate_receipt(value, cohort):
    import particle_sliders.initialization as shared
    if (not isinstance(value, dict) or value.get("schema") != "particle-sliders-scoped-fresh-initialization-v1"
            or not isinstance(cohort, str) or not cohort or value.get("cohort_id") != cohort
            or type(value.get("completed_steps")) is not int or value["completed_steps"] != 0
            or value.get("restored_from") is not None or value.get("source_identity") != source_identity()
            or value.get("helper_source") != {"file": Path(shared.__file__).name,
                "bytes": Path(shared.__file__).stat().st_size, "sha256": file_hash(shared.__file__)}
            or value.get("particlegan", {}).get("version") != "0.9.5"
            or value.get("particlegan", {}).get("initializer_source", {}).get("sha256") !=
               "3534ff23efa413cdf8b4125b91e0d30671d873b3e2958547d4e744b750454b9b"):
        raise ValueError("receipt is not this new H3 fresh-initialization lineage")
    return InitializationReceipt(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def initialize_owned(adapter, critic, construction, options):
    if (not isinstance(construction, Construction) or construction.mode not in ("fresh", "resume")
            or construction.options_sha256 != digest(options)):
        raise ValueError("explicit fresh/resume construction required before optimizers")
    scopes = policies(adapter, critic, options.get("seed", 7))
    if construction.mode == "fresh":
        if construction.initialization_json is not None:
            raise ValueError("fresh cannot import a restore receipt")
        return initialize_fresh_components(scopes, cohort_id=construction.cohort, source_identity=source_identity(),
                                           completed_steps=0, restored_from=None, optimizers=())
    if construction.initialization_json is None:
        raise ValueError("resume must retain its original fresh receipt")
    value = json.loads(construction.initialization_json)
    receipt = validate_receipt(value, construction.cohort)
    if set(value["components"]) != set(scopes):
        raise ValueError("saved initialization scopes differ")
    for name, scope in scopes.items():
        saved = value["components"][name]
        if scope.method == "keep":
            if saved != {"method":"keep","module_type":type(scope.module).__qualname__}:
                raise ValueError("saved host-law KEEP scope differs")
            continue
        parameters = {n: p for n, p in scope.module.named_parameters() if p.requires_grad and p.numel()}
        if (saved["method"] != scope.method or saved["gain"] != scope.gain
                or set(saved["parameters"]) != set(parameters)):
            raise ValueError("saved initialization method/parameters differ")
        for key, parameter in parameters.items():
            entry = saved["parameters"][key]
            if (entry["shape"] != list(parameter.shape) or entry["dtype"] != str(parameter.dtype)
                    or entry["seed"] != scope.parameter_seeds[key]):
                raise ValueError("saved initialization parameter identity differs")
    return receipt  # No call to initialize_fresh_components on restore.


def construction_for(config, packet):
    if config.backend != "minimax_h3_native_av95":
        raise ValueError("historical H3 cohorts require their original loaders")
    if packet is None:
        return Construction("fresh", config.cohort, digest(config.options))
    from particle_sliders.lifecycle import _check_packet
    from particle_sliders.response import TRAINING
    _check_packet(packet)
    if packet["schema"] != TRAINING or packet["manifest"]["config"] != config.as_dict():
        raise ValueError("resume requires exact native response FULL packet/config, no migration")
    game = packet["payload"]["game"]
    if game["config"].get("pg_version") != "0.9.5":
        raise ValueError("historical PG packets retain their original loaders")
    receipt = validate_receipt(deepcopy(game["config"]["backend"]["fresh_initialization"]), config.cohort)
    return Construction("resume", config.cohort, digest(config.options), receipt.canonical_json)
