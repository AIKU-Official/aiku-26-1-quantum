# pooling 방식 디스패처 — STEP A 정본(cond) 전환용.
#   --method cry  → src/circuit.py      (CRY coherent pooling, 기존/기본값)
#   --method cond → src/circuit_cond.py (정본 mid-circuit measurement + qml.cond)
#
#   두 모듈은 make_qnode / n_quantum_params / n_party_params / default_routing /
#   n_readout 시그니처가 동일하므로 워커는 get(method)로 받아 그대로 쓰면 된다.
#
#   캐시 분리: cry 결과는 기존 cache/<group>/ 그대로(절대 안 건드림),
#             cond 결과는 cache/cond/<group>/ 아래로 분리 저장.

import os
from . import config
from . import circuit as _cry
from . import circuit_cond as _cond

METHODS = ("cry", "cond")


def get(method):
    """method 문자열 → 회로 모듈."""
    if method == "cry":
        return _cry
    if method == "cond":
        return _cond
    raise ValueError(f"알 수 없는 method: {method} (가능: {METHODS})")


def cache_subdir(group, method):
    """그룹별 캐시 디렉터리. cry=cache/<group>, cond=cache/cond/<group>."""
    if method == "cry":
        d = os.path.join(config.CACHE_DIR, group)
    else:
        d = os.path.join(config.CACHE_DIR, "cond", group)
    os.makedirs(d, exist_ok=True)
    return d
