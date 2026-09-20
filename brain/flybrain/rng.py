"""Counter-based random numbers for Poisson drive.

Instead of a stateful generator (whose stream depends on tensor shapes and call order), every Bernoulli draw is a
pure function of (trial seed, neuron index, step):

    u32 = mix32( key(neuron, seed) XOR mix32(step XOR STEP_SALT) )        spike  <=>  u32 < p_u32

Consequences: outcomes are invariant to batch size, chunk size, column recycling and device; the NumPy reference
simulator can reproduce Poisson drive bit-for-bit; and different experimental conditions can share common random
numbers. All functions work on NumPy int64 arrays and on torch int64 tensors alike (only `& | ^ >> << * +`).

`mix32` is the "lowbias32" integer hash (Chris Wellons, public domain). 32-bit multiplication is done in two 16-bit
halves so no intermediate ever exceeds 2^49 — no reliance on signed-overflow wraparound.
"""

from __future__ import annotations

M32 = 0xFFFFFFFF
STEP_SALT = 0x243F6A88
NEURON_SALT = 0x9E3779B9


def _mul32(x, c: int):
    """(x * c) mod 2^32 for 0 <= x < 2^32, without overflowing int64."""
    lo = x & 0xFFFF
    hi = x >> 16
    return (lo * c + (((hi * c) & 0xFFFF) << 16)) & M32


def mix32(x):
    """Bijective 32-bit mixer (lowbias32). Input/outputs are int64 containers holding values in [0, 2^32)."""
    x = x ^ (x >> 16)
    x = _mul32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _mul32(x, 0x846CA68B)
    x = x ^ (x >> 16)
    return x


def split_seed(seed: int) -> tuple[int, int]:
    """64-bit trial seed → (low 32 bits, high 32 bits)."""
    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    return seed & M32, (seed >> 32) & M32


def stream_key(neuron_idx, seed_lo, seed_hi):
    """Per-(neuron, trial) stream key. Arguments broadcast against each other."""
    h = mix32((neuron_idx + NEURON_SALT) & M32)
    h = mix32(h ^ seed_lo)
    h = mix32(h ^ seed_hi)
    return h


def draw_u32(key, step):
    """Uniform uint32 for stream `key` at (trial-relative) `step`. Arguments broadcast against each other."""
    return mix32(key ^ mix32((step & M32) ^ STEP_SALT))


def rate_to_p_u32(rate_hz: float, dt_ms: float) -> int:
    """Bernoulli probability rate·dt as a 32-bit threshold (Brian2's PoissonInput with N=1 draws Binomial(1, rate·dt))."""
    p = float(rate_hz) * float(dt_ms) * 1e-3
    if p < 0:
        raise ValueError("negative rate")
    return min(int(p * 4294967296.0), 4294967296)
