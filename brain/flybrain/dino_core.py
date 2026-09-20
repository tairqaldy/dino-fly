"""Python port of `packages/dino-core` (engine v1) — the TypeScript engine is the source of truth.

Mirrors `src/engine.ts` statement by statement; `tests/test_dino_core_golden.py` requires this port to reproduce the
committed TS fixtures hash-for-hash (the canonical state string of every frame). All arithmetic is on Python ints:
positions/speeds are fixed point (1 px = 1000), `//` and `%` are only ever applied with positive divisors.

Coordinates: x grows to the right, y is the height above the ground.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

CONSTANTS_PATH = Path(__file__).with_name("data") / "dino_constants.json"
C = json.loads(CONSTANTS_PATH.read_text(encoding="utf-8"))
ENGINE_VERSION: int = C["engineVersion"]
FPX: int = C["fp"]
D = C["dino"]
M32 = 0xFFFFFFFF


# ------------------------------------------------------------------------------------------------ PRNG (rng.ts)
def seed_to_state(seed: int) -> int:
    return seed % 4294967296


def _imul(a: int, b: int) -> int:
    return (a * b) & M32


def rng_next_u32(state: int) -> tuple[int, int]:
    """One mulberry32 step → (new state, uint32 value)."""
    a = (state + 0x6D2B79F5) & M32
    t = _imul(a ^ (a >> 15), 1 | a)
    t = ((t + _imul(t ^ (t >> 7), 61 | t)) & M32) ^ t
    return a, (t ^ (t >> 14)) & M32


def rng_rand_int(state: int, lo: int, hi: int) -> tuple[int, int]:
    if hi < lo:
        raise ValueError("rand_int needs lo <= hi")
    state, value = rng_next_u32(state)
    return state, lo + value % (hi - lo + 1)


# ------------------------------------------------------------------------------------------------ state (types.ts)
class Obstacle(NamedTuple):
    type: int
    x: int
    y_bottom_px: int
    size: int
    width_px: int
    height_px: int
    gap: int
    speed_offset: int
    following_created: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class GameState:
    frame: int
    rng: int
    speed: int
    distance: int
    crashed: bool
    death_type: int
    dino_y: int
    dino_vy: int
    jumping: bool
    ducking: bool
    reached_min_height: bool
    speed_drop: bool
    prev_jump: bool
    jumps: int
    ducks: int
    cleared: int
    history0: int
    history1: int
    obstacles: tuple[Obstacle, ...]


def create_initial_state(seed: int) -> GameState:
    return GameState(
        frame=0,
        rng=seed_to_state(seed),
        speed=C["speed"]["initial"],
        distance=0,
        crashed=False,
        death_type=-1,
        dino_y=0,
        dino_vy=0,
        jumping=False,
        ducking=False,
        reached_min_height=False,
        speed_drop=False,
        prev_jump=False,
        jumps=0,
        ducks=0,
        cleared=0,
        history0=-1,
        history1=-1,
        obstacles=(),
    )


def score(state: GameState) -> int:
    return state.distance // C["distancePerPoint"]


# ------------------------------------------------------------------------------------------------ collisions
def obstacle_boxes(type_: int, size: int) -> list[list[int]]:
    t = C["obstacles"][type_]
    left, middle, right = t["boxes"]
    if size == 1:
        return [left, middle, right]
    width = t["width"] * size
    return [
        left,
        [middle[0], middle[1], width - left[2] - right[2], middle[3]],
        [width - right[2], right[1], right[2], right[3]],
    ]


def _overlaps(a_l: int, a_r: int, a_b: int, a_t: int, b_l: int, b_r: int, b_b: int, b_t: int) -> bool:
    return a_l < b_r and a_r > b_l and a_b < b_t and a_t > b_b


def collides(dino_y: int, ducking: bool, o: Obstacle) -> bool:
    w = D["duckWidth"] if ducking else D["width"]
    h = D["duckHeight"] if ducking else D["height"]
    dino_left = D["x"] * FPX
    obs_bottom = o.y_bottom_px * FPX
    if not _overlaps(
        dino_left + FPX,
        dino_left + (w - 1) * FPX,
        dino_y + FPX,
        dino_y + (h - 1) * FPX,
        o.x + FPX,
        o.x + (o.width_px - 1) * FPX,
        obs_bottom + FPX,
        obs_bottom + (o.height_px - 1) * FPX,
    ):
        return False
    dino_boxes = D["duckBoxes"] if ducking else D["boxes"]
    obs_boxes = obstacle_boxes(o.type, o.size)
    for ax, ay, aw, ah in dino_boxes:
        for bx, by, bw, bh in obs_boxes:
            if _overlaps(
                dino_left + ax * FPX,
                dino_left + (ax + aw) * FPX,
                dino_y + ay * FPX,
                dino_y + (ay + ah) * FPX,
                o.x + bx * FPX,
                o.x + (bx + bw) * FPX,
                obs_bottom + by * FPX,
                obs_bottom + (by + bh) * FPX,
            ):
                return True
    return False


# ------------------------------------------------------------------------------------------------ spawn + step
def _spawn_obstacle(rng: int, speed: int, history0: int, history1: int) -> tuple[int, Obstacle]:
    """Draw order (part of the spec): type (retry while rejected), size, altitude index, speed-offset sign, gap."""
    types = C["obstacles"]
    while True:
        rng, type_ = rng_rand_int(rng, 0, len(types) - 1)
        duplicated = history0 == type_ and history1 == type_
        t = types[type_]
        if not duplicated and speed >= t["minSpeed"]:
            break
    rng, size_draw = rng_rand_int(rng, 1, C["maxObstacleLength"])
    size = 1 if size_draw > 1 and t["multipleSpeed"] > speed else size_draw
    rng, alt = rng_rand_int(rng, 0, len(t["yBottoms"]) - 1)
    rng, sign = rng_rand_int(rng, 0, 1)
    speed_offset = 0 if t["speedOffset"] == 0 else (t["speedOffset"] if sign == 1 else -t["speedOffset"])
    width_px = t["width"] * size
    min_gap = width_px * speed + t["minGap"] * C["gap"]["coefficientMilli"]
    max_gap = (min_gap * C["gap"]["maxNumerator"]) // C["gap"]["maxDenominator"]
    rng, gap = rng_rand_int(rng, min_gap, max_gap)
    return rng, Obstacle(
        type=type_,
        x=(C["world"]["width"] + t["width"]) * FPX,
        y_bottom_px=t["yBottoms"][alt],
        size=size,
        width_px=width_px,
        height_px=t["height"],
        gap=gap,
        speed_offset=speed_offset,
        following_created=False,
        passed=False,
    )


def step(state: GameState, jump: bool, duck: bool) -> GameState:
    if state.crashed:
        return state
    dino_y, dino_vy, jumping, ducking = state.dino_y, state.dino_vy, state.jumping, state.ducking
    reached_min_height, speed_drop, jumps, ducks = state.reached_min_height, state.speed_drop, state.jumps, state.ducks

    # 1) input → dino
    if jumping:
        if duck and not speed_drop:
            speed_drop = True
            dino_vy = D["speedDropVelocity"]
        elif state.prev_jump and not jump and reached_min_height and dino_vy > D["dropVelocity"]:
            dino_vy = D["dropVelocity"]
    elif duck:
        if not ducking:
            ducks += 1
        ducking = True
    else:
        ducking = False
        if jump:
            jumping = True
            dino_vy = D["jumpVelocityBase"] + state.speed // D["jumpVelocitySpeedDivisor"]
            reached_min_height = False
            speed_drop = False
            jumps += 1

    # 2) jump physics
    if jumping:
        dino_y += dino_vy * D["speedDropCoefficient"] if speed_drop else dino_vy
        dino_vy -= D["gravity"]
        if dino_y > D["minJumpHeight"] or speed_drop:
            reached_min_height = True
        if (dino_y > D["maxJumpHeight"] or speed_drop) and reached_min_height and dino_vy > D["dropVelocity"]:
            dino_vy = D["dropVelocity"]
        if dino_y < 0:
            dino_y = 0
            dino_vy = 0
            jumping = False
            speed_drop = False
            reached_min_height = False

    # 3) world
    frame = state.frame + 1
    rng, cleared, history0, history1 = state.rng, state.cleared, state.history0, state.history1
    obstacles: list[Obstacle] = []
    if frame > C["clearFrames"]:
        for o in state.obstacles:
            x = o.x - (state.speed + o.speed_offset)
            passed = o.passed
            if not passed and x + o.width_px * FPX < D["x"] * FPX:
                passed = True
                cleared += 1
            if x + o.width_px * FPX > 0:
                obstacles.append(o._replace(x=x, passed=passed))
        spawn = None
        if not obstacles:
            spawn = _spawn_obstacle(rng, state.speed, history0, history1)
        else:
            last = obstacles[-1]
            if not last.following_created and last.x + last.width_px * FPX + last.gap < C["world"]["width"] * FPX:
                obstacles[-1] = last._replace(following_created=True)
                spawn = _spawn_obstacle(rng, state.speed, history0, history1)
        if spawn is not None:
            rng, new = spawn
            obstacles.append(new)
            history1 = history0
            history0 = new.type

    # 4) collision
    crashed, death_type = False, -1
    for o in obstacles:
        if collides(dino_y, ducking, o):
            crashed, death_type = True, o.type
            break

    # 5) progress
    speed, distance = state.speed, state.distance
    if not crashed:
        distance += speed
        if speed < C["speed"]["max"]:
            speed += C["speed"]["accelPerFrame"]

    return GameState(
        frame=frame,
        rng=rng,
        speed=speed,
        distance=distance,
        crashed=crashed,
        death_type=death_type,
        dino_y=dino_y,
        dino_vy=dino_vy,
        jumping=jumping,
        ducking=ducking,
        reached_min_height=reached_min_height,
        speed_drop=speed_drop,
        prev_jump=jump,
        jumps=jumps,
        ducks=ducks,
        cleared=cleared,
        history0=history0,
        history1=history1,
        obstacles=tuple(obstacles),
    )


# ------------------------------------------------------------------------------------------------ serialize.ts
def state_to_ints(s: GameState) -> list[int]:
    out = [
        s.frame,
        s.rng,
        s.speed,
        s.distance,
        int(s.crashed),
        s.death_type,
        s.dino_y,
        s.dino_vy,
        int(s.jumping),
        int(s.ducking),
        int(s.reached_min_height),
        int(s.speed_drop),
        int(s.prev_jump),
        s.jumps,
        s.ducks,
        s.cleared,
        s.history0,
        s.history1,
        len(s.obstacles),
    ]
    for o in s.obstacles:
        out.extend(int(v) for v in o)
    return out


def canonical_string(s: GameState) -> str:
    return ",".join(str(v) for v in state_to_ints(s))


def state_hash(s: GameState, chars: int = 8) -> str:
    return hashlib.sha256(canonical_string(s).encode("utf-8")).hexdigest()[:chars]


def bits_to_input(bits: int) -> tuple[bool, bool]:
    if bits not in (0, 1, 2, 3):
        raise ValueError(f"input bits must be 0..3, got {bits}")
    return bits in (1, 3), bits >= 2


# ------------------------------------------------------------------------------------------------ replay.ts
def replay(seed: int, log: list[tuple[int, int]], max_frames: int, on_frame=None) -> GameState:
    prev = -1
    for frame, bits in log:
        if frame <= prev:
            raise ValueError("action log frames must be strictly increasing")
        bits_to_input(bits)
        prev = frame
    state = create_initial_state(seed)
    bits, cursor = 0, 0
    for f in range(max_frames):
        if state.crashed:
            break
        if cursor < len(log) and log[cursor][0] == f:
            bits = log[cursor][1]
            cursor += 1
        jump, duck = bits_to_input(bits)
        state = step(state, jump, duck)
        if on_frame is not None:
            on_frame(state)
    return state


def with_obstacles(state: GameState, obstacles: tuple[Obstacle, ...]) -> GameState:
    """Test helper: same state with a different obstacle list."""
    return replace(state, obstacles=obstacles)
