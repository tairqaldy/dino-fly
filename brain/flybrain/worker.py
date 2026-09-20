"""The brain worker: a long-running process in which the fly plays, streamed live over WebSocket.

    uv run --no-sync flybrain worker                       # local dashboard feed on ws://localhost:8765
    uv run --no-sync flybrain worker --api wss://…/worker --token …   # also push to the public API hub (outbound only)

One brain (B = 1, dense state + CUDA graph for the lowest per-frame latency) plays game after game. Every frame a
`fly.frame` message carries the canonical game state, the Giant Fiber spikes, the commanded LC4/LPLC2 rates and the
spike counts of a few cell groups for the brain map. Nothing inbound is exposed beyond the local WebSocket.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from collections import deque

import numpy as np

from flybrain import dino_core as dc
from flybrain import protocol as proto
from flybrain.play import BIO_MS_PER_FRAME, MAX_FRAMES, stream_key
from flybrain.transducer.looming import FROZEN, TRANSDUCER_VERSION, obstacle_view, population_rates
from flybrain.transducer.motor import JumpMotor, MotorParams

GROUPS = ("LC4", "LPLC2", "GF", "DN", "KC", "MBON", "PAM", "PPL1")


class LiveFly(threading.Thread):
    """Plays in a background thread and hands protocol envelopes to `emit` (thread-safe callable)."""

    def __init__(self, emit, *, connectome: str = "flywire783", device: str = "cuda", first_seed: int = 1_000_000,
                 target_fps: float = 60.0, worker_id: str = "laptop") -> None:
        super().__init__(daemon=True)
        self.emit, self.connectome, self.device = emit, connectome, device
        self.seed, self.target_fps, self.worker_id = first_seed, target_fps, worker_id
        self.running = threading.Event()
        self.running.set()
        self.stop_flag = False
        self.generation = 0
        self.games, self.best, self.jumps, self.deaths = 0, 0, 0, 0
        self.recent: deque[int] = deque(maxlen=50)
        self.learning_curve: list[proto.LearningPoint] = []
        self.pending_dopamine: list[proto.DopamineEvent] = []
        self.hello: dict | None = None

    # -- commands from the lab UI / hardware
    def command(self, cmd: proto.LabCommand) -> None:
        if cmd.command == "start":
            self.running.set()
        elif cmd.command == "stop":
            self.running.clear()
        elif cmd.command == "set_seed" and cmd.value is not None:
            self.seed = int(cmd.value)
        elif cmd.command in ("reward", "punish"):
            self.dopamine(proto.DopamineEvent(kind=cmd.command, magnitude=cmd.value or 1.0, source="human_button"))

    def dopamine(self, event: proto.DopamineEvent) -> None:
        self.pending_dopamine.append(event)

    def run(self) -> None:  # pragma: no cover - needs a GPU and the downloaded connectome
        import torch

        from flybrain import neurons
        from flybrain.connectome import load_connectome
        from flybrain.lif import LIFNetwork, LIFParams, PoissonDrive

        conn = load_connectome(self.connectome)
        idx = {g: neurons.indices(conn, "DN_ALL" if g == "DN" else g) for g in GROUPS}
        p = LIFParams()
        net = LIFNetwork(conn, p, batch_size=1, device=self.device, chunk_steps=10,
                         use_cuda_graph=self.device == "cuda", active_set=False)
        drive = PoissonDrive(np.concatenate([idx["LC4"], idx["LPLC2"]]), 1, p.dt_ms, device=self.device)
        net.set_drive(drive)
        groups_t = [torch.as_tensor(idx[g], device=self.device) for g in GROUPS]
        steps = round(BIO_MS_PER_FRAME / p.dt_ms)
        self.hello = proto.envelope("fly.hello", proto.FlyHello(
            workerId=self.worker_id, connectome=conn.name, neurons=conn.n, engineVersion=dc.ENGINE_VERSION,
            transducerVersion=TRANSDUCER_VERSION, generation=self.generation, bioMsPerFrame=BIO_MS_PER_FRAME,
            gpu=torch.cuda.get_device_name(0) if self.device == "cuda" else "cpu", groups=[*GROUPS, "other"]))
        self.emit(self.hello)

        while not self.stop_flag:
            seed = self.seed
            self.seed += 1
            state, motor = dc.create_initial_state(seed), JumpMotor(MotorParams())
            net.reset()
            drive.set_seeds(np.array([stream_key(self.generation, seed)], dtype=np.uint64))
            actions, bits, t_game, jump = [], 0, time.perf_counter(), False
            n_lc4, n_lplc2 = len(idx["LC4"]), len(idx["LPLC2"])
            while not state.crashed and state.frame < MAX_FRAMES and not self.stop_flag:
                self.running.wait()
                t0 = time.perf_counter()
                view = obstacle_view(state, BIO_MS_PER_FRAME)
                lc4, lplc2 = population_rates(view.theta_deg, view.theta_dot_deg_s, FROZEN)
                drive.set_rates(np.concatenate([np.full(n_lc4, float(lc4)), np.full(n_lplc2, float(lplc2))]))
                net.run(steps, record="counts")
                per_group = torch.stack([net.spike_counts[g, 0].sum() for g in groups_t]).tolist()
                total = int(net.spike_counts[:, 0].sum())
                gf = int(per_group[GROUPS.index("GF")])
                jump = motor.update(gf, airborne=state.jumping)
                b = 1 if jump else 0
                if b != bits:
                    actions.append((state.frame, b))
                    bits = b
                state = dc.step(state, jump, False)
                dopa = None
                if self.pending_dopamine:
                    ev = self.pending_dopamine.pop(0)
                    dopa = [ev.magnitude if ev.kind == "reward" else 0.0, ev.magnitude if ev.kind == "punish" else 0.0]
                    self.emit(proto.envelope("dopamine.event", ev.model_copy(update={"seed": seed, "frame": state.frame})))
                self.emit(proto.envelope("fly.frame", proto.FlyFrame(
                    seed=seed, state=dc.state_to_ints(state), gf=gf, jump=jump, theta=view.theta_deg,
                    thetaDot=view.theta_dot_deg_s, rates=(float(lc4), float(lplc2)),
                    activity=[*(int(x) for x in per_group), max(total - int(sum(per_group)), 0)], dopamine=dopa)))
                spare = 1.0 / self.target_fps - (time.perf_counter() - t0)
                if spare > 0:
                    time.sleep(spare)
            wall = max(time.perf_counter() - t_game, 1e-6)
            self.games += 1
            self.deaths += int(state.crashed)
            self.jumps += state.jumps
            self.best = max(self.best, dc.score(state))
            self.recent.append(dc.score(state))
            self.emit(proto.envelope("fly.run_end", proto.FlyRunEnd(
                seed=seed, generation=self.generation, score=dc.score(state), frames=state.frame, cleared=state.cleared,
                deathType=state.death_type, jumps=state.jumps, actions=actions)))
            self.emit(proto.envelope("fly.stats", proto.FlyStats(
                generation=self.generation, gamesPlayed=self.games, bestScore=self.best,
                meanScoreRecent=float(np.mean(self.recent)), totalJumps=self.jumps, totalDeaths=self.deaths,
                realtimeFactor=state.frame * BIO_MS_PER_FRAME * 1e-3 / wall, learningCurve=self.learning_curve)))


async def serve(host: str, port: int, *, api: str | None, token: str | None, device: str, connectome: str) -> None:  # pragma: no cover
    import websockets

    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue[dict] = asyncio.Queue(maxsize=512)
    clients: set = set()

    def emit(message: dict) -> None:
        def put() -> None:
            if outbox.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    outbox.get_nowait()  # drop the oldest frame rather than stall the brain
            outbox.put_nowait(message)

        loop.call_soon_threadsafe(put)

    fly = LiveFly(emit, device=device, connectome=connectome)

    def handle_incoming(raw: str) -> None:
        try:
            kind, payload = proto.parse(json.loads(raw))
        except Exception as exc:  # malformed input must never kill the worker
            print(f"[worker] ignored message: {exc}")
            return
        if kind == "lab.command":
            fly.command(payload)
        elif kind == "dopamine.event":
            fly.dopamine(payload)

    async def client(ws) -> None:
        clients.add(ws)
        try:
            if fly.hello is not None:
                await ws.send(json.dumps(fly.hello))
            async for raw in ws:
                handle_incoming(raw)
        finally:
            clients.discard(ws)

    async def hub() -> None:
        """Outbound connection to the public API (the laptop exposes nothing inbound)."""
        frame_no = 0
        while api:
            try:
                async with websockets.connect(f"{api}?token={token or ''}") as ws:
                    print(f"[worker] connected to hub {api}")
                    if fly.hello is not None:
                        await ws.send(json.dumps(fly.hello))
                    hub_clients.add(ws)
                    async for raw in ws:
                        handle_incoming(raw)
            except Exception as exc:
                print(f"[worker] hub connection lost ({exc}); retrying in 5 s")
            finally:
                hub_clients.clear()
            await asyncio.sleep(5)
            frame_no += 1

    hub_clients: set = set()

    async def pump() -> None:
        n = 0
        while True:
            message = await outbox.get()
            data = json.dumps(message)
            websockets.broadcast(clients, data)
            n += 1
            # the public hub gets every third frame (~20 Hz) and all non-frame messages
            if hub_clients and (message["type"] != "fly.frame" or n % 3 == 0):
                websockets.broadcast(hub_clients, data)

    fly.start()
    async with websockets.serve(client, host, port):
        print(f"[worker] local feed on ws://{host}:{port}  (connectome {connectome}, device {device})")
        await asyncio.gather(pump(), hub())


def main(host: str = "127.0.0.1", port: int = 8765, api: str | None = None, token: str | None = None,
         device: str = "cuda", connectome: str = "flywire783") -> None:  # pragma: no cover
    asyncio.run(serve(host, port, api=api, token=token, device=device, connectome=connectome))
