# Protocol

Single source of truth: [`packages/protocol/schemas/messages.schema.json`](../packages/protocol/schemas/messages.schema.json)
(JSON Schema 2020-12). TypeScript types (`packages/protocol/src/index.ts`) and pydantic models
(`brain/flybrain/protocol.py`) mirror it; both test suites validate the shared examples in
`packages/protocol/examples/messages.json` against the schema, and everything the Python side emits is validated too.

## Envelope

```json
{ "type": "fly.frame", "v": 1, "ts": 1790000000016, "payload": { } }
```

`v` is the protocol version (1), `ts` the sender's clock in ms. FlyWire root IDs never travel as JSON numbers
(they exceed 2^53); where IDs are needed they are strings.

## Messages

| type | direction | payload |
|---|---|---|
| `fly.hello` | worker → hub → browsers | connectome, neuron count, engine / transducer versions, generation, `bioMsPerFrame`, GPU, names of the activity groups |
| `fly.frame` | worker → … (every frame locally, every 3rd frame to the public hub) | `seed`, canonical game `state` (the integer list of `stateToInts`), `gf` spikes this frame, `jump` key, looming `theta` / `thetaDot`, commanded `rates` [LC4, LPLC2] Hz, `activity` = spikes per group, optional `dopamine` [reward, punish] |
| `fly.run_end` | worker → hub | seed, generation, score, frames, cleared, deathType, jumps, `actions` log — the hub **re-validates it by replay** before storing |
| `fly.stats` | worker → … | generation, gamesPlayed, bestScore, meanScoreRecent, totals, realtimeFactor, learningCurve |
| `fly.status` | hub → browsers | `online` (is a worker connected?) |
| `dopamine.event` | worker / lab UI / ESP32 buttons | kind `reward` \| `punish`, magnitude, source `game` \| `human_button` \| `hardware` |
| `human.input` | pose pipeline / clients | source `keyboard` \| `touch` \| `pose`, action `jump` \| `duck` \| `release` |
| `lab.command` | lab UI → worker (local only) | `start`, `stop`, `set_seed`, `set_generation`, `reward`, `punish` (+ `value`) |
| `leaderboard.update` | hub → browsers | top humans, fly best per generation, `flyPercentile` |

## Endpoints

| Where | URL | Notes |
|---|---|---|
| worker, local feed | `ws://<laptop>:8765` | full-rate feed; accepts `lab.command` and `dopamine.event` |
| hub, browsers | `wss://<api>/ws` | read-only |
| hub, worker ingress | `wss://<api>/worker?token=WORKER_TOKEN` | the laptop connects **outbound** |
| REST | `POST /api/runs/start` → `{seed, runToken}` · `POST /api/runs` `{seed, runToken, name, actions, frames}` → `{accepted, score, rank}` · `GET /api/leaderboard` · `GET /api/ghost?seed=` · `GET /api/runs/human?limit=` · `GET /health` | scores are always the server's replay, never the client's claim |

Action log format (shared by fixtures, API and worker): `[[stepIndex, bits], …]`, strictly increasing step indices,
`bits` = 1 jump, 2 duck, 3 both; an entry holds until the next one.
