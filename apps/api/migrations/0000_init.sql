-- dino-fly schema v0. Applied by `pnpm --filter @dino-fly/api migrate` (runs on deploy). Postgres holds metadata only;
-- action logs and checkpoints live in object storage (R2) and are referenced by URL/key.

CREATE TABLE brains (
  id              serial PRIMARY KEY,
  name            text NOT NULL UNIQUE,
  connectome_version text NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE generations (
  id                      serial PRIMARY KEY,
  brain_id                integer NOT NULL REFERENCES brains(id),
  gen_number              integer NOT NULL,
  parent_generation_id    integer REFERENCES generations(id),
  checkpoint_url          text,
  plastic_synapse_count   integer,
  trained_on_run_count    integer NOT NULL DEFAULT 0,
  human_run_count_used    integer NOT NULL DEFAULT 0,
  heldout_mean_score      double precision,
  notes                   text,
  created_at              timestamptz NOT NULL DEFAULT now(),
  UNIQUE (brain_id, gen_number)
);

CREATE TYPE agent_type AS ENUM ('human', 'fly', 'ghost');

CREATE TABLE runs (
  id              bigserial PRIMARY KEY,
  agent_type      agent_type NOT NULL,
  generation_id   integer REFERENCES generations(id),
  player_name     text,
  seed            bigint NOT NULL,
  score           integer NOT NULL,
  duration_frames integer NOT NULL,
  jumps           integer NOT NULL,
  ducks           integer NOT NULL,
  cleared         integer NOT NULL,
  death_cause     integer NOT NULL,
  action_log_url  text NOT NULL,
  run_nonce       text UNIQUE,
  client_meta     jsonb NOT NULL DEFAULT '{}'::jsonb,
  validated       boolean NOT NULL DEFAULT false,
  created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX runs_agent_score_idx ON runs (agent_type, score DESC);
CREATE INDEX runs_seed_idx ON runs (seed, agent_type);

CREATE TABLE leaderboard_snapshots (
  id          bigserial PRIMARY KEY,
  payload     jsonb NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TYPE dopamine_kind AS ENUM ('reward', 'punish');
CREATE TYPE dopamine_source AS ENUM ('game', 'human_button', 'hardware');

CREATE TABLE dopamine_events (
  id          bigserial PRIMARY KEY,
  run_id      bigint REFERENCES runs(id),
  seed        bigint,
  frame       integer,
  kind        dopamine_kind NOT NULL,
  magnitude   double precision NOT NULL,
  source      dopamine_source NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE human_teaching_sessions (
  id                  bigserial PRIMARY KEY,
  generation_id       integer NOT NULL REFERENCES generations(id),
  human_run_ids       bigint[] NOT NULL,
  mechanism           text NOT NULL,           -- 'observational_replay' | 'death_curriculum'
  heldout_mean_before double precision,
  heldout_mean_after  double precision,
  notes               text,
  created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE hardware_devices (
  device_id         text PRIMARY KEY,
  kind              text NOT NULL,
  firmware_version  text,
  last_seen         timestamptz NOT NULL DEFAULT now()
);
