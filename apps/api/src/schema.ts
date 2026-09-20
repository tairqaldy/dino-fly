/** Drizzle schema — mirrors migrations/0000_init.sql (the committed DDL is what actually runs on deploy). */
import { bigint, bigserial, boolean, doublePrecision, integer, jsonb, pgEnum, pgTable, serial, text, timestamp } from "drizzle-orm/pg-core";

export const agentType = pgEnum("agent_type", ["human", "fly", "ghost"]);
export const dopamineKind = pgEnum("dopamine_kind", ["reward", "punish"]);
export const dopamineSource = pgEnum("dopamine_source", ["game", "human_button", "hardware"]);

export const brains = pgTable("brains", {
  id: serial("id").primaryKey(),
  name: text("name").notNull().unique(),
  connectomeVersion: text("connectome_version").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const generations = pgTable("generations", {
  id: serial("id").primaryKey(),
  brainId: integer("brain_id").notNull(),
  genNumber: integer("gen_number").notNull(),
  parentGenerationId: integer("parent_generation_id"),
  checkpointUrl: text("checkpoint_url"),
  plasticSynapseCount: integer("plastic_synapse_count"),
  trainedOnRunCount: integer("trained_on_run_count").notNull().default(0),
  humanRunCountUsed: integer("human_run_count_used").notNull().default(0),
  heldoutMeanScore: doublePrecision("heldout_mean_score"),
  notes: text("notes"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const runs = pgTable("runs", {
  id: bigserial("id", { mode: "number" }).primaryKey(),
  agentType: agentType("agent_type").notNull(),
  generationId: integer("generation_id"),
  playerName: text("player_name"),
  seed: bigint("seed", { mode: "number" }).notNull(),
  score: integer("score").notNull(),
  durationFrames: integer("duration_frames").notNull(),
  jumps: integer("jumps").notNull(),
  ducks: integer("ducks").notNull(),
  cleared: integer("cleared").notNull(),
  deathCause: integer("death_cause").notNull(),
  actionLogUrl: text("action_log_url").notNull(),
  runNonce: text("run_nonce").unique(),
  clientMeta: jsonb("client_meta").notNull().default({}),
  validated: boolean("validated").notNull().default(false),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const dopamineEvents = pgTable("dopamine_events", {
  id: bigserial("id", { mode: "number" }).primaryKey(),
  runId: bigint("run_id", { mode: "number" }),
  seed: bigint("seed", { mode: "number" }),
  frame: integer("frame"),
  kind: dopamineKind("kind").notNull(),
  magnitude: doublePrecision("magnitude").notNull(),
  source: dopamineSource("source").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const hardwareDevices = pgTable("hardware_devices", {
  deviceId: text("device_id").primaryKey(),
  kind: text("kind").notNull(),
  firmwareVersion: text("firmware_version"),
  lastSeen: timestamp("last_seen", { withTimezone: true }).notNull().defaultNow(),
});
