/** Build-time configuration (Vite env). Everything dynamic comes from the API / the worker feed at run time. */
const env = import.meta.env;

export const CONFIG = {
  /** Live fly feed: the laptop worker on the LAN in local mode, or the public API hub. */
  flyWs: (env.VITE_FLY_WS as string | undefined) ?? "ws://localhost:8765",
  /** REST API (leaderboard, run submission, ghosts). Empty string = offline mode (bundled ghosts only). */
  apiUrl: (env.VITE_API_URL as string | undefined) ?? "http://localhost:8787",
  labMode: (env.VITE_LAB_MODE as string | undefined) === "1" || env.DEV,
};
