import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { CONFIG } from "./lib/config.js";
import { useFlyFeed } from "./lib/feed.js";
import { Brain } from "./pages/Brain.js";
import { Lab, Leaderboard, Research } from "./pages/Other.js";
import { Play } from "./pages/Play.js";
import "./styles.css";

const PAGES = ["play", "brain", "leaderboard", "research", ...(CONFIG.labMode ? ["lab"] : [])] as const;
type Page = (typeof PAGES)[number];

function currentPage(): Page {
  const hash = window.location.hash.replace(/^#\/?/, "");
  return (PAGES as readonly string[]).includes(hash) ? (hash as Page) : "play";
}

function App() {
  const [page, setPage] = useState<Page>(currentPage);
  const [feed, control] = useFlyFeed(CONFIG.flyWs);
  useEffect(() => {
    const onHash = () => setPage(currentPage());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return (
    <>
      <header>
        <a className="brand" href="#/play">
          dino<span className="accent">·</span>fly
        </a>
        <nav>
          {PAGES.map((p) => (
            <a key={p} href={`#/${p}`} className={p === page ? "active" : ""}>
              {p}
            </a>
          ))}
        </nav>
      </header>
      <main>
        {page === "play" && <Play feed={feed} />}
        {page === "brain" && <Brain feed={feed} />}
        {page === "leaderboard" && <Leaderboard feed={feed} />}
        {page === "research" && <Research />}
        {page === "lab" && <Lab feed={feed} control={control} />}
      </main>
      <footer>
        FlyWire connectome (CC BY 4.0) · whole-brain model after Shiu et al., Nature 2024 · no trained readout, ever ·{" "}
        <a href="https://github.com/tairqaldy/dino-fly">source</a>
      </footer>
    </>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
