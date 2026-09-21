import { useState } from "react";

/**
 * First-run explainer and the name prompt. The page has to say what this is before it asks anyone to race a
 * connectome, and every run is stored under the name given here, so it is asked for once, up front.
 */
const KEY = "dino-fly-onboarded-v2";
const NAME_KEY = "dino-fly-name";

const STEPS = [
  {
    title: "This dino is driven by a real fly brain",
    body: "The whole Drosophila connectome — 138,639 neurons, 15 million connections, reconstructed from electron microscopy — runs as a spiking network. No neural network was trained on top of it.",
    hint: "Top screen: the fly. Bottom screen: you.",
  },
  {
    title: "How the fly sees and jumps",
    body: "Each cactus is shown to the fly the way neuroscientists show looming objects to real flies: it drives the LPLC2 and LC4 looming detectors. When the fly's own escape neuron — the Giant Fiber — fires, the JUMP key is pressed.",
    hint: "Watch the little fly at the keyboard: it presses ↑ exactly when the Giant Fiber spikes.",
  },
] as const;

function remember(name: string): void {
  try {
    localStorage.setItem(NAME_KEY, name);
  } catch {
    /* private mode: keep it for this session only */
  }
}

export function Onboarding({ name, onName }: { name: string; onName: (n: string) => void }) {
  const [step, setStep] = useState(() => {
    try {
      return localStorage.getItem(KEY) ? -1 : 0;
    } catch {
      return 0;
    }
  });
  const [draft, setDraft] = useState(name);
  if (step < 0) return null;
  const last = step === STEPS.length;
  const card = STEPS[step];
  const close = () => {
    const chosen = draft.trim();
    if (chosen) {
      onName(chosen);
      remember(chosen);
    }
    try {
      localStorage.setItem(KEY, "1");
    } catch {
      /* ignore */
    }
    setStep(-1);
  };
  return (
    <div className="onboard-backdrop" role="dialog" aria-modal="true" aria-label="What is dino-fly?">
      <div className="onboard">
        <div className="onboard-dots">
          {[...STEPS, { title: "name" }].map((s, i) => (
            <span key={s.title} className={i === step ? "onboard-dot on" : "onboard-dot"} />
          ))}
        </div>
        {last ? (
          <>
            <h2>What should we call you?</h2>
            <p>
              Every run you play is stored with this name: it goes on the leaderboard, and the input log becomes part of the
              corpus the fly can later be taught from.
            </p>
            <input
              value={draft}
              maxLength={24}
              placeholder="anonymous"
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && close()}
            />
            <p className="hint">You can change it any time under the game.</p>
          </>
        ) : (
          <>
            <h2>{card?.title}</h2>
            <p>{card?.body}</p>
            <p className="hint">{card?.hint}</p>
          </>
        )}
        <div className="onboard-actions">
          <button type="button" className="ghost" onClick={close}>
            skip
          </button>
          <button type="button" onClick={() => (last ? close() : setStep(step + 1))}>
            {last ? "let me play" : "next"}
          </button>
        </div>
      </div>
    </div>
  );
}

/** If the onboarding was skipped earlier and there is still no name, ask for it once before the first run. */
export function NameGate({ name, onName }: { name: string; onName: (n: string) => void }) {
  const [draft, setDraft] = useState("");
  const [done, setDone] = useState(false);
  let seen = true;
  try {
    seen = localStorage.getItem(KEY) !== null;
  } catch {
    seen = true;
  }
  if (name || done || !seen) return null;
  const save = () => {
    const chosen = draft.trim();
    if (chosen) {
      onName(chosen);
      remember(chosen);
    }
    setDone(true);
  };
  return (
    <div className="onboard-backdrop" role="dialog" aria-modal="true" aria-label="Your name">
      <div className="onboard">
        <h2>What should we call you?</h2>
        <p>Your runs are saved under this name — on the leaderboard, and in the corpus the fly can be taught from.</p>
        <input value={draft} maxLength={24} placeholder="anonymous" onChange={(e) => setDraft(e.target.value)} onKeyDown={(e) => e.key === "Enter" && save()} />
        <div className="onboard-actions">
          <button type="button" className="ghost" onClick={() => setDone(true)}>
            play anonymously
          </button>
          <button type="button" onClick={save}>
            save
          </button>
        </div>
      </div>
    </div>
  );
}

/** Small button that replays the onboarding. */
export function OnboardingReplayButton() {
  return (
    <button
      type="button"
      className="ghost"
      onClick={() => {
        try {
          localStorage.removeItem(KEY);
        } catch {
          /* ignore */
        }
        window.location.reload();
      }}
    >
      how does this work?
    </button>
  );
}
