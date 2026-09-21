import { useState } from "react";

/**
 * First-run explainer. Three cards, dismissible, remembered per browser — the page has to say what this is
 * before it asks anyone to race a connectome.
 */
const KEY = "dino-fly-onboarded-v1";

const STEPS = [
  {
    title: "This dino is driven by a real fly brain",
    body: "The whole Drosophila connectome — 138,639 neurons, 15 million connections, reconstructed from electron microscopy — runs as a spiking network on a GPU. No neural network was trained on top of it.",
    hint: "The orange dino is the fly. The white one is you.",
  },
  {
    title: "How the fly sees and jumps",
    body: "Each cactus is shown to the fly the way neuroscientists show looming objects to real flies: it drives the LPLC2 and LC4 looming detectors. When the fly's own escape neuron — the Giant Fiber — fires, the JUMP key is pressed.",
    hint: "Watch the little fly at the keyboard: it presses ↑ exactly when the Giant Fiber spikes.",
  },
  {
    title: "Your turn",
    body: "Press SPACE (or tap) to race the fly on the same obstacles. ↓ ducks. Your score is validated by replaying your run, so the leaderboard is honest. You can also play with your body through the laptop camera.",
    hint: "Every run you submit is stored and can later be used to teach the fly.",
  },
] as const;

export function Onboarding() {
  const [step, setStep] = useState(() => {
    try {
      return localStorage.getItem(KEY) ? -1 : 0;
    } catch {
      return 0;
    }
  });
  if (step < 0) return null;
  const card = STEPS[step] as (typeof STEPS)[number];
  const close = () => {
    try {
      localStorage.setItem(KEY, "1");
    } catch {
      /* private mode: just close for this session */
    }
    setStep(-1);
  };
  return (
    <div className="onboard-backdrop" role="dialog" aria-modal="true" aria-label="What is dino-fly?">
      <div className="onboard">
        <div className="onboard-dots">
          {STEPS.map((s, i) => (
            <span key={s.title} className={i === step ? "onboard-dot on" : "onboard-dot"} />
          ))}
        </div>
        <h2>{card.title}</h2>
        <p>{card.body}</p>
        <p className="hint">{card.hint}</p>
        <div className="onboard-actions">
          <button type="button" className="ghost" onClick={close}>
            skip
          </button>
          <button type="button" onClick={() => (step === STEPS.length - 1 ? close() : setStep(step + 1))}>
            {step === STEPS.length - 1 ? "let me play" : "next"}
          </button>
        </div>
      </div>
    </div>
  );
}

/** Small "?" button that replays the onboarding. */
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
