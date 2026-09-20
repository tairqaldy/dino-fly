import type { GameState } from "@dino-fly/dino-core";
import { type RenderOptions, render, WORLD } from "@dino-fly/dino-render";
import { useEffect, useRef } from "react";

interface Props {
  state: GameState | null;
  options?: RenderOptions;
  /** Re-render trigger for states mutated outside React (the 60 Hz game loop). */
  getState?: () => { state: GameState; options?: RenderOptions } | null;
}

/** 600×150 world, integer-scaled with crisp pixels; draws either a prop state or polls `getState` every frame. */
export function GameCanvas({ state, options, getState }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    ctx.imageSmoothingEnabled = false;
    let raf = 0;
    const draw = () => {
      const current = getState ? getState() : state ? { state, options } : null;
      if (current) render(ctx, current.state, current.options);
      if (getState) raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [state, options, getState]);

  return <canvas ref={ref} className="game-canvas" width={WORLD.width} height={WORLD.height} />;
}
