import { useEffect, useRef } from "react";
import * as THREE from "three";

/**
 * The fly at the keyboard. This is a *depiction of the motor transducer*, not a measurement: when the Giant Fiber
 * fires, `flybrain/transducer/motor.py` presses JUMP — here a leg presses the ↑ key. Ducking presses ↓. The dance
 * and the tears are decoration (a fly does not celebrate); everything scientific lives in the research log.
 */
export type FlyAction = "idle" | "jump" | "duck";
export type FlyMood = "neutral" | "win" | "lose";

const BODY = 0x6b6f80; // light enough to read against the dark panel behind the canvas
const ACCENT = 0xff7a45;

function keyTexture(glyph: string): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d") as CanvasRenderingContext2D;
  g.fillStyle = "#1b1c22";
  g.fillRect(0, 0, 128, 128);
  g.strokeStyle = "#3a3c48";
  g.lineWidth = 6;
  g.strokeRect(6, 6, 116, 116);
  g.fillStyle = "#e8e8ee";
  g.font = "bold 78px system-ui, sans-serif";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(glyph, 64, 68);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/** One leg: two segments with a knee, so it can be folded to reach a key. */
function makeLeg(sign: number, z: number, spread: number): { group: THREE.Group; knee: THREE.Group } {
  const mat = new THREE.MeshStandardMaterial({ color: 0x35384a, roughness: 0.8 });
  const group = new THREE.Group();
  const upper = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.015, 0.3, 6), mat);
  upper.position.y = -0.15;
  upper.rotation.z = sign * spread;
  upper.position.x = sign * 0.06;
  group.add(upper);
  const knee = new THREE.Group();
  knee.position.set(sign * 0.13, -0.27, 0);
  const lower = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.01, 0.26, 6), mat);
  lower.position.y = -0.13;
  knee.add(lower);
  group.add(knee);
  group.position.set(0, -0.04, z);
  return { group, knee };
}

function makeFly(): {
  root: THREE.Group;
  wings: THREE.Mesh[];
  legs: { group: THREE.Group; knee: THREE.Group }[];
  head: THREE.Group;
  tears: THREE.Mesh[];
} {
  const root = new THREE.Group();
  const shell = new THREE.MeshStandardMaterial({ color: BODY, roughness: 0.45, metalness: 0.25 });

  const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.22, 24, 18), shell);
  thorax.scale.set(1, 0.85, 1.15);
  root.add(thorax);

  const abdomen = new THREE.Mesh(new THREE.SphereGeometry(0.2, 24, 18), new THREE.MeshStandardMaterial({ color: 0x4e5262, roughness: 0.5 }));
  abdomen.scale.set(0.95, 0.8, 1.5);
  abdomen.position.set(0, -0.02, -0.34);
  root.add(abdomen);

  const head = new THREE.Group();
  head.position.set(0, 0.04, 0.26);
  const skull = new THREE.Mesh(new THREE.SphereGeometry(0.15, 22, 16), shell);
  skull.scale.set(1, 0.9, 0.85);
  head.add(skull);
  const eyeMat = new THREE.MeshStandardMaterial({ color: 0xff4a2f, emissive: 0x8d1c0a, emissiveIntensity: 1.2, roughness: 0.2 });
  for (const s of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.085, 18, 14), eyeMat);
    eye.position.set(s * 0.095, 0.02, 0.05);
    eye.scale.set(0.9, 1.05, 0.9);
    head.add(eye);
    const antenna = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.006, 0.12, 5), new THREE.MeshStandardMaterial({ color: 0x15161b }));
    antenna.position.set(s * 0.04, 0.12, 0.06);
    antenna.rotation.set(-0.5, 0, s * 0.35);
    head.add(antenna);
  }
  root.add(head);

  const wingMat = new THREE.MeshPhysicalMaterial({
    color: 0xeaf0ff,
    transparent: true,
    opacity: 0.55,
    roughness: 0.1,
    transmission: 0.6,
    side: THREE.DoubleSide,
  });
  const wings: THREE.Mesh[] = [];
  for (const s of [-1, 1]) {
    const wing = new THREE.Mesh(new THREE.CircleGeometry(0.34, 20), wingMat);
    wing.scale.set(0.5, 1, 1);
    wing.position.set(s * 0.1, 0.16, -0.18);
    wing.rotation.set(-Math.PI / 2.2, 0, s * 0.5);
    wings.push(wing);
    root.add(wing);
  }

  const legs = [makeLeg(-1, 0.14, 0.9), makeLeg(1, 0.14, 0.9), makeLeg(-1, -0.04, 1.2), makeLeg(1, -0.04, 1.2), makeLeg(-1, -0.22, 1.35), makeLeg(1, -0.22, 1.35)];
  for (const l of legs) root.add(l.group);

  const tearMat = new THREE.MeshStandardMaterial({ color: 0x8ec5ff, emissive: 0x1b3f6b, transparent: true, opacity: 0.9 });
  const tears = [0, 1, 2, 3].map((i) => {
    const drop = new THREE.Mesh(new THREE.SphereGeometry(0.035, 10, 8), tearMat.clone());
    drop.scale.set(1, 1.35, 1);
    drop.position.set((i % 2 === 0 ? -1 : 1) * 0.1, 0, 0.32);
    drop.visible = false;
    root.add(drop);
    return drop;
  });

  return { root, wings, legs, head, tears };
}

function makeLaptop(): { group: THREE.Group; up: THREE.Mesh; down: THREE.Mesh } {
  const group = new THREE.Group();
  const shell = new THREE.MeshStandardMaterial({ color: 0x3a3d47, roughness: 0.6, metalness: 0.4 });

  const base = new THREE.Mesh(new THREE.BoxGeometry(2.4, 0.09, 1.6), shell);
  base.position.y = -0.045;
  base.receiveShadow = true;
  group.add(base);

  const hinge = new THREE.Group();
  hinge.position.z = -0.8;
  hinge.rotation.x = -0.32; // screen leaning back
  const screen = new THREE.Mesh(new THREE.BoxGeometry(2.4, 1.5, 0.07), shell);
  screen.position.set(0, 0.75, 0);
  hinge.add(screen);
  const glow = new THREE.Mesh(
    new THREE.PlaneGeometry(2.2, 1.32),
    new THREE.MeshStandardMaterial({ color: 0x0d1016, emissive: 0x18324d, emissiveIntensity: 0.9, roughness: 1 }),
  );
  glow.position.set(0, 0.75, 0.04);
  hinge.add(glow);
  group.add(hinge);

  const keyFaces = (glyph: string) => {
    const side = new THREE.MeshStandardMaterial({ color: 0x25262e, roughness: 0.7 });
    const top = new THREE.MeshStandardMaterial({ map: keyTexture(glyph), roughness: 0.55 });
    return [side, side, top, side, side, side];
  };
  const up = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.08, 0.42), keyFaces("↑"));
  up.position.set(0.28, 0.035, 0.22);
  const down = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.08, 0.42), keyFaces("↓"));
  down.position.set(-0.28, 0.035, 0.22);
  group.add(up, down);

  // a few dummy keys so it reads as a keyboard
  const dummy = new THREE.MeshStandardMaterial({ color: 0x20212a, roughness: 0.8 });
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 9; col++) {
      const k = new THREE.Mesh(new THREE.BoxGeometry(0.19, 0.05, 0.19), dummy);
      k.position.set(-0.85 + col * 0.215, 0.02, -0.42 + row * 0.24);
      group.add(k);
    }
  }
  return { group, up, down };
}

export function Fly3D({ action, mood, className }: { action: FlyAction; mood: FlyMood; className?: string }) {
  const host = useRef<HTMLDivElement | null>(null);
  const state = useRef({ action, mood, pressAt: -10, moodAt: -10 });

  // keep the animation loop reading fresh props without re-creating the scene
  useEffect(() => {
    const s = state.current;
    if (action !== s.action) {
      if (action !== "idle") s.pressAt = performance.now() / 1000;
      s.action = action;
    }
    if (mood !== s.mood) {
      s.moodAt = performance.now() / 1000;
      s.mood = mood;
    }
  }, [action, mood]);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
    camera.position.set(1.55, 1.35, 2.35);
    camera.lookAt(0, 0.3, 0.02);

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      return; // no WebGL: the page simply shows nothing here
    }
    renderer.setClearAlpha(0);
    el.appendChild(renderer.domElement);
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.display = "block";

    scene.add(new THREE.AmbientLight(0xffffff, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    key.position.set(2.5, 4, 3);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xbcd2ff, 1.1);
    fill.position.set(-2, 1.5, 2.5);
    scene.add(fill);
    const rim = new THREE.PointLight(ACCENT, 9, 8);
    rim.position.set(-1.5, 1.2, -1.1);
    scene.add(rim);

    const laptop = makeLaptop();
    scene.add(laptop.group);
    const fly = makeFly();
    fly.root.position.set(0.02, 0.44, 0.18);
    fly.root.scale.setScalar(1.0);
    scene.add(fly.root);

    const resize = () => {
      const w = el.clientWidth || 320;
      const h = el.clientHeight || 220;
      renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      // redraw right away: in a background tab requestAnimationFrame is frozen, and the first layout often
      // happens after this effect ran — without this the panel would stay empty until the tab is focused
      renderer.render(scene, camera);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(el);

    // do not burn a GPU on an off-screen canvas
    let onScreen = true;
    const io = new IntersectionObserver(([e]) => {
      onScreen = e?.isIntersecting ?? true;
    });
    io.observe(el);

    const clock = new THREE.Clock();
    const restY = fly.root.position.y;
    let raf = 0;
    const frame = () => {
      raf = requestAnimationFrame(frame);
      const t = clock.getElapsedTime();
      if (!onScreen || document.hidden) return;
      const now = performance.now() / 1000;
      const s = state.current;
      const sincePress = now - s.pressAt;
      const sinceMood = now - s.moodAt;
      const dancing = s.mood === "win" && sinceMood < 6;
      const crying = s.mood === "lose" && sinceMood < 6;

      // wings: always fluttering, faster while dancing
      const flutter = dancing ? 90 : 55;
      for (const [i, wing] of fly.wings.entries()) {
        const sign = i === 0 ? -1 : 1;
        wing.rotation.z = sign * (crying ? 0.95 : 0.5 + Math.sin(t * flutter) * (crying ? 0.05 : 0.45));
      }

      // press animation: 0 → 1 → 0 over 0.32 s
      const press = sincePress >= 0 && sincePress < 0.32 ? Math.sin((sincePress / 0.32) * Math.PI) : 0;
      const pressing = s.action !== "idle" ? Math.max(press, sincePress < 0.32 ? 0 : 0) : 0;
      laptop.up.position.y = 0.035 - (s.action === "jump" ? pressing * 0.045 : 0);
      laptop.down.position.y = 0.035 - (s.action === "duck" ? pressing * 0.045 : 0);

      // body pose
      const duck = s.action === "duck" ? 1 : 0;
      const hop = dancing ? Math.abs(Math.sin(t * 7)) * 0.14 : 0;
      fly.root.position.y = restY - duck * 0.1 + hop + Math.sin(t * 2) * 0.008 - (crying ? 0.06 : 0);
      fly.root.rotation.y = dancing ? Math.sin(t * 3.4) * 0.5 : Math.sin(t * 0.7) * 0.06;
      fly.root.rotation.z = dancing ? Math.sin(t * 6.8) * 0.12 : 0;
      fly.root.rotation.x = crying ? 0.22 : duck * 0.18;
      fly.head.rotation.x = crying ? 0.5 : -duck * 0.25;

      // legs: the two front legs are the ones on the keys
      for (const [i, leg] of fly.legs.entries()) {
        const phase = t * 3 + i;
        const front = i < 2;
        const onUpKey = i === 1 && s.action === "jump"; // right front leg → ↑
        const onDownKey = i === 0 && s.action === "duck"; // left front leg → ↓
        const stomp = (onUpKey || onDownKey ? pressing : 0) * 0.5;
        leg.group.rotation.x = (front ? -0.25 : 0.2) + Math.sin(phase) * (dancing ? 0.35 : 0.05) + stomp;
        leg.knee.rotation.x = 0.7 + stomp * 0.6 + duck * 0.35 + (dancing ? Math.cos(phase) * 0.2 : 0);
        if (dancing && !front) leg.group.rotation.z = Math.sin(phase * 1.3) * 0.3;
        else leg.group.rotation.z = 0;
      }

      // tears
      for (const [i, drop] of fly.tears.entries()) {
        drop.visible = crying;
        if (!crying) continue;
        const p = ((t * 0.9 + i * 0.25) % 1);
        drop.position.y = 0.02 - p * 0.55;
        drop.position.z = 0.3 - p * 0.05;
        (drop.material as THREE.MeshStandardMaterial).opacity = 0.9 * (1 - p);
      }

      rim.intensity = 6 + (s.action === "jump" ? pressing * 14 : 0);
      renderer.render(scene, camera);
    };
    renderer.render(scene, camera); // one static frame, so the fly is there even where rAF is throttled
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      renderer.dispose();
      scene.traverse((o) => {
        if (o instanceof THREE.Mesh) {
          o.geometry.dispose();
          for (const m of Array.isArray(o.material) ? o.material : [o.material]) m.dispose();
        }
      });
      el.removeChild(renderer.domElement);
    };
  }, []);

  return <div ref={host} className={className ?? "fly3d"} aria-hidden="true" />;
}
