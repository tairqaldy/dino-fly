"""Phase 4, connectome only (CPU, seconds): which MBONs can the visual context reach, and which dopamine reaches them?

For every MBON cell type: predicted transmitter (→ valence class used by H2), synapses received from the 388 visual
Kenyon cells that carry the context, from all Kenyon cells, from the PAM cluster (reward transducer) and from the PPL1
cluster (punishment transducer). Read next to the MBON → GF influence map, this says what either learning hypothesis
can possibly change.

    uv run --no-sync python -u -m experiments.mb_wiring
"""

from __future__ import annotations

import numpy as np

from experiments.common import load_result, write_result
from flybrain.transducer.context import visual_kenyon_cells

NAME = "mb_wiring"
VALENCE = {"glutamate": "avoidance", "gaba": "approach", "acetylcholine": "approach"}


def main() -> int:
    import pandas as pd

    from flybrain import data_manifest, neurons
    from flybrain.connectome import load_connectome

    conn = load_connectome("flywire783")
    ix = {n: neurons.indices(conn, n) for n in ("LC4", "LPLC2", "KC", "MBON", "PAM", "PPL1")}
    ann = pd.read_csv(data_manifest.ensure("flywire_annotations", log=lambda _m: None), sep="\t", dtype={"root_id": "Int64"},
                      usecols=["root_id", "super_class", "cell_type", "top_nt"], low_memory=False)
    vpn_ids = ann.loc[ann["super_class"] == "visual_projection", "root_id"].astype("int64").to_numpy()
    vpn = np.setdiff1d(conn.index_of(vpn_ids[np.isin(vpn_ids, conn.root_ids)]), np.concatenate([ix["LC4"], ix["LPLC2"]]))
    visual_kc = visual_kenyon_cells(conn, vpn, ix["KC"], 0.05)
    by_id = ann.set_index(ann["root_id"].astype("int64"))
    mbon_type = np.array([str(by_id.at[int(conn.root_ids[i]), "cell_type"]) for i in ix["MBON"]])
    mbon_nt = np.array([str(by_id.at[int(conn.root_ids[i]), "top_nt"]) for i in ix["MBON"]])

    def synapses(pre: np.ndarray, post: np.ndarray) -> int:
        return int(np.abs(conn.weight[conn.edge_mask(pre_idx=pre, post_idx=post)]).sum())

    influence = {v["cell_type"]: v["delta_p_spike"] for v in (load_result("mbon_influence") or {"mbon_types": []})["mbon_types"]}
    band = (load_result("mbon_influence") or {}).get("noise_band_delta_p")
    rows = []
    for t in sorted(set(mbon_type)):
        members = ix["MBON"][mbon_type == t]
        nts = sorted(set(mbon_nt[mbon_type == t]))
        rows.append({"cell_type": t, "n_neurons": len(members), "top_nt": "/".join(nts),
                     "valence_class": "/".join(sorted({VALENCE.get(n, "none") for n in nts})),
                     "synapses_from_visual_kcs": synapses(visual_kc, members), "synapses_from_all_kcs": synapses(ix["KC"], members),
                     "synapses_from_pam": synapses(ix["PAM"], members), "synapses_from_ppl1": synapses(ix["PPL1"], members),
                     "gf_influence_delta_p": influence.get(t)})
    rows.sort(key=lambda r: -r["synapses_from_visual_kcs"])
    total = sum(r["synapses_from_visual_kcs"] for r in rows)
    write_result(NAME, {"protocol": {"n_visual_kcs": len(visual_kc), "n_kcs": len(ix["KC"]), "gf_influence_noise_band": band},
                        "visual_kc_to_mbon_synapses": total, "rows": rows})
    for r in rows[:12]:
        print(r)
    return 0


def render_report(r: dict) -> str:
    band = r["protocol"]["gf_influence_noise_band"]
    lines = [
        f"The {r['protocol']['n_visual_kcs']} visual Kenyon cells make {r['visual_kc_to_mbon_synapses']:,} synapses onto MBONs. The twelve MBON types that "
        "receive most of them:",
        "",
        "| MBON type | neurons | predicted transmitter → H2 class | synapses from visual KCs | share | from all KCs | from PAM | from PPL1 | ΔP(GF spike) when driven |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in r["rows"][:12]:
        d = row["gf_influence_delta_p"]
        mark = "" if d is None or band is None or abs(d) <= band else " **(outside noise band)**"
        lines.append(f"| {row['cell_type']} | {row['n_neurons']} | {row['top_nt']} → {row['valence_class']} | {row['synapses_from_visual_kcs']:,} | "
                     f"{row['synapses_from_visual_kcs'] / max(1, r['visual_kc_to_mbon_synapses']):.0%} | {row['synapses_from_all_kcs']:,} | "
                     f"{row['synapses_from_pam']:,} | {row['synapses_from_ppl1']:,} | {'—' if d is None else f'{d:+.2f}'}{mark} |")
    strong = [row for row in r["rows"] if row["gf_influence_delta_p"] is not None and band is not None and abs(row["gf_influence_delta_p"]) > band]
    share = sum(row["synapses_from_visual_kcs"] for row in strong) / max(1, r["visual_kc_to_mbon_synapses"])
    lines += ["", f"MBON types that move the Giant Fiber outside the noise band of the influence map receive **{share:.0%}** of the visual-KC → MBON "
                  f"synapses ({', '.join(f'{row["cell_type"]} {row["synapses_from_visual_kcs"]:,}' for row in sorted(strong, key=lambda x: -x['synapses_from_visual_kcs'])[:5])}, …).",
              "", "Source: `results/mb_wiring.json` (connectome only; ΔP from `results/mbon_influence.json`)."]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
