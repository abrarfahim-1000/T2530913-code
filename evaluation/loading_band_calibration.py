"""
P(contingency is a violation | base-case loading band) -- the WARN channel's
calibration source, and the evidence that 100 is not an arbitrary threshold.

findings 13.4 measures a single number: P(violation | base case already
overloaded) = 91-96%, which establishes that the served thermal rule is not a
label restatement. This extends that to the full curve, and the curve is what
decides whether lower-threshold thermal rules (the corpus carries 84/90/95/110/
116/125%) can be admitted.

The answer is a cliff, not a slope. Immediately below 1.00 the predicate is
roughly a coin flip; above it, it is right 92-97% of the time. That is why the
corpus converged on 100 and why admitting `loading_pct > 95` to the BLOCK
channel would trade the headline intervention-precision result for volume.

Every WARN rule needs a row like this attached before it is shown to anyone: a
warning without a measured rate is an alarm, a warning with one is evidence.

Run:  .venv\\Scripts\\python.exe evaluation\\loading_band_calibration.py
      .venv\\Scripts\\python.exe evaluation\\loading_band_calibration.py --max-frames 4000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TAGS = ("neurips2020", "case14", "wcci2022")
BANDS: list[tuple[float, float]] = [
    (0.00, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90),
    (0.90, 0.95), (0.95, 1.00), (1.00, float("inf")),
]


def band_label(lo: float, hi: float) -> str:
    return f"{lo:.2f}-{hi:.2f}" if hi != float("inf") else ">= 1.00"


def scan(tag: str, data_dir: Path, max_frames: int | None) -> dict:
    path = data_dir / f"grid_dataset_{tag}_n1.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    tot = {b: 0 for b in BANDS}
    vio = {b: 0 for b in BANDS}
    frames = {b: 0 for b in BANDS}
    n_frames = 0

    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_frames is not None and i >= max_frames:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rho = np.asarray(rec.get("rho", []), dtype=float)
            y = np.asarray(rec.get("n1_violation", []), dtype=int)
            if y.size == 0:
                continue
            mask = y >= 0                      # -1 == not evaluated, never a negative
            if not mask.any():
                continue
            n_frames += 1
            rmax = float(np.nanmax(rho)) if rho.size else 0.0
            for b in BANDS:
                if b[0] <= rmax < b[1]:
                    tot[b] += int(mask.sum())
                    vio[b] += int((y[mask] == 1).sum())
                    frames[b] += 1
                    break

    return {
        "n_frames": n_frames,
        "bands": {
            band_label(*b): {
                "frames": frames[b],
                "contingencies": tot[b],
                "violations": vio[b],
                "p_violation": (vio[b] / tot[b]) if tot[b] else None,
            }
            for b in BANDS
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap frames per grid (default: every frame). "
                         "Small caps make the 0.95-1.00 band noisy.")
    ap.add_argument("--json", default="results/audit/loading_band_calibration.json")
    args = ap.parse_args()

    results = {}
    for tag in TAGS:
        try:
            results[tag] = scan(tag, Path(args.data_dir), args.max_frames)
        except FileNotFoundError as exc:
            print(f"  [{tag}] dataset missing, skipping: {exc}")

    if not results:
        print("no dataset found")
        return 1

    print("\n" + "=" * 78)
    print("P(contingency is a violation | base-case rho_max band)")
    print("=" * 78)
    header = f"  {'base rho_max':<16}" + "".join(f"{t:>19}" for t in results)
    print("\n" + header)
    print("  " + "-" * (16 + 19 * len(results)))
    for b in BANDS:
        lab = band_label(*b)
        cells = ""
        for tag in results:
            d = results[tag]["bands"][lab]
            cells += (f"{d['p_violation']:>12.3f} (n={d['frames']:>4})"
                      if d["p_violation"] is not None else f"{'-':>19}")
        mark = "   <-- the served rule" if b[0] == 1.00 else ""
        print(f"  {lab:<16}{cells}{mark}")

    print("\n  Read the cliff, not the slope: immediately below 1.00 the predicate is")
    print("  near a coin flip; at and above it, 92-97%. The served threshold sits on a")
    print("  physical discontinuity, which is why lower-threshold thermal rules belong")
    print("  in WARN and not in BLOCK.")

    out = {
        "bands": [band_label(*b) for b in BANDS],
        "max_frames": args.max_frames,
        "grids": results,
        "note": ("Calibration source for the WARN channel (findings 19.3) and the "
                 "full-curve extension of findings 13.4. -1 labels are excluded as "
                 "missing, never counted as secure."),
    }
    dest = Path(args.json)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
