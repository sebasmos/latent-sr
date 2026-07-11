#!/usr/bin/env bash
# Regenerate the three hallucination figures used by the paper and install them
# under the filenames supplementary-tracked.tex / supplementary.tex expect.
#
#   Supplementary Figure 2 -> figures/fig4a_mrnet_hallucination.png
#   Supplementary Figure 3 -> figures/fig4b_brats_hallucination.png
#   Supplementary Figure 4 -> figures/fig4c_cxr_hallucination.png
#
# Each is the `summary_bar_chart.png` emitted by eval_hallucination_quantification.py
# for that dataset. The copy step used to be manual and undocumented, which is how the
# panel annotation drifted out of sync with the paper's mu/sigma notation.
#
# Deterministic: no RNG, no GPU. Reads SR/AE/HR PNGs and recomputes the pixel statistics,
# so it also serves as a reproducibility check -- the aggregate numbers in results.json
# must not change.
#
# Usage:   bash scripts/make_hallucination_figures.sh [--check]
#   --check   regenerate into a temp dir and diff the aggregate metrics against the
#             committed results.json instead of overwriting the paper figures.
#
# Cost: reads (120 + 700 + 1000) x 4 PNGs ~= 7,300 images. Minutes, not hours. No GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
FIGDIR="${ROOT}/figures"
CHECK=0
[[ "${1:-}" == "--check" ]] && CHECK=1

export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mplconfig-$$}"
mkdir -p "$MPLCONFIGDIR"

declare -A TARGET=( [mrnet]=fig4a_mrnet_hallucination.png
                    [brats]=fig4b_brats_hallucination.png
                    [cxr]=fig4c_cxr_hallucination.png )

for ds in mrnet brats cxr; do
  echo "=== ${ds} ==="
  before="${ROOT}/outputs/experiments/hallucination_${ds}/results.json"
  saved=""
  if [[ -f "$before" ]]; then saved="$(mktemp)"; cp "$before" "$saved"; fi

  "$PYTHON" "${ROOT}/scripts/eval_hallucination_quantification.py" --dataset "$ds"

  if [[ -n "$saved" ]]; then
    "$PYTHON" - "$saved" "$before" <<'PY'
import json, sys
def flat(d, p=""):
    o = {}
    if isinstance(d, dict):
        for k, v in d.items():
            o.update(flat(v, f"{p}/{k}"))
    elif isinstance(d, (int, float)):
        o[p] = d
    return o
old, new = (flat(json.load(open(f))) for f in sys.argv[1:3])
keys = [k for k in sorted(set(old) | set(new)) if "per_image" not in k]
diff = [k for k in keys if old.get(k) != new.get(k)]
if diff:
    print(f"  !! {len(diff)} aggregate metric(s) CHANGED: {diff[:5]}")
    sys.exit(1)
print(f"  reproducibility: {len(keys)}/{len(keys)} aggregate metrics identical")
PY
    rm -f "$saved"
  fi

  src="${ROOT}/outputs/experiments/hallucination_${ds}/summary_bar_chart.png"
  dst="${FIGDIR}/${TARGET[$ds]}"
  if (( CHECK )); then
    if cmp -s "$src" "$dst"; then echo "  figure up to date"; else echo "  !! figure STALE: $dst"; fi
  else
    cp "$src" "$dst"
    echo "  installed -> ${TARGET[$ds]}"
  fi
done

echo
echo "Done. Figures installed under ${FIGDIR}."
