from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = (
    "outputs/mimii_hierarchical_gating_ablation_v7/gating_ablation_summary.json",
    "outputs/mimii_representation_ablation_v7/representation_ablation_summary.json",
    "outputs/mimii_spectral_baselines_v5/results.json",
    "outputs/mimii_corrected_v5/egfn_seed42/results.json",
)


def build_evidence(source_paths: list[str]) -> dict:
    items = []
    for index, relative_path in enumerate(source_paths, start=1):
        path = ROOT / relative_path
        if not path.is_file():
            continue
        items.append(
            {
                "evidence_id": f"egfn-{index:03d}",
                "source_path": relative_path,
                "title": path.stem,
                "payload": json.loads(path.read_text(encoding="utf-8")),
            }
        )
    return {
        "schema_version": "egfn-evidence-v1",
        "project": "frequency_gated_nn",
        "items": items,
        "references": [
            {
                "reference_id": "ref-experiment-log",
                "title": "EGFN experiment log",
                "location": "reports/experiment_log.md",
                "note": "Human-readable experiment history; JSON artifacts remain the metric source of truth.",
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact, cited EGFN evidence JSON.")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence" / "egfn_evidence.json")
    parser.add_argument("--source", action="append", default=[])
    args = parser.parse_args()
    sources = args.source or list(DEFAULT_SOURCES)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = build_evidence(sources)
    output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output} with {len(evidence['items'])} evidence items.")


if __name__ == "__main__":
    main()
