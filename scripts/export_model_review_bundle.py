from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_GROUPS: dict[str, tuple[str, ...]] = {
    "Architecture": (
        "src/blocks/hierarchical_spectral_frontend.py",
        "src/models/hierarchical_anomaly_detector.py",
        "src/models/event_pooling.py",
        "src/models/conv1d_anomaly_encoder.py",
        "src/blocks/__init__.py",
        "src/models/__init__.py",
    ),
    "Anomaly objective, profiles, and scoring": (
        "src/anomaly/feature_memory.py",
        "src/anomaly/one_class.py",
        "src/anomaly/normal_profile.py",
        "src/anomaly/scoring.py",
        "src/anomaly/gate_regularization.py",
        "src/anomaly/spectral_baseline.py",
        "src/anomaly/__init__.py",
    ),
    "Data protocol": (
        "src/data/anomaly_protocol.py",
        "src/data/anomaly_window_dataset.py",
        "src/data/mimii_dataset.py",
        "src/data/__init__.py",
    ),
    "Training and ablation entry points": (
        "scripts/train_mimii_one_class.py",
        "scripts/run_mimii_hierarchical_gating_ablation.py",
        "scripts/run_mimii_representation_ablation.py",
        "scripts/evaluate_mimii_spectral_baselines.py",
    ),
    "Evaluation utilities": (
        "src/utils/binary_evaluation.py",
        "requirements.txt",
    ),
}

TEST_FILES: tuple[str, ...] = (
    "tests/test_hierarchical_spectral_frontend.py",
    "tests/test_hierarchical_anomaly_detector.py",
    "tests/test_feature_memory.py",
    "tests/test_one_class_training_components.py",
    "tests/test_anomaly_protocol.py",
    "tests/test_anomaly_scoring.py",
    "tests/test_gate_regularization.py",
    "tests/test_spectral_baseline.py",
    "tests/test_hierarchical_gating_ablation.py",
)

DEFAULT_RESULT_FILES: tuple[str, ...] = (
    "outputs/mimii_hierarchical_gating_ablation_v7/gating_ablation_summary.json",
    "outputs/mimii_representation_ablation_v7/representation_ablation_summary.json",
    "outputs/mimii_hierarchical_gating_ablation_v6/summary.json",
    "outputs/mimii_hierarchical_gating_ablation_v6/runs/none/seed_42/results.json",
    "outputs/mimii_spectral_baselines_v5/results.json",
    "outputs/mimii_corrected_v5/egfn_seed42/results.json",
)

REVIEW_BRIEF = """You are reviewing an experimental one-class acoustic anomaly detector.
Treat the included working-tree files as the source of truth. Do not infer behavior
from class names or comments when it conflicts with executable code.

Audit the system end to end:

1. Trace tensor shapes and gradients from waveform to STFT, macro gates, subgates,
   activation signature, embedding/profile representation, memory, and final score.
2. Check whether the training objective optimizes information used by the primary
   anomaly score, and identify train/inference or fit/score mismatches.
3. Check data isolation: normal-only fitting, train/validation/test separation,
   window sampling, condition batching, calibration, thresholds, and leakage.
4. Check the causal validity of the none/macro/subband/hierarchical gating ablation:
   matched capacity, inactive trainable parameters, conditional inputs, and whether
   each mode changes only the intended factor.
5. Check numerical stability, normalization axes, distance calculations, fallback
   behavior for unknown conditions, and aggregation across windows and machine IDs.
6. Separate architecture-general logic from MIMII-specific adapter assumptions.
7. Compare tests with claimed behavior and list important missing tests.

Report findings as Critical, Major, Moderate, or Minor. For every finding cite the
file and line, explain the failure mechanism, its likely metric impact, and the
smallest defensible correction. Clearly distinguish confirmed defects from
hypotheses requiring an experiment. End with a short execution-flow diagram and a
prioritized validation plan. Do not propose a wholesale redesign before checking
the current logic.
"""


def _run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        return f"unavailable: {completed.stderr.strip()}"
    return completed.stdout.strip() or "(clean/empty)"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _language(path: Path) -> str:
    return {
        ".py": "python",
        ".json": "json",
        ".toml": "toml",
        ".md": "markdown",
        ".txt": "text",
    }.get(path.suffix.lower(), "text")


def _numbered_text(text: str) -> str:
    lines = text.splitlines()
    width = max(4, len(str(len(lines))))
    return "\n".join(f"{index:0{width}d} | {line}" for index, line in enumerate(lines, 1))


def _render_file(relative_path: str, max_file_bytes: int) -> tuple[str, str | None]:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        return f"## `{relative_path}`\n\nMissing from the current working tree.\n", relative_path

    content = path.read_bytes()
    digest = _sha256(content)
    if len(content) > max_file_bytes:
        return (
            f"## `{relative_path}`\n\n"
            f"Omitted: {len(content):,} bytes exceeds --max-file-bytes "
            f"({max_file_bytes:,}). SHA-256: `{digest}`.\n",
            None,
        )

    text = content.decode("utf-8", errors="replace")
    return (
        f"## `{relative_path}`\n\n"
        f"Bytes: {len(content):,}  \nSHA-256: `{digest}`\n\n"
        f"```{_language(path)}\n{_numbered_text(text)}\n```\n",
        None,
    )


def build_bundle(
    output_path: Path,
    include_tests: bool,
    include_results: bool,
    extra_results: list[str],
    max_file_bytes: int,
) -> list[str]:
    generated_at = datetime.now(timezone.utc).isoformat()
    branch = _run_git("branch", "--show-current")
    commit = _run_git("rev-parse", "HEAD")
    status = _run_git("status", "--short")
    diff_stat = _run_git("diff", "--stat")

    sections = [
        "# EGFN current-model external review bundle\n",
        f"Generated (UTC): `{generated_at}`  \n"
        f"Branch: `{branch}`  \nCommit: `{commit}`\n",
        "This is a static snapshot of the current working tree, not only the last "
        "commit. Raw audio, checkpoints, environments, and secrets are intentionally "
        "excluded. MIMII appears only as the active dataset adapter; it is not assumed "
        "to define the architecture's intended scope.\n",
        "# Instructions for the external reviewer\n",
        REVIEW_BRIEF,
        "# Repository state\n",
        "## `git status --short`\n\n```text\n"
        f"{status}\n```\n\n## `git diff --stat`\n\n```text\n{diff_stat}\n```\n",
        "# Included implementation\n",
    ]

    missing: list[str] = []
    for group, paths in SOURCE_GROUPS.items():
        sections.append(f"# {group}\n")
        for relative_path in paths:
            rendered, missing_path = _render_file(relative_path, max_file_bytes)
            sections.append(rendered)
            if missing_path:
                missing.append(missing_path)

    if include_tests:
        sections.append("# Relevant tests\n")
        for relative_path in TEST_FILES:
            rendered, missing_path = _render_file(relative_path, max_file_bytes)
            sections.append(rendered)
            if missing_path:
                missing.append(missing_path)

    if include_results:
        sections.append("# Available experiment evidence\n")
        result_paths = list(dict.fromkeys([*DEFAULT_RESULT_FILES, *extra_results]))
        found_result = False
        for relative_path in result_paths:
            if not (PROJECT_ROOT / relative_path).is_file():
                continue
            found_result = True
            rendered, _ = _render_file(relative_path, max_file_bytes)
            sections.append(rendered)
        if not found_result:
            sections.append("No requested result JSON files were present.\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8", newline="\n")
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the current EGFN implementation into one reviewable Markdown file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "exports" / "current_model_review_bundle.md",
    )
    parser.add_argument("--no-tests", action="store_true", help="Exclude test source files.")
    parser.add_argument(
        "--no-results",
        action="store_true",
        help="Exclude locally available experiment result JSON files.",
    )
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        help="Additional result file relative to the repository root; repeat as needed.",
    )
    parser.add_argument("--max-file-bytes", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_file_bytes <= 0:
        raise ValueError("--max-file-bytes must be positive.")

    output = args.output
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    missing = build_bundle(
        output_path=output,
        include_tests=not args.no_tests,
        include_results=not args.no_results,
        extra_results=args.result,
        max_file_bytes=args.max_file_bytes,
    )
    print(f"saved={output}")
    print(f"bytes={output.stat().st_size}")
    if missing:
        print("missing=" + ",".join(missing))


if __name__ == "__main__":
    main()
