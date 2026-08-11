# Figures to upload to Overleaf

Create a folder named `figures` in the Overleaf project and upload these ten
PDF files without changing their names. `main.tex` is the compact paper;
`supplementary_material.tex` compiles the detailed secondary analyses.

| File | Paper content |
| --- | --- |
| `fig01_frequency_sweep.pdf` | V0 sinusoidal activation sweep |
| `fig02_easy_vs_hard.pdf` | Easy versus hard synthetic behavior |
| `fig03_filter_evolution.pdf` | Coarse, fine, and free-filter evolution |
| `fig04_gate_evolution.pdf` | Independent, contextual, and sparse gates |
| `fig05_fsdd_splits.pdf` | Random versus speaker-disjoint FSDD |
| `fig06_fsdd_temporal.pdf` | Pooled EGFN, Temporal EGFN, and Conv1D |
| `fig07_speech_commands.pdf` | Speech Commands comparison |
| `fig08_mimii_initial.pdf` | Initial MIMII valve application |
| `fig09_mimii_multiseed.pdf` | Three-seed MIMII experiment |
| `fig10_mimii_v2.pdf` | Capacity-matched controlled V2 study |

Expected Overleaf structure:

```text
main.tex
supplementary_material.tex
figures/
  fig01_frequency_sweep.pdf
  fig02_easy_vs_hard.pdf
  fig03_filter_evolution.pdf
  fig04_gate_evolution.pdf
  fig05_fsdd_splits.pdf
  fig06_fsdd_temporal.pdf
  fig07_speech_commands.pdf
  fig08_mimii_initial.pdf
  fig09_mimii_multiseed.pdf
  fig10_mimii_v2.pdf
```

Use pdfLaTeX as the compiler. References are embedded in `main.tex`, so no
separate `.bib` file is required.
