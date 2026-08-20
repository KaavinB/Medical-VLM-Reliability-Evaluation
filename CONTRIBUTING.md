# Contributing

This is a course research repository (COMP 646, Rice University). Small fixes to docs, metrics, or notebook usability are welcome.

## Reproduce the experiment

1. Use a GPU with enough memory for a 7–8B VLM in 4-bit (A100 is what the report used).
2. Follow the [Quick start](README.md#quick-start) in the README.
3. Do **not** commit the Kaggle chest X-ray images, LoRA adapters, or `outputs/`. Those paths are in `.gitignore` on purpose.

## Notebook notes

- All experiment knobs live in the configuration cell near the top (`MAX_SAMPLES`, `RUN_LORA_FINETUNING`, model names, output directory).
- Official reported numbers are in [`docs/report.pdf`](docs/report.pdf) and the README table. A fresh run can differ.
- Keep the clinical disclaimer in the notebook title cell if you edit that section.

## Pull requests

Please keep PRs focused (docs, a metric bug, a plot fix) and say what you ran. If you change a metric definition, update both the notebook and the README description so they stay aligned.
