# Trained Model Weights

The Git repository intentionally does not contain training checkpoints.

## Why the web UI does not need weights

`system_ui/` is a deterministic replay viewer. It reads exported held-out MuJoCo trajectories from `system_ui/data/`; it does not import PyTorch or execute PPO in the browser. Consequently, cloning the repository is sufficient to run the website.

## When weights are needed

Weights are required only to:

- run the learned PPO policy in MuJoCo;
- regenerate the static rollout dataset;
- perform new evaluation conditions without retraining.

Training and evaluation scripts create checkpoints under `runs/`, which is excluded by `.gitignore`.

## Recommended publication channel

Do not commit `.pt`, `.pth`, or `.ckpt` files to Git history. For an archival release, publish only the final V12 checkpoints for training seeds 0, 1, and 2 through a versioned GitHub Release or Zenodo record, together with SHA-256 checksums and the corresponding `config.json` files. Link that artifact from this document and the repository README.
