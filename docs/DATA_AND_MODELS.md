# Data & Models

Trained models and full datasets are intentionally not stored in Git. They are generated locally so the repository stays lightweight.

## Regenerate data

From the repository root:

```bash
python download_data.py
```

This writes task-specific `data/train.csv`, `data/test.csv`, and sample files. Small `sample_*.csv` files remain in Git so each task can be explored without a download.

## Regenerate models

Enter a task folder and run its pipeline, for example:

```bash
cd Task_2_News_Classification
python src/main.py
python src/generate_results.py
```

Each `main.py` writes artifacts to its local `models/` directory. `generate_results.py` reads those artifacts and writes `data/results.csv`.

## Notes

- `*.pkl`, `models/`, `data/train.csv`, `data/test.csv`, and `data/results.csv` are ignored by Git.
- Binary model files should be published as GitHub Releases if they need to be shared.
- Pickle files are not safe to load from untrusted sources; regenerate them locally instead.
