# Datasets

Drop CSV files here. They are detected automatically by column names.

| Purpose | Required columns | Example |
|---|---|---|
| Resume categorization (training) | `Category`, `Resume` | `UpdatedResumeDataSet.csv` |
| Job recommendations | a title column (`title`, `job title`, `position`) and a description column (`description`, `job description`); optional `company`, `location`, `skills` | `jobs.csv` |

After adding a resume dataset, delete `model.joblib` (or run `python train_model.py`) to retrain.
After adding a jobs dataset, delete `screening.db` so jobs are re-imported (this also removes test accounts).
