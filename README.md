# SSMBERT

BERT-based models for software size measurement.

> **Note:** The industrial datasets for Case 1 and Case 2 are not publicly available due to intellectual property restrictions.

> **Availability:** Models trained on the heterogeneous dataset are publicly available at [https://huggingface.co/smtnkc/SSMBERT](https://huggingface.co/smtnkc/SSMBERT/tree/main).

## Data Shuffling

Run the following commands to shuffle each dataset:

```bash
python shuffle_data.py --dataset heterounique
python shuffle_data.py --dataset case1
python shuffle_data.py --dataset case2
```

This script reads:

* `data/heterounique.csv`
* `data/case1.csv`
* `data/case2.csv`

and produces:

* `data/heterounique-shuffled-data.csv`
* `data/case1-shuffled-data.csv`
* `data/case2-shuffled-data.csv`

## Evaluation Protocol

Two evaluation modes are supported:

* **Generic evaluation**
  Training and testing both use the heterogeneous dataset.

* **Internal evaluation**
  Training and testing both use an organization‑ or project‑specific dataset.

Both modes employ 5‑fold cross validation with the `bert` and `se-bert` models.

### Example

```bash
python cross_val.py --dataset heterounique --target entry --model bert
```

This command will:

1. Load `data/heterounique-shuffled-data.csv` and perform 5‑fold cross validation (5 epochs per fold, 25 epochs total).
2. For each fold, select the epoch with lowest MSE and generate predictions.
3. Concatenate all fold predictions into
   `data/heterounique-shuffled-data-predby-bert-heterounique.csv`.
4. Save the best model as
   `checkpoints/bert-heterounique-entry.pt`.
5. Log metrics to
   `logs/heterounique-entry-predby-bert-heterounique.csv`.

To run every combination of model, dataset, and target, use:

```bash
./run_cross_val.sh
```

This executes 2 models (`bert`, `se-bert`) × 4 datasets (`heterounique`, `heterogrouped`, `case1`, `case2`) × 7 targets (`entry`, `read`, `write`, `exit`, `interaction`, `communication`, `process`) for 56 total runs.

## External Evaluation

Use a model trained on the heterogeneous dataset to predict on an organization‑specific set. The full dataset then serves as the test set. Available models: `bert-heterounique` or `se-bert-heterounique`.

### Example

```bash
python pred.py --dataset case1 --target entry --model bert-heterounique
```

This command will:

1. Load `data/case1-shuffled-data.csv` and `checkpoints/bert-heterounique-entry.pt`.
2. Generate predictions and concatenate them into
   `data/case1-shuffled-data-predby-bert-heterounique.csv`.
3. Log metrics to
   `logs/case1-entry-predby-bert-heterounique.csv`.

To run every combination of model, dataset, and target, use:

```bash
./run_pred.sh
```

This executes 2 models (`bert-heterounique`, `se-bert-heterounique`) ×
 2 datasets (`case1`, `case2`) × 7 targets (`entry`, `read`, `write`, `exit`, `interaction`, `communication`, `process`) for 28 total runs.

## Postprocessing

### Compute Total CFP and MicroM

```bash
python sum_preds.py
```

For each CSV under `preds/`, this script:

* Adds a `cfp_pred` column:

  ```text
  cfp_pred = entry_pred + read_pred + write_pred + exit_pred
  ```
* Adds a `microm_pred` column:

  ```text
  microm_pred = interaction_pred + communication_pred + process_pred
  ```
* Reorders columns for consistency.

### Calculate Metrics

```bash
python calculate_metrics.py
```

Processes all prediction files in `preds/` and outputs these CSVs in `stats/`:

1. `mse.csv` – Mean Squared Error
2. `mae.csv` – Mean Absolute Error
3. `nmae.csv` – Normalized MAE (MAE / mean actual)
4. `acc.csv` – Exact Match Accuracy (after rounding)
5. `mmre.csv` – Mean Magnitude of Relative Error
6. `pred30.csv` – Percentage of predictions with MRE ≤ 0.30
7. `nonzero.csv` – Percentage of non-zero actual values

Each file reports metrics for all features (`entry`, `read`, `write`, `exit`, `cfp`, `interaction`, `communication`, `process`, `microm`) across:

* Model (`bert`, `se-bert`)
* Training set (`heterounique`, `case1`, `case2`)
* Test set (`heterounique`, `case1`, `case2`)

All metric values are rounded to four decimal places.
