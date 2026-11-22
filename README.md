# AI-Powered IT Support Ticket Routing

Neural-network text classifiers that read an internal support request and predict which of 8 support groups should handle it.

## Quick start (run the demo on your own machine)
Requires Python 3.11 or 3.12 and git. The trained models and data are included, so nothing needs retraining.

```bash
git clone https://github.com/Dark102102/servicenow-ticket-classifier.git
cd servicenow-ticket-classifier
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app/app.py                  # then open http://127.0.0.1:7860
```
The ServiceNow button is optional and needs your own `.env` (see "ServiceNow integration" below).

## Project status
| Stage | Notebook | Status |
|---|---|---|
| 1. Data exploration | `notebooks/01_data_exploration.ipynb` | ✅ done |
| 2. Cleaning & splits | `notebooks/02_cleaning_and_splits.ipynb` | ✅ done |
| 3. Baseline (TF-IDF + Logistic Regression) | `notebooks/03_baseline.ipynb` | ✅ test acc 86.1%, macro-F1 0.862 |
| 4a. ANN (multi-layer perceptron) | `notebooks/04a_ann.ipynb` | ✅ test acc 84.7%, macro-F1 0.846 |
| 4b. BiLSTM | `notebooks/04b_bilstm.ipynb` | ✅ test acc 83.2%, macro-F1 0.831 |
| 4c. DistilBERT | `notebooks/04c_distilbert_colab.ipynb` (train on Colab GPU) → `notebooks/04c_distilbert_results.ipynb` | ⏳ waiting for Colab run |
| 5. Evaluation & error analysis | `notebooks/05_evaluation.ipynb` | ✅ (re-run after DistilBERT to include it) |
| 6. Demo app | `app/app.py` | ✅ |

## Repository layout
```
data/raw/         original Kaggle CSV (unchanged)
data/processed/   train / validation / test splits
src/              shared code (paths, plotting style, cleaning, models)
notebooks/        one notebook per stage – run top to bottom
figures/          every chart used on the website (PNG)
results/          metrics tables and summaries
docs/             data card and plain-language notes for the website
app/              demo application
```

## Data
See [`docs/data_card.md`](docs/data_card.md). Technical model descriptions: [`docs/model_details.md`](docs/model_details.md). 47,837 tickets, 8 support groups, text already pre-processed by the dataset author.

## Reproduce
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cd notebooks
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_data_exploration.ipynb
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 02_cleaning_and_splits.ipynb
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 03_baseline.ipynb   # ~3 min
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 04a_ann.ipynb       # ~9 min on Apple M2 CPU
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=10800 04b_bilstm.ipynb   # ~16 min
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 05_evaluation.ipynb   # ~1 min
```

## DistilBERT on Google Colab (GPU)
DistilBERT is too slow to fine-tune on a laptop CPU, so it is trained on a free Colab GPU:

1. Go to https://colab.research.google.com → **File → Upload notebook** → `notebooks/04c_distilbert_colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. **Runtime → Run all**. When the upload button appears, select `data/processed/train.csv`, `val.csv`, `test.csv`.
4. After ~15–30 min `distilbert_outputs.zip` downloads. Unzip it into `results/distilbert/`
   (it contains `preds_val.csv`, `preds_test.csv`, `run_info.json`).
5. Run locally:
   `cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 04c_distilbert_results.ipynb`
6. *(Optional, demo app)* in Colab set `DOWNLOAD_MODEL = True` in the last cell, run it, and unzip `distilbert_model.zip` into `models/distilbert/`.

## Demo app
```bash
.venv/bin/python app/app.py
```
Open http://127.0.0.1:7860, type or pick a ticket, choose a model. The app shows the predicted support group,
the top-3 probabilities, an **Auto-route / Needs human review** decision (threshold chosen on validation tickets so that
auto-routed tickets are ≥ 95% correct: baseline 0.68, ANN 0.82, BiLSTM 0.92), and for the baseline the words that drove the prediction.
DistilBERT appears in the model list automatically once its fine-tuned model is unzipped into `models/distilbert/`.

## ServiceNow integration (optional)
The demo can create a real incident in a free ServiceNow Personal Developer Instance (PDI), already assigned to the
predicted support group (or to **Service Desk Triage** when the model is not confident), with the model's reasoning in the work notes.

1. Request a PDI at https://developer.servicenow.com ("Request Instance").
2. `cp .env.example .env` and fill in `SN_INSTANCE`, `SN_USER`, `SN_PASSWORD` (`.env` is git-ignored).
3. One-time setup (checks the login, creates the 9 assignment groups): `.venv/bin/python scripts/servicenow_setup.py`
4. Start the demo and click **Route + create ServiceNow incident**.

PDIs hibernate when idle (log in via the browser to wake them) and are reclaimed after ~10 days without activity.
