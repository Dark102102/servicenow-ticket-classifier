# Data Card – IT Service Ticket Dataset

| Field | Value |
|---|---|
| Name | IT Service Ticket Classification Dataset |
| Source | Kaggle (public) – **TODO: add dataset URL and author** |
| License | **TODO: confirm on the Kaggle page before publishing a data link** |
| File | `data/raw/all_tickets_processed_improved_v3.csv` |
| SHA-256 | `044fdace33fa564e1e60453f2941dafc95539c99878b0d32746950394b9dd4d4` |
| Rows | 47,837 tickets |
| Columns | `Document` (ticket text), `Topic_group` (support group label) |
| Task | Single-label, 8-class text classification |

## Labels (support groups)

| Group | Tickets | Share | Typical clues |
|---|---:|---:|---|
| Hardware | 13,617 | 28.5% | laptop, monitor, connection, meeting room, cable |
| HR Support | 10,915 | 22.8% | new starter, timecard, absence, maternity |
| Access | 7,125 | 14.9% | Confluence, password reset, license, git repository |
| Miscellaneous | 7,060 | 14.8% | change approval, owner, mailing list |
| Storage | 2,777 | 5.8% | folder, size increase, mailbox quota |
| Purchase | 2,464 | 5.2% | PO, purchased, ordered, receipts |
| Internal Project | 2,119 | 4.4% | project code, PAS setup, pipeline |
| Administrative rights | 1,760 | 3.7% | upgrade, Outlook, Windows, installs |

Largest-to-smallest class ratio: **7.7 : 1**.

## Text characteristics
- Median length **26 words**; 95th percentile 136; max 981 (long email threads).
- Already pre-processed by the dataset author: lowercased, punctuation and most stop words removed, personal names stripped. Some function words (*for, with, you*) and dates (*tuesday, pm*) remain.
- Vocabulary: 12,327 distinct words.

## Quality checks (from `notebooks/01_data_exploration.ipynb`)
- 0 missing values, 0 exact duplicates, 0 duplicates after whitespace normalisation.
- 0 identical texts with conflicting labels.
- 49 tickets with ≤ 3 words (to be reviewed in Stage 2).

## Cleaning and splits (from `notebooks/02_cleaning_and_splits.ipynb`)
- `src/preprocess.py::clean_text()` reproduces the author's pre-processing (letters only, removed stop words, 1-letter tokens dropped) and maps scrubbed words to seen equivalents (*email→mail, wifi→wireless, login→log, username→user, internet→network*). Applied to the corpus it changes only 0.85% of tickets (stray punctuation, contraction fragments).
- **Near-duplicates:** after removing dates/weekdays/reply markers, 386 groups of templated or recurring tickets were found; one copy of each is kept (811 copies removed) to prevent train/test leakage.
- **Conflicting labels:** 13 near-duplicate groups (48 tickets) routed to different groups were dropped.
- **Final dataset:** 46,978 tickets (−1.8%); class shares change by ≤ 0.4 points.
- **Splits:** stratified 70 / 15 / 15, seed 42 → train 32,884 · val 7,047 · test 7,047 (`data/processed/`, columns `text`, `label`).

| File | SHA-256 |
|---|---|
| `train.csv` | `315a22db900d6607a8044be5f8a4d42f64702736beca5199c1be6216409e4864` |
| `val.csv` | `3d4d5391e37283aa6a195c62a24c0e08e92ac03785b8b33211121df18bce15ff` |
| `test.csv` | `333ba9ba34895451ca69e8c36cdcc098b6ad5ec11dd5b361141baefb4b556cd7` |

## Known caveats
- **Organisation-specific routing.** Labels reflect one company's conventions (e.g. Outlook issues → Administrative rights; physical restricted-area access → HR Support).
- **Ambiguity / label noise.** Some keywords split across groups (e.g. *password*: 71% Access, 20% Hardware). Labels are kept as-is; impact is studied in error analysis.
- **Mixed scope.** Not all groups are IT (HR Support, Purchase, Internal Project) – they are best described as internal *support groups*.
- **Pre-processed text.** Any new ticket (demo input, extra test sets) must pass through the same cleaning before prediction.
