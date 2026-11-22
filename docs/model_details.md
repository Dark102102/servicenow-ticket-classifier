# Model Details (technical reference for the Analysis tab)

The notebooks keep short summaries; this file holds the full technical description of each method.
Plain-language versions for the website are in `notes_for_website.md`.

**Protocol for every model:** settings are chosen on the validation set (macro-F1); the test set is used once at the end.
Class imbalance is handled with class weights `n / (8 × count)` in every loss.

## Data preparation
- `clean_text()` reproduces the dataset's pre-processing (lowercase, letters only, removed stop words, no 1-letter tokens) and maps
  words scrubbed by anonymisation to seen equivalents (*email→mail, wifi→wireless, login→log, username→user, internet→network*).
- Near-duplicates: dates/weekdays/reply markers are normalised away; one copy per group is kept (811 removed) to prevent train/test leakage.
  13 near-duplicate groups (48 tickets) with conflicting labels are dropped.
- Stratified 70 / 15 / 15 split, seed 42: train 32,884 · validation 7,047 · test 7,047.

## Baseline – TF-IDF + Logistic Regression (not a neural network)
1. **TF-IDF** turns each ticket into a vector with one entry per word or word pair. A term scores high if frequent in this ticket but
   rare across tickets (*please* low, *po*/*timecard* high). Sub-linear term frequency, min document frequency 2.
2. **Logistic Regression** learns one weight per term for each group and picks the highest-scoring group (softmax over 8). Linear, no hidden layers.
- Tuned: words vs. words + pairs, class weighting, C ∈ {0.5, 1, 2, 5, 10, 20} → best: words + pairs, balanced, C = 5 (128,435 features).

## ANN – multi-layer perceptron
`text → TextVectorization (TF-IDF, 50k words + word pairs) → L2 normalisation → Dense(256) + ReLU → Dropout(0.5) → Dense(8) + Softmax`
- **L2 normalisation** rescales each TF-IDF vector to length 1 so long email threads do not swamp training.
- **Dense:** every input connects to every unit (weighted sum + bias). **ReLU** `max(0, x)` adds the non-linearity that lets the network
  learn word *combinations* (e.g. *access + card* vs. *access + confluence*).
- **Dropout** randomly disables units during training to limit memorisation. **Softmax** gives 8 probabilities summing to 1.
- Loss: weighted categorical cross-entropy; Adam (lr 0.001), batch 64, early stopping on validation loss (best epoch restored).
- Validation experiments (256 units): raw TF-IDF 20k 0.835 · L2-normalised 20k 0.849 · **L2-normalised 50k 0.855** · 512→128 units 0.840.
  Final size comparison: 128 units 0.853 · **256 units 0.856** · 512→128 units 0.833. 12.8M trainable parameters.

## BiLSTM – bidirectional recurrent network
`text → TextVectorization (word IDs, first 150 words) → Embedding (11,135 words × 128) → Bidirectional LSTM (64 per direction) → Dropout(0.4) → Dense(8) + Softmax`
- **Embedding:** a learned 128-number vector per word (trained from scratch); similar words end up close together.
- **LSTM:** a recurrent cell whose input, forget and output gates decide what to store, forget and emit at each word, avoiding the
  vanishing-gradient problem of plain RNNs.
- **Bidirectional:** one LSTM reads left→right and one right→left; outputs are concatenated so each position sees both sides.
- **Masking** ignores padding; spatial dropout 0.2 on embeddings. Same loss/optimiser/early stopping as the ANN.
- Variants on validation: **final states 0.845** · max-pool 0.839 · BiGRU max-pool 0.841. 1.5M trainable parameters.
  150 words fully covers ~96% of tickets.

## DistilBERT – fine-tuned Transformer (prepared, GPU run pending)
`text → WordPiece tokenizer (≤ 256 tokens) → 6 Transformer encoder layers (768-d, 12 attention heads) → [CLS] vector → Linear(768) + ReLU → Dropout(0.2) → Linear(8) → Softmax`
- Pre-trained on general English text (masked-word prediction), then **fine-tuned** end to end on our tickets.
- **WordPiece** splits words into sub-word pieces, so there are no unknown words. **Self-attention** lets each token weigh every other
  token. The **[CLS]** token's final vector summarises the ticket for the classifier.
- AdamW (lr 3e-5, weight decay 0.01), 10% linear warm-up then decay, batch 32, fp16 on GPU, weighted cross-entropy, 4 epochs,
  best epoch by validation macro-F1. 67M trainable parameters.

## Evaluation and routing
- Metrics: accuracy, macro and weighted precision/recall/F1, per-class F1, row-normalised confusion matrices.
- Ensemble: probability averages of every model combination compared on validation (no combination beat the best single model).
- Confidence routing: auto-route only when max probability ≥ threshold; threshold = lowest value giving ≥ 95% accuracy on validation
  (baseline 0.68 → test: 70.9% auto-routed at 95.6% accuracy, 77.7% of all errors sent to review).
