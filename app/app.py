"""Demo: type a support ticket -> predicted support group, confidence, and an auto-route / needs-review decision.

Run from the project root:
    .venv/bin/python app/app.py
then open http://127.0.0.1:7860
"""
import os
import sys
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr
import joblib
import numpy as np
import pandas as pd

from src.data import LABELS, RESULTS_DIR
from src.preprocess import clean_text

MODELS_DIR = ROOT / "models"
TARGET_ACCURACY = 0.95  # same rule as notebooks/05_evaluation.ipynb
PCOLS = [f"p_{l}" for l in LABELS]

AVAILABLE = {
    "Baseline (TF-IDF + Logistic Regression)": "baseline_tfidf_lr",
    "ANN (multi-layer perceptron)": "ann_mlp",
    "BiLSTM (recurrent network)": "bilstm",
}
if (MODELS_DIR / "distilbert" / "config.json").exists():
    AVAILABLE["DistilBERT (fine-tuned transformer)"] = "distilbert"
DEFAULT_MODEL = next(iter(AVAILABLE))


def pick_threshold(key: str) -> float:
    """Lowest confidence threshold whose auto-routed validation tickets are >= 95% correct."""
    val = pd.read_csv(RESULTS_DIR / f"preds_val_{key}.csv")
    conf = val[PCOLS].to_numpy().max(1)
    correct = np.array(LABELS)[val[PCOLS].to_numpy().argmax(1)] == val["label"].to_numpy()
    for t in np.round(np.arange(0.0, 1.0, 0.01), 2):
        keep = conf >= t
        if keep.any() and correct[keep].mean() >= TARGET_ACCURACY:
            return float(t)
    return 0.99


THRESHOLDS = {k: pick_threshold(k) for k in AVAILABLE.values() if (RESULTS_DIR / f"preds_val_{k}.csv").exists()}


@lru_cache(maxsize=None)
def load(key: str):
    if key == "baseline_tfidf_lr":
        return joblib.load(MODELS_DIR / "baseline_tfidf_lr.joblib")
    if key in ("ann_mlp", "bilstm"):
        import keras
        return keras.models.load_model(MODELS_DIR / f"{key}.keras")
    if key == "distilbert":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = MODELS_DIR / "distilbert"
        return AutoTokenizer.from_pretrained(path), AutoModelForSequenceClassification.from_pretrained(path).eval()
    raise ValueError(key)


def probabilities(key: str, cleaned: str) -> np.ndarray:
    model = load(key)
    if key == "baseline_tfidf_lr":
        p = model.predict_proba([cleaned])[0]
        return p[[list(model.classes_).index(l) for l in LABELS]]
    if key in ("ann_mlp", "bilstm"):
        import tensorflow as tf
        return model.predict(tf.constant([cleaned]), verbose=0)[0]
    import torch
    tok, m = model
    with torch.no_grad():
        logits = m(**tok(cleaned, truncation=True, max_length=256, return_tensors="pt")).logits
    return torch.softmax(logits, -1)[0].numpy()


def explain_baseline(cleaned: str, label: str):
    """Per-word evidence for the predicted group: TF-IDF value x model weight (word pairs split between their words)."""
    pipe = load("baseline_tfidf_lr")
    vec, clf = pipe.named_steps["tfidfvectorizer"], pipe.named_steps["logisticregression"]
    x = vec.transform([cleaned]).tocoo()
    w = clf.coef_[list(clf.classes_).index(label)]
    vocab = vec.get_feature_names_out()
    score = {}
    for j, v in zip(x.col, x.data):
        parts = vocab[j].split()
        for part in parts:
            score[part] = score.get(part, 0.0) + v * w[j] / len(parts)
    top = max((abs(s) for s in score.values()), default=1.0) or 1.0
    return [(word + " ", float(np.clip(score.get(word, 0.0) / top, -1, 1))) for word in cleaned.split()]


def badge(decision: str, label: str, conf: float, threshold: float) -> str:
    auto = decision == "auto"
    bg, fg, border = ("#e8f1fc", "#123e75", "#2a78d6") if auto else ("#fdf3e1", "#6b4400", "#eda100")
    icon = "✓" if auto else "!"
    strong = f'<strong style="color:{fg} !important">{label}</strong>'
    title = f"Auto-route to {strong}" if auto else "Needs human review"
    detail = (f"Confidence {conf:.0%} is above the {threshold:.0%} threshold."
              if auto else f"Confidence {conf:.0%} is below the {threshold:.0%} threshold. Best guess: {strong}.")
    # Every element carries its own colour so Gradio's light/dark theme cannot override it.
    return (f'<div style="border:2px solid {border};background:{bg};border-radius:10px;padding:14px 16px;line-height:1.4">'
            f'<div style="color:{fg} !important;font-size:16px"><span style="color:{fg} !important;font-size:20px;'
            f'font-weight:700;margin-right:8px">{icon}</span>{title}</div>'
            f'<div style="color:{fg} !important;font-size:13px;margin-top:4px">{detail}</div></div>')


def route(text: str, model_name: str):
    cleaned = clean_text(text or "")
    if not cleaned:
        return "", {}, "<i>Type a ticket to see a prediction.</i>", []
    key = AVAILABLE[model_name]
    p = probabilities(key, cleaned)
    label, conf = LABELS[int(p.argmax())], float(p.max())
    threshold = THRESHOLDS.get(key, 0.68)
    decision = "auto" if conf >= threshold else "review"
    words = explain_baseline(cleaned, label) if key == "baseline_tfidf_lr" else \
        [("Word highlighting is available for the baseline model only.", None)]
    return cleaned, {l: float(v) for l, v in zip(LABELS, p)}, badge(decision, label, conf, threshold), words


@lru_cache(maxsize=1)
def servicenow_client():
    from src.servicenow import ServiceNowClient
    client = ServiceNowClient.from_env()
    client.ensure_groups()
    return client


def note(html: str, color: str = "#52514e") -> str:
    return f'<div style="color:{color} !important;font-size:14px;padding:6px 2px">{html}</div>'


def route_and_create_incident(text: str, model_name: str):
    """Route the ticket, then create a ServiceNow incident assigned to the predicted group (or to triage)."""
    outputs = route(text, model_name)
    cleaned = outputs[0]
    if not cleaned:
        return (*outputs, "")
    from src.servicenow import is_configured
    if not is_configured():
        return (*outputs, note("ServiceNow is not configured yet: fill in <code>.env</code> (see <code>.env.example</code>)."))
    key = AVAILABLE[model_name]
    p = probabilities(key, cleaned)
    label, conf = LABELS[int(p.argmax())], float(p.max())
    try:
        inc = servicenow_client().create_incident(text, label, conf, THRESHOLDS.get(key, 0.68), model_name)
    except Exception as e:  # instance asleep, wrong password, network...
        return (*outputs, note(f"Could not reach ServiceNow: {e}. If the instance is hibernating, log in to it "
                               f"in your browser to wake it, then try again.", "#c0392b"))
    where = f"assigned to <b>{inc['group']}</b>" + ("" if inc["auto"] else " for human review")
    return (*outputs, note(f'ServiceNow incident <a href="{inc["url"]}" target="_blank"><b>{inc["number"]}</b></a> '
                           f'created and {where}.', "#123e75"))


EXAMPLES = [
    "My laptop screen keeps flickering and the docking station does not detect my second monitor.",
    "Please raise a PO for two new headsets, the purchase has been approved by my manager.",
    "I cannot open Confluence, it says my account has no access to the project space.",
    "Our shared folder is full, could you please increase its size?",
    "We have a new starter joining on Monday, please create her account and send the starter form.",
    "Outlook keeps crashing after the latest Windows upgrade.",
    "I changed my password yesterday and now I cannot log in to the VPN or my email.",
    "Can someone help me with this, it is not working since this morning.",
]

with gr.Blocks(title="Support Ticket Router") as demo:
    gr.Markdown("# Support Ticket Router\nDescribe an IT or internal support problem. A trained model predicts which of 8 support "
                "teams should handle it, and only **auto-routes** when it is confident; otherwise the ticket goes to a person.")
    with gr.Row():
        with gr.Column(scale=5):
            text = gr.Textbox(label="Ticket", lines=5, placeholder="e.g. My laptop will not connect to the wireless network in meeting room 3")
            model = gr.Dropdown(list(AVAILABLE), value=DEFAULT_MODEL, label="Model")
            with gr.Row():
                submit = gr.Button("Route ticket", variant="primary")
                snow_btn = gr.Button("Route + create ServiceNow incident", variant="secondary")
                gr.ClearButton([text])
            snow_result = gr.HTML()
            gr.Examples(EXAMPLES, inputs=text, label="Example tickets")
        with gr.Column(scale=5):
            decision = gr.HTML(label="Decision")
            probs = gr.Label(num_top_classes=3, label="Top predictions")
            words = gr.HighlightedText(label="Words that pushed towards (red) or away from (blue) the prediction",
                                       show_legend=False, combine_adjacent=False)
            cleaned = gr.Textbox(label="What the model sees after cleaning", interactive=False, lines=2)
    gr.Markdown(f"Confidence thresholds are chosen on validation tickets so that auto-routed tickets are at least "
                f"{TARGET_ACCURACY:.0%} correct: " + ", ".join(f"{n.split(' (')[0]} {THRESHOLDS[k]:.2f}"
                                                            for n, k in AVAILABLE.items() if k in THRESHOLDS) + ".")
    outputs = [cleaned, probs, decision, words]
    submit.click(route, [text, model], outputs)
    text.submit(route, [text, model], outputs)
    model.change(route, [text, model], outputs)
    snow_btn.click(route_and_create_incident, [text, model], outputs + [snow_result])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)))
