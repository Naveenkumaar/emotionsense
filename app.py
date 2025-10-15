### Hugging Face Space (Gradio) 

# app.py
import json
import numpy as np
import gradio as gr

# ====== choose one stack ======
USE_CLASSICAL = True   # set False if using transformers artifacts

if USE_CLASSICAL:
    import pickle
    from scipy.sparse import hstack
    import pandas as pd

    with open("tfidf_word.pkl","rb") as f: WORD = pickle.load(f)
    with open("tfidf_char.pkl","rb") as f: CHAR = pickle.load(f)
    with open("tfidf_lr_ovr.pkl","rb") as f: CLF  = pickle.load(f)
    with open("tfidf_thresholds.json","r") as f: THR = json.load(f)

    LABELS = ["anger","fear","joy","sadness","surprise"]
    TH = np.array([THR[l] for l in LABELS], dtype=float)

    def predict(txt, thresholds=None):
        if thresholds is None: thresholds = TH
        Xw = WORD.transform([txt])
        Xc = CHAR.transform([txt])
        X  = hstack([Xw, Xc])
        # get per-class probabilities from OvR estimators
        proba = np.vstack([est.predict_proba(X)[:,1] for est in CLF.estimators_]).T[0]
        pred  = (proba >= thresholds).astype(int)
        return {LABELS[i]: float(proba[i]) for i in range(len(LABELS))}, \
               [l for i,l in enumerate(LABELS) if pred[i] == 1]
else:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    LABELS = ["anger","fear","joy","sadness","surprise"]
    id2label = {i:l for i,l in enumerate(LABELS)}
    model = AutoModelForSequenceClassification.from_pretrained("hf_model")
    tok   = AutoTokenizer.from_pretrained("tokenizer")
    with open("thresholds.json","r") as f: THR = json.load(f)
    TH = np.array([THR[l] for l in LABELS], dtype=float)

    @torch.no_grad()
    def predict(txt, thresholds=None, max_len=192):
        if thresholds is None: thresholds = TH
        enc = tok(txt, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        logits = model(**enc).logits
        proba = torch.sigmoid(logits).cpu().numpy()[0]
        pred  = (proba >= thresholds).astype(int)
        return {LABELS[i]: float(proba[i]) for i in range(len(LABELS))}, \
               [l for i,l in enumerate(LABELS) if pred[i] == 1]

# ---------- Gradio UI ----------
def ui_predict(text, t_anger, t_fear, t_joy, t_sadness, t_surprise):
    thresholds = np.array([t_anger, t_fear, t_joy, t_sadness, t_surprise], dtype=float)
    probs, labels = predict(text, thresholds)
    return probs, ", ".join(labels) if labels else "(none)"

with gr.Blocks() as demo:
    gr.Markdown("# Multi-Label Emotion Classifier\nEnter text, adjust thresholds, and predict.")
    with gr.Row():
        txt = gr.Textbox(lines=5, label="Input text", placeholder="Type or paste short text...")
    with gr.Row():
        t1 = gr.Slider(0.05, 0.95, value=float(TH[0]), step=0.01, label="anger threshold")
        t2 = gr.Slider(0.05, 0.95, value=float(TH[1]), step=0.01, label="fear threshold")
        t3 = gr.Slider(0.05, 0.95, value=float(TH[2]), step=0.01, label="joy threshold")
        t4 = gr.Slider(0.05, 0.95, value=float(TH[3]), step=0.01, label="sadness threshold")
        t5 = gr.Slider(0.05, 0.95, value=float(TH[4]), step=0.01, label="surprise threshold")
    btn = gr.Button("Predict")
    probs_json = gr.JSON(label="Per-label probabilities")
    labels_txt = gr.Textbox(label="Predicted labels", interactive=False)

    btn.click(ui_predict, inputs=[txt, t1,t2,t3,t4,t5], outputs=[probs_json, labels_txt])

demo.launch()
