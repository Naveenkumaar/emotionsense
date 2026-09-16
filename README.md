<div align="center">

# 🎭 EmotionSense — Multi-Label Emotion Classification

**Detect the multiple emotions present in a piece of text at once — anger, fear, joy, sadness, surprise — with per-class thresholds tuned for a strong multi-label score.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-FF7C00)](app.py)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](app_streamlit.py)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](fastapi.py)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

> Emotions overlap — a single sentence can be angry *and* sad at once. EmotionSense
> frames this as a **multi-label** problem (not single-class), predicting each
> emotion independently with its own decision threshold.

---

## Table of contents

- [Highlights](#highlights)
- [The task](#the-task)
- [Approach](#approach)
- [Quick start](#quick-start)
- [Serving it](#serving-it)
- [Repository map](#repository-map)

---

## Highlights

- **5 emotions, predicted independently** — anger · fear · joy · sadness · surprise.
- **TF-IDF (word + char n-grams) + One-vs-Rest Logistic Regression** — a fast, strong classical baseline.
- **Per-class thresholds** tuned on validation (`tfidf_thresholds.json`) instead of a flat 0.5 cutoff — the key lever for multi-label F1.
- **Optional transformer stack** — the same interface swaps in fine-tuned embeddings when artifacts are present.
- **Three ready front-ends** on one model: Gradio, Streamlit, and a FastAPI service.

---

## The task

Given a sentence, output the set of emotions it expresses. Each example can carry
**zero, one, or several** labels, so the model scores every emotion separately and
applies a per-label threshold — the standard multi-label formulation, evaluated
with a macro/micro-F1 style metric.

Training data (`train.csv`) is labelled across the five emotion columns; the model
produces a `submission.csv` of predictions for the held-out `test.csv`.

---

## Approach

1. **Features** — TF-IDF on both **word** and **character** n-grams, concatenated (character n-grams help with misspellings and intensifiers).
2. **Model** — One-vs-Rest **Logistic Regression**: one calibrated classifier per emotion.
3. **Thresholding** — a threshold is tuned per emotion on validation and saved to `tfidf_thresholds.json`; at inference each probability is compared to its own threshold.
4. **Inference** — `predict(text)` returns both the per-emotion probabilities and the final predicted label set.

The notebook (`*.ipynb`) covers EDA, the training run, and threshold tuning; `Analysis.ipynb` holds the exploratory work.

---

## Quick start

```bash
pip install -r requirements.txt

# Gradio demo (loads the saved TF-IDF + LR artifacts)
python app.py
```

```python
# programmatic use
from app import predict
probs, labels = predict("the dentist did a lousy job and now my teeth are ruined")
# probs -> {"anger": 0.71, "sadness": 0.66, ...}   labels -> ["anger", "sadness"]
```

---

## Serving it

The same model is exposed three ways:

```bash
python app.py                       # Gradio (Hugging Face Space style)
streamlit run app_streamlit.py      # Streamlit UI
uvicorn fastapi:app --reload        # FastAPI REST service
```

---

## Repository map

```
app.py                 Gradio demo + the predict() entry point
app_streamlit.py       Streamlit UI
fastapi.py             FastAPI REST service
*.ipynb                training notebook (EDA, model, threshold tuning)
Analysis.ipynb         exploratory analysis
train.csv / test.csv   labelled training data + held-out test set
submission.csv         model predictions on the test set
requirements.txt       pinned dependencies
```

---

<div align="center">

Multi-label NLP project · [MIT License](LICENSE)

</div>
