import json, numpy as np, streamlit as st, pickle
from scipy.sparse import hstack

st.set_page_config(page_title="Emotion Classifier", layout="centered")
st.title("Multi-Label Emotion Classifier")

# load artifacts (classical example)
WORD = pickle.load(open("tfidf_word.pkl","rb"))
CHAR = pickle.load(open("tfidf_char.pkl","rb"))
CLF  = pickle.load(open("tfidf_lr_ovr.pkl","rb"))
THR  = json.load(open("tfidf_thresholds.json"))
LABELS = ["anger","fear","joy","sadness","surprise"]
TH = np.array([THR[l] for l in LABELS], dtype=float)

txt = st.text_area("Input text", height=180)

cols = st.columns(5)
thr = []
for i,lab in enumerate(LABELS):
    thr.append(cols[i].slider(f"{lab} thr", 0.05, 0.95, float(TH[i]), 0.01))
thr = np.array(thr, dtype=float)

if st.button("Predict"):
    X = hstack([WORD.transform([txt]), CHAR.transform([txt])])
    proba = np.vstack([est.predict_proba(X)[:,1] for est in CLF.estimators_]).T[0]
    pred  = (proba >= thr).astype(int)
    st.json({LABELS[i]: float(proba[i]) for i in range(len(LABELS))})
    st.success("Labels: " + ", ".join([LABELS[i] for i in range(len(LABELS)) if pred[i]==1]) or "(none)")
