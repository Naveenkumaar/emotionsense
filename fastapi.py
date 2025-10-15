from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np, pickle, json
from scipy.sparse import hstack

app = FastAPI(title="Emotion API")

# load classical artifacts
WORD = pickle.load(open("tfidf_word.pkl","rb"))
CHAR = pickle.load(open("tfidf_char.pkl","rb"))
CLF  = pickle.load(open("tfidf_lr_ovr.pkl","rb"))
THR  = json.load(open("tfidf_thresholds.json"))
LABELS = ["anger","fear","joy","sadness","surprise"]
TH = np.array([THR[l] for l in LABELS], dtype=float)

class Inp(BaseModel):
    text: str
    thresholds: list[float] | None = None

@app.post("/predict")
def predict(inp: Inp):
    thresholds = np.array(inp.thresholds, dtype=float) if inp.thresholds else TH
    X = hstack([WORD.transform([inp.text]), CHAR.transform([inp.text])])
    proba = np.vstack([est.predict_proba(X)[:,1] for est in CLF.estimators_]).T[0]
    pred  = (proba >= thresholds).astype(int)
    return {
        "proba": {LABELS[i]: float(proba[i]) for i in range(len(LABELS))},
        "labels": [LABELS[i] for i in range(len(LABELS)) if pred[i]==1]
    }
