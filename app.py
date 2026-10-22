"""GhostSpace AI - Flask API + dashboard.   Run: python app.py  ->  http://localhost:5000"""
import json, os, random
import joblib, pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from features import AGG, FEATS, RAW, engineer, from_aggregates

app = Flask(__name__, static_folder="static")
M = joblib.load("models/model.joblib")
RESULTS = json.load(open("models/results.json"))
LABELS = ["Empty", "1 person", "2 people", "3 people"]


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", source=RESULTS["meta"]["source"], rows=RESULTS["meta"]["rows"])


@app.get("/api/results")
def results():
    return jsonify(RESULTS)


@app.get("/api/sample")
def sample():
    c = int(request.args.get("count", 0))
    return jsonify(random.choice(RESULTS["samples"][str(max(0, min(3, c)))]))


@app.post("/api/predict")
def predict():
    body = request.get_json(force=True, silent=True) or {}
    missing = [k for k in AGG if k not in body]
    if missing:
        return jsonify(error=f"Missing fields: {', '.join(missing)}", expected=AGG), 400
    try:
        a = {k: float(body[k]) for k in AGG}
    except (TypeError, ValueError):
        return jsonify(error="All fields must be numbers"), 400
    row = from_aggregates(a, M["base"])
    X = engineer(pd.DataFrame([row[RAW]]))[FEATS]
    proba = M["cls"].predict_proba(X)[0]
    count = int(M["cls"].classes_[proba.argmax()])
    reg = float(max(0, min(3, M["reg"].predict(X)[0])))
    cl = int(M["km"].predict(M["pre"].transform(pd.DataFrame([row[RAW]])))[0])
    lights_on = a["light"] > M["thr"]
    ghost = lights_on and count == 0
    advice = ("Ghost space: lights are on but no one is detected. Switch off lighting and relax HVAC."
              if ghost else "Well used right now." if count >= 2 else
              "Lightly used - consider consolidating bookings." if count == 1 else "Empty and dark - nothing to save.")
    return jsonify(count=count, label=LABELS[count], proba=[round(float(p), 4) for p in proba], regression=round(reg, 2),
                   state=M["rank"][cl], lights_on=bool(lights_on), ghost=bool(ghost), advice=advice)


if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)))
