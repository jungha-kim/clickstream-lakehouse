# serve.py
import pickle
from flask import Flask, request, jsonify
import pandas as pd

# 1) Pickle 로 모델 불러오기
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# 2) Flask 앱 생성
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "UP"})

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    records = payload.get("dataframe_records", [])
    df = pd.DataFrame(records)
    preds = model.predict(df)
    return jsonify(preds.tolist())

if __name__ == "__main__":
    # 모든 인터페이스, 포트 1234
    app.run(host="0.0.0.0", port=1234)
