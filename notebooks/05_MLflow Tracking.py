# Databricks notebook source
%pip install mlflow scikit-learn pandas
%pip install typing_extensions==4.7.1

import mlflow, mlflow.sklearn
from pyspark.sql import SparkSession
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# SparkSession (CE 클러스터)
spark = SparkSession.builder.getOrCreate()

experiment_name = "/Shared/sales_predict_demo"

exp = mlflow.get_experiment_by_name(experiment_name)
if exp is None:
    exp_id = mlflow.create_experiment(experiment_name)
    print(f"Experiment '{experiment_name}' created with ID = {exp_id}")
else:
    exp_id = exp.experiment_id
    print(f"Using existing experiment ID = {exp_id} for '{experiment_name}'")

# Delta 테이블에서 데이터 로드 & pandas 변환
df = (spark.read
           .format("delta")
           .load("/delta/sales_bronze")
           .selectExpr("amount AS label",
                       "CASE WHEN category='Book' THEN 1 ELSE 0 END AS feature")
           .toPandas())
X, y = df[["feature"]], df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow Experiment 설정
with mlflow.start_run(experiment_id=exp_id) as run:
    model = LinearRegression().fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    # 로그 남기기
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    print("Run finished. ID =", run.info.run_id)
    print("Artifact URI =", run.info.artifact_uri + "/model")


dbutils.fs.cp(
    "dbfs:/FileStore/mlflow_models/sales_model/model.pkl",
    "file:/tmp/model.pkl"
)

# 2) 노트북에서 Python으로 읽어서 로컬로 저장
with open("/tmp/model.pkl", "rb") as src:
    data = src.read()
    
# 3) Jupyter-like 환경이라면 아래로 다운로드 링크 생성
import base64, urllib
b64 = base64.b64encode(data).decode("utf-8")
html = f'<a download="model.pkl" href="data:application/octet-stream;base64,{b64}">Click to download model.pkl</a>'
displayHTML(html)