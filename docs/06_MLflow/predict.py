import requests, json

# 1) 학습 때 사용한 feature 형식에 맞춰 payload 구성
payload = {
    "dataframe_records": [
        {"feature": 0},
        {"feature": 1}
    ]
}

# 2) REST API 호출
url = "http://127.0.0.1:1234/predict"
headers = {"Content-Type": "application/json"}
resp = requests.post(url, headers=headers, data=json.dumps(payload))

print("Status Code:", resp.status_code)
print("Predictions:", resp.json())
