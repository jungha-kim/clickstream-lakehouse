# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand, when, col, unix_timestamp, to_timestamp
import datetime

spark = SparkSession.builder.getOrCreate()

# 1) 가상 사용자 1,000명 만들기 (user_id 0 ~ 999)
user_count = 1000
users = spark.range(user_count).withColumnRenamed("id", "user_id")

# 2) A/B 분할: rand() < 0.5 이면 'A', 아니면 'B'
users = users.withColumn(
    "variant",
    when(rand() < 0.5, "A").otherwise("B")
)

# 3) 방문(timestamp)과 전환(converted) 이벤트 생성
#    - variant A 그룹에선 10% 확률로 converted=1
#    - variant B 그룹에선 12% 확률로 converted=1
users = users.withColumn("days_ago", (rand() * 7).cast("int"))

# 2) timestamp: 현재 시각(epoch 초) - days_ago*86400 → to_timestamp
users = users.withColumn(
    "timestamp",
    to_timestamp(
        unix_timestamp() - col("days_ago") * 86400
    )
)

# 3) converted: A 그룹은 10% 확률로 1, B 그룹은 12% 확률로 1, 나머지는 0
events = users.withColumn(
    "converted",
    when((col("variant") == "A") & (rand() < 0.10), 1)
    .when((col("variant") == "B") & (rand() < 0.12), 1)
    .otherwise(0)
).select("user_id", "variant", "timestamp", "converted")

events.show(5, truncate=False)

# 4) Delta Lake에 덮어쓰기 (overwrite)
events.write.format("delta").mode("overwrite").save("/delta/ab_test_events")

# 5) 메타테이블로 등록 (한 번만 실행하면 이후 읽을 때 편함)
spark.sql("""
  CREATE TABLE IF NOT EXISTS ab_test_events
  USING DELTA
  LOCATION '/delta/ab_test_events'
""")

print("✅ A/B 테스트용 이벤트가 '/delta/ab_test_events' 에 저장되었습니다.")


# COMMAND ----------

# ┌── 셀 2: 그룹별 전환율 집계
from pyspark.sql.functions import sum as _sum, count, col, expr

df = spark.table("ab_test_events")

# 1) 그룹별 총 사용자 수 (uniq)와 전환 수(합계) 계산
#    - 여기선 events에 user_id가 한 번씩만 있다고 가정(중복 없음)
agg = df.groupBy("variant").agg(
    count("*").alias("total_events"),          # 이벤트 수 (여기선 사용자 수와 동일)
    _sum("converted").alias("total_conversions")# 전환된 건수 합계
)

# 2) 전환율 계산: total_conversions / total_events
agg = agg.withColumn(
    "conversion_rate",
    (col("total_conversions") / col("total_events")).cast("double")
)

display(agg)

# COMMAND ----------

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

pdf = agg.toPandas()

# A, B 그룹 total_conversions (성공 횟수)와 total_events (시행 횟수) 리스트로
success_counts = pdf["total_conversions"].tolist()
nobs = pdf["total_events"].tolist()

# proportions_ztest(성공수, 시행수)
z_stat, p_value = proportions_ztest(count=success_counts, nobs=nobs)

print(f"z-statistic = {z_stat:.4f}")
print(f"p-value     = {p_value:.4f}")

if p_value < 0.05:
    print("=> 두 그룹 간 전환율 차이가 통계적으로 유의미합니다 (p < 0.05).")
else:
    print("=> 두 그룹 간 전환율 차이가 통계적으로 유의미하지 않습니다 (p ≥ 0.05).")