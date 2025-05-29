# Databricks notebook source
from pyspark.sql.functions import window, col

# 1) rate 소스 & 5초 윈도우 집계
streamDF = spark.readStream.format("rate").option("rowsPerSecond", 10).load()
aggDF = (streamDF
         .withWatermark("timestamp", "10 seconds")
         .groupBy(window(col("timestamp"), "5 seconds"))
         .count())

# 2) 한번만 실행하고 멈추는 trigger
query = (aggDF.writeStream
            .format("delta")
            .outputMode("complete")
            .option("checkpointLocation", "/delta/_checkpoints/rate_agg")
            .trigger(once=True)        
            .start("/delta/rate_agg"))

# 동기적으로 마이크로배치가 끝날 때까지 기다림
query.awaitTermination()


# COMMAND ----------

df = spark.read.format("delta").load("/delta/rate_agg")
df.show(truncate=False)

# COMMAND ----------

# MAGIC %fs ls /delta/rate_agg