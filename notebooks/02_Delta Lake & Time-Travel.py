# Databricks notebook source
from pyspark.sql.types import *
schema = StructType([
    StructField("order_id", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("category", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("order_ts", TimestampType(),True),
])

csv_path = "/FileStore/tables/sales.csv"
sales_df = (spark.read.option("header", True).schema(schema).csv(csv_path))

delta_path = "/delta/sales_bronze"
(sales_df.write.format("delta").mode("overwrite").save(delta_path))

spark.sql(f"""
          CREATE TABLE IF NOT EXISTS sales_bronze
          USING DELTA
          LOCATION '{delta_path}'
          """)

spark.sql("""
          UPDATE sales_bronze
          SET amount = amount * 1.10
          WHERE category = 'Book'
          """)

hist = spark.sql("DESCRIBE HISTORY sales_bronze")
hist.show(truncate=False)