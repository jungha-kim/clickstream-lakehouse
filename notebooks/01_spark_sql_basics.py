# Databricks notebook source
schema = """
  order_id STRING,
  user_id  STRING,
  category STRING,
  amount   DOUBLE,
  order_ts TIMESTAMP
"""

sales_df = (spark.read
              .option("header", True)
              .schema(schema)
              .csv("/FileStore/tables/sales.csv"))
sales_df.printSchema()

plan_rows = spark.sql("""
EXPLAIN FORMATTED
SELECT category, SUM(amount) AS total
FROM sales
GROUP BY category
ORDER BY total DESC
""").collect()

print("\n".join(r.plan for r in plan_rows))


# COMMAND ----------

