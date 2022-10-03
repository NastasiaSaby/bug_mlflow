# Databricks notebook source
# MAGIC %pip install mlflow-skinny==1.28

# COMMAND ----------

from typing import List
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, IntegerType

data: List = [
              (1, 3, 1),
  (1, 3, 1),
  (1, 3, 1),
  (1, 3, 1),
  (1, 3, 1),
              (2, 4, 1),
              (2, 3, 1),
              (3, 3, 1),
              (3, 4, 1)
  ]

schema: StructType = StructType([ \
    StructField("utilisateur_identifiant", IntegerType(), True), \
    StructField("diamant_identifiant", IntegerType(), True),
    StructField("nombre_de_fois_achetes", IntegerType(), True)
  ])

diamants_pre_features: DataFrame = spark.createDataFrame(data=data,schema=schema)


from pyspark.ml.recommendation import ALS, ALSModel

als: ALS = ALS(
  userCol="utilisateur_identifiant", 
  itemCol="diamant_identifiant", 
  ratingCol="nombre_de_fois_achetes",
  implicitPrefs=True,
  alpha=40,
  nonnegative=True
)
model: ALSModel = als.fit(diamants_pre_features)

import mlflow
mlflow.set_experiment("/nastasia/ALS_experiment")

with mlflow.start_run() as last_run:
  mlflow.spark.log_model(model, "als_exp")

from mlflow.tracking import MlflowClient
# Get last run from Mlflow experiment
client = MlflowClient()

model_experiment_id = client.get_experiment_by_name("/nastasia/ALS_experiment").experiment_id

runs = client.search_runs(
        model_experiment_id, order_by=["start_time DESC"]
)

run_uuid = runs[0].info.run_uuid

# can be loaded from s3
# model = ALSModel.load(sources_jobs['ALS_model'])
loaded_model = mlflow.spark.load_model(f"runs:/{run_uuid}/als_exp")

# COMMAND ----------

# MAGIC %sh pip freeze

# COMMAND ----------


