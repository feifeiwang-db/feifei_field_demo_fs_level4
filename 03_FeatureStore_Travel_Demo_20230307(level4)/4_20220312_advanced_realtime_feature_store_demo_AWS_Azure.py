# Databricks notebook source
# MAGIC %md # Important Note (must read)
# MAGIC You can directly run this demo notebook in `e2-field-eng-west` [AWS workspace](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#ml/dashboard) or `field-eng-east` [Azure workspace](https://adb-984752964297111.11.azuredatabricks.net/?o=984752964297111#), since the secret scopes for accessing dynamoDB/cosmosDB are set up for all field eng users. 
# MAGIC 
# MAGIC How to run this demo if you are in `field-eng-east` [Azure workspace](https://adb-984752964297111.11.azuredatabricks.net/?o=984752964297111#): 
# MAGIC * **important!! must read!!** If you are in the above Azure workspace with UC enabled, **please use a non-UC cluster 11.3+ ML, and select `Shared Compute` policy**, also **you must** install the additional spark connector library by input maven coordinate based on the instructions [here](https://github.com/Azure/azure-sdk-for-java/blob/main/sdk/cosmos/azure-cosmos-spark_3-2_2-12/README.md#download) (i.e. `com.azure.cosmos.spark:azure-cosmos-spark_3-2_2-12:4.17.2`) . Currently UC clusters are not supported. **Once 13.0 DBR is available, UC clusters can then be used to run this demo.** 
# MAGIC * You can run all commands above the last "cleanup" command to check the generated feature store tables, model serving endpoint, and published tables to cosmosDB [here](https://portal.azure.com/#@DataBricksInc.onmicrosoft.com/resource/subscriptions/3f2e4d32-8e8d-46d6-82bc-5bb8d962328b/resourcegroups/field-eng-east/providers/Microsoft.DocumentDB/databaseAccounts/field-demo/collectionSetting) etc.
# MAGIC * Then run the last cleanup command to delete all generated resources after you finish doing this demo.
# MAGIC 
# MAGIC How to run this demo if you are in `e2-field-eng-west` [AWS workspace](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#ml/dashboard): 
# MAGIC * Please use cluster 11.3+ ML 
# MAGIC * You can run all commands above the last "cleanup" command and check the generated feature store tables, model serving endpoint, and published tables to dynamoDB etc.
# MAGIC * Then run the last cleanup command afterwards.
# MAGIC 
# MAGIC For anyone who wants to set up credentials in a different AWS workspace or Azure workspace other than above, please follow instructions in [original blog post's ](https://www.databricks.com/blog/2023/02/16/best-practices-realtime-feature-computation-databricks.html) notebook examples (scroll to the end of the blog), and follow documentations for "work with online stores" ([AWS](https://docs.databricks.com/machine-learning/feature-store/online-feature-stores.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/online-feature-stores)). If you have to obtain internal field eng credentials in order to do the demo in your own workspaces, please contact feifei.wang@databricks.com(AWS) or amine.elhelou@databricks.com (Azure).

# COMMAND ----------

# MAGIC %md # Travel recommendation example notebook
# MAGIC 
# MAGIC This notebook illustrates the use of different feature computation modes: batch, streaming and on-demand. It has been shown that machine learning models degrade in performance as the features become stale. This is true more so for certain type of features than others. If the data being generated updates quickly and factors heavily into the outcome of the model, it should be updated regularly. However, updating static data often would lead to increased costs with no perceived benefits. This notebook illustrates various feature computation modes available in Databricks using Databricks Feature Store based on the feature freshness requirements for a travel recommendation website. 
# MAGIC 
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/realtime/freshness.png"/>
# MAGIC 
# MAGIC This notebook builds a ranking model to predict likelihood of a user purchasing a destination package.
# MAGIC 
# MAGIC The notebook is structured as follows:
# MAGIC 
# MAGIC 1. Explore the dataset
# MAGIC 1. Compute the features in three computation modes
# MAGIC    * Batch features
# MAGIC    * Streaming features
# MAGIC    * On-demand features
# MAGIC 1. Publishes the features to the online store, based on the freshness requirements using streaming or batch mode (This notebook uses cosmosDB. For a list of supported online stores, see the Databricks documentation ([AWS](https://docs.databricks.com/machine-learning/feature-store/publish-features.html)|[Azure](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/publish-features))
# MAGIC 1. Train and deploy the model
# MAGIC 1. Serve real-time queries with automatic feature lookup
# MAGIC 1. Clean up
# MAGIC 
# MAGIC ### Note
# MAGIC There is a blog post [here](https://www.databricks.com/blog/2023/02/16/best-practices-realtime-feature-computation-databricks.html) that further explains this notebook. 

# COMMAND ----------

# MAGIC %run ./initialization

# COMMAND ----------

# MAGIC %md ## Data sets
# MAGIC 
# MAGIC For the travel recommendation model, there are different types of data available: 
# MAGIC 
# MAGIC * __Destination location__ data - A static dataset of destinations for vacation packages. The destination location dataset consists of `latitude`, `longitude`, `name` and `price`. This dataset only changes when a new destination is added. The update frequency for this data is once a month and these features are computed in __batch-mode__. 
# MAGIC * __Destination popularity__ data - The website gathers the popularity information from the website usage logs based on number of impressions (e.g. `mean_impressions`, `mean_clicks_7d`) and user activity on those impressions. In this example, __batch-mode__ is used since the data sees shifts in patterns over longer periods of time. 
# MAGIC * __Destination availability__ data - Whenever a user books a room for the hotel, this affects the destination availability and price (e.g. `destination_availability`, `destination_price`). Because price and availability are a big driver for users booking vacation destinations, this data needs to be kept fairly up-to-date, especially around holiday time. Batch-mode computation with hours of latency would not work, so Spark structured streaming is used to update the data in __streaming-mode__.
# MAGIC * __User preferences__ - Some users prefer to book closer to their current location whereas some prefer to go global and far-off. Because user location can only be determined at the booking time, the __on-demand feature computation__ calculates the `distance` between a context feature such as user location (`user_longitude`, `user_latitude`) and static feature destination location. This way data in offline training and online model serving can remain in sync. 

# COMMAND ----------

# MAGIC %md <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/realtime/schema.png"/>

# COMMAND ----------

# MAGIC %md # Compute features

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}")
print("created/used database_name: ", database_name)

# COMMAND ----------

# MAGIC %md ## Compute batch features
# MAGIC 
# MAGIC Calculate the aggregated features from the vacation purchase logs for destination and users. The destination features include popularity features such as impressions, clicks, and pricing features like price at the time of booking. The user features capture the user profile information such as past purchased price. Because the booking data does not change very often, it can be computed once per day in batch.

# COMMAND ----------

import pyspark.sql.functions as F

vacation_purchase_df = spark.read.option("inferSchema", "true").load("/databricks-datasets/travel_recommendations_realtime/raw_travel_data/fs-demo_vacation-purchase_logs/", format="csv", header="true")
vacation_purchase_df = vacation_purchase_df.withColumn("booking_date", F.to_date("booking_date"))
display(vacation_purchase_df)

# COMMAND ----------

import pyspark.sql.window as w
from pyspark.sql import DataFrame

def user_features_fn(vacation_purchase_df: DataFrame) -> DataFrame:
    """
    Computes the user_features feature group.
    """
    return (
        vacation_purchase_df.withColumn(
            "lookedup_price_7d_rolling_sum",
            F.sum("price").over(
                w.Window.partitionBy("user_id")
                .orderBy(F.col("ts").cast("long"))
                .rangeBetween(start=-(7 * 86400), end=0)
            ),
        )
        .withColumn(
            "lookups_7d_rolling_sum",
            F.count("*").over(
                w.Window.partitionBy("user_id")
                .orderBy(F.col("ts").cast("long"))
                .rangeBetween(start=-(7 * 86400), end=0)
            ),
        )
        .withColumn(
            "mean_price_7d",
            F.col("lookedup_price_7d_rolling_sum") / F.col("lookups_7d_rolling_sum"),
        )
        .withColumn(
            "tickets_purchased",
            F.when(F.col("purchased") == True, F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "last_6m_purchases",
            F.sum("tickets_purchased").over(
                w.Window.partitionBy("user_id")
                .orderBy(F.col("ts").cast("long"))
                .rangeBetween(start=-(6 * 30 * 86400), end=0)
            ),
        )
        .withColumn("day_of_week", F.dayofweek("ts"))
        .select("user_id", "ts", "mean_price_7d", "last_6m_purchases", "day_of_week")
    )

def destination_features_fn(vacation_purchase_df: DataFrame) -> DataFrame:
    """
    Computes the destination_features feature group.
    """
    return (
        vacation_purchase_df.withColumn(
            "clicked", F.when(F.col("clicked") == True, 1).otherwise(0)
        )
        .withColumn(
            "sum_clicks_7d",
            F.sum("clicked").over(
                w.Window.partitionBy("destination_id")
                .orderBy(F.col("ts").cast("long"))
                .rangeBetween(start=-(7 * 86400), end=0)
            ),
        )
        .withColumn(
            "sum_impressions_7d",
            F.count("*").over(
                w.Window.partitionBy("destination_id")
                .orderBy(F.col("ts").cast("long"))
                .rangeBetween(start=-(7 * 86400), end=0)
            ),
        )
        .select("destination_id", "ts", "sum_clicks_7d", "sum_impressions_7d")
    )
    return destination_df

# COMMAND ----------

from databricks import feature_store
fs = feature_store.FeatureStoreClient()

fs.create_table(
    name=fs_table_user_features, 
    primary_keys=["user_id"],
    timestamp_keys="ts",
    df=user_features_fn(vacation_purchase_df),
    description="User Features",
)

fs.create_table(
    name=fs_table_destination_popularity_features, 
    primary_keys=["destination_id"],
    timestamp_keys="ts",
    df=destination_features_fn(vacation_purchase_df),
    description="Destination Popularity Features",
)

# COMMAND ----------

# MAGIC %md Another static dataset is destination location feature which only updates every month because it only needs to be refreshed when a new destination package is offered. 

# COMMAND ----------

destination_location_df = (
  spark.read
 .option("inferSchema", "true")
 .load("/databricks-datasets/travel_recommendations_realtime/raw_travel_data/fs-demo_destination-locations/", 
       format="csv", 
       header="true")
)

fs.create_table(
  name=fs_table_destination_location_features, 
  primary_keys="destination_id",
  df=destination_location_df,
  description="Destination location features."
)

# COMMAND ----------

# MAGIC %md ## Compute streaming features
# MAGIC 
# MAGIC Availability of the destination can hugely affect the prices. Availability can change frequently especially around the holidays or long weekends during busy season. This data has a freshness requirement of every few minutes, so we use Spark structured streaming to ensure data is fresh when doing model prediction. 

# COMMAND ----------

# MAGIC %md <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/realtime/streaming.png"/>

# COMMAND ----------

from pyspark.sql.types import IntegerType, DoubleType, TimestampType, DateType, StringType, StructType, StructField
from pyspark.sql.functions import col

# Create schema 
destination_availability_schema = StructType([StructField("event_ts", TimestampType(), True),
                                             StructField("destination_id", IntegerType(), True),
                                             StructField("name", StringType(), True),
                                             StructField("booking_date", DateType(), True),
                                             StructField("price", DoubleType(), True),
                                             StructField("availability", IntegerType(), True),
                                             ])

destination_availability_log = (
  spark.readStream
  .format("delta")
  .option("maxFilesPerTrigger", 1000)
  .option("inferSchema", "true")
  .schema(destination_availability_schema)
  .json("/databricks-datasets/travel_recommendations_realtime/raw_travel_data/fs-demo_destination-availability_logs/json/*")
)

destination_availability_df = destination_availability_log.select(
  col("event_ts"),
  col("destination_id"),
  col("name"),
  col("booking_date"),
  col("price"),
  col("availability")
)
display(destination_availability_df)

# COMMAND ----------

fs.create_table(
    name=fs_table_destination_availability_features, 
    primary_keys=["destination_id", "booking_date"],
    timestamp_keys=["event_ts"],
    schema=destination_availability_schema,
    description="Destination Availability Features",
)

# Now write the data to the feature table in "merge" mode
query = fs.write_table(
    name=fs_table_destination_availability_features, 
    df=destination_availability_df,
    mode="merge",
    checkpoint_location=fs_destination_availability_features_delta_checkpoint
)

# COMMAND ----------

# MAGIC %md ## Compute realtime/on-demand features
# MAGIC 
# MAGIC User location is a context feature that is captured at the time of the query. This data is not known in advance, hence the derived feature. For example, user distance from destination can only be computed in realtime at the prediction time. MLflow `pyfunc` captures this feature transformation using a preprocessing code that manipulates the input data frame before passing it to the model at training and serving time. 

# COMMAND ----------

# MAGIC %md <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/realtime/pyfunc.png"/>

# COMMAND ----------

import geopy
import mlflow
import logging
import lightgbm as lgb
import pandas as pd
import geopy.distance as geopy_distance

from typing import Tuple


# Define the model class with on-demand computation model wrapper
class OnDemandComputationModelWrapper(mlflow.pyfunc.PythonModel):
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        try: 
            new_model_input = self._compute_ondemand_features(X_train)
            self.model = lgb.train(
              {"num_leaves": 32, "objective": "binary"}, 
              lgb.Dataset(new_model_input, label=y_train.values),
              5)
        except Exception as e:
            logging.error(e)
            
    def _distance(
        self,
        lon_lat_user: Tuple[float, float],
        lon_lat_destination: Tuple[float, float],
    ) -> float:
        """
        Wrapper call to calculate pair distance in miles
        ::lon_lat_user (longitude, latitude) tuple of user location
        ::lon_lat_destination (longitude, latitude) tuple of destination location
        """
        return geopy_distance.distance(
            geopy_distance.lonlat(*lon_lat_user),
            geopy_distance.lonlat(*lon_lat_destination),
        ).miles
        
    def _compute_ondemand_features(self, model_input: pd.DataFrame)->pd.DataFrame:
      try:
        # Fill NAs first
        loc_cols = ["user_longitude","user_latitude","longitude","latitude"]
        location_noNAs_pdf = model_input[loc_cols].fillna(model_input[loc_cols].median().to_dict())
        
        # Calculate distances
        model_input["distance"] = location_noNAs_pdf.apply(lambda x: self._distance((x[0], x[1]), (x[2], x[3])), axis=1)
        
        # Drop columns
        model_input.drop(columns=loc_cols)
        
      except Exception as e:
        logging.error(e)
        raise e
      return model_input

    def predict(self, context, model_input: pd.DataFrame)->pd.DataFrame:
        new_model_input = self._compute_ondemand_features(model_input)
        return  self.model.predict(new_model_input)

# COMMAND ----------

# MAGIC %md # Train a custom model with batch, on-demand and streaming features
# MAGIC 
# MAGIC The following uses all the features created above to train a ranking model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get ground-truth labels and on-demand input features

# COMMAND ----------

# Random split to define a training and inference set
training_labels_df = (
  vacation_purchase_df
    .where("ts < '2022-11-23'")
)

test_labels_df = (
  vacation_purchase_df
    .where("ts >= '2022-11-23'")
)
display(training_labels_df.limit(5))

# COMMAND ----------

# MAGIC %md ## Create a training set

# COMMAND ----------

# DBTITLE 1,Define Feature Lookups (for batch and streaming input features)
from databricks.feature_store.client import FeatureStoreClient
from databricks.feature_store.entities.feature_lookup import FeatureLookup

fs = FeatureStoreClient()

feature_lookups = [ # Grab all useful features from different feature store tables
    FeatureLookup(
        table_name=fs_table_destination_popularity_features, 
        lookup_key="destination_id",
        timestamp_lookup_key="ts"
    ),
    FeatureLookup(
        table_name=fs_table_destination_location_features,  
        lookup_key="destination_id",
        feature_names=["latitude", "longitude"]
    ),
    FeatureLookup(
        table_name=fs_table_user_features, 
        lookup_key="user_id",
        timestamp_lookup_key="ts",
        feature_names=["mean_price_7d"]
    ),
      FeatureLookup(
        table_name=fs_table_destination_availability_features, 
        lookup_key=["destination_id", "booking_date"],
        timestamp_lookup_key="ts",
        feature_names=["availability"]
    )
]

# COMMAND ----------

training_set = fs.create_training_set(
    training_labels_df,
    feature_lookups=feature_lookups,
    exclude_columns=['user_id', 'destination_id', 'ts', 'booking_date', 'clicked', 'price'],
    label='purchased',
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load as a Spark DataFrame

# COMMAND ----------

training_df = training_set.load_df()
display(training_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and log model to MLflow

# COMMAND ----------

# Record specific additional dependencies required by model serving
def get_conda_env():
  model_env = mlflow.pyfunc.get_default_conda_env()
  model_env["dependencies"][-1]["pip"] += [
    f"geopy=={geopy.__version__}",
    f"lightgbm=={lgb.__version__}",
    f"pandas=={pd.__version__}"
  ]
  return model_env

# COMMAND ----------

from sklearn.model_selection import train_test_split

with mlflow.start_run():
  
  # Split features and labels
  features_and_label = training_df.columns
 
  # Collect data into a Pandas array for training and testing
  data = training_df.toPandas()[features_and_label]
  train, test = train_test_split(data, random_state=123) # By default, 75%, 25% split
  X_train = train.drop(["purchased"], axis=1)
  X_test = test.drop(["purchased"], axis=1)
  y_train = train.purchased
  y_test = test.purchased

  # Fit
  pyfunc_model = OnDemandComputationModelWrapper()
  pyfunc_model.fit(X_train, y_train)
  
  # Log custom model to MLflow
  fs.log_model(
    artifact_path="model",
    model=pyfunc_model,
    flavor = mlflow.pyfunc,
    training_set=training_set,
    registered_model_name=model_name,
    conda_env=get_conda_env()
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch score test set

# COMMAND ----------

scored_df = fs.score_batch(
  f"models:/{model_name}/{get_latest_model_version(model_name)}",
  vacation_purchase_df,
  result_type="float"
)

# COMMAND ----------

display(scored_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Accuracy calculation

# COMMAND ----------

scored_df2 = scored_df.withColumnRenamed("prediction", "original_prediction")
scored_df2 = scored_df2.withColumn("prediction", (F.when(F.col("original_prediction") >= 0.2, True).otherwise(False))) # simply convert the original probability predictions to true or false
pd_scoring = scored_df2.select("purchased", "prediction").toPandas()

from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(pd_scoring["purchased"], pd_scoring["prediction"]))

# COMMAND ----------

# MAGIC %md # Publish feature tables to online store
# MAGIC 
# MAGIC In order to use the above models in a realtime scenario, you can publish the table to a online store. This allows the model to serve prediction queries with low-latency. If you are currently in the field eng demo workspaces `e2-field-eng-west` [AWS workspace](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#ml/dashboard) or `field-eng-east` [Azure workspace](https://adb-984752964297111.11.azuredatabricks.net/?o=984752964297111#), you do not need further actions to set up any secrets, just go ahead and run the code. 
# MAGIC 
# MAGIC You may check the published online feature tables in field eng accounts below after runing the next several cells with commands `fs.publish_table`:
# MAGIC * AWS dynamoDB: Open Okta-> Open AWS app-> Click DynamoDB-> Click Tables
# MAGIC * Azure cosmosDB [here](https://portal.azure.com/#@DataBricksInc.onmicrosoft.com/resource/subscriptions/3f2e4d32-8e8d-46d6-82bc-5bb8d962328b/resourcegroups/field-eng-east/providers/Microsoft.DocumentDB/databaseAccounts/field-demo/collectionSetting): Under Azure Portal `field-demo` account -> settings ->field_demos database
# MAGIC 
# MAGIC Otherwise, follow the instructions in "Work with online feature stores" ([AWS doc](https://docs.databricks.com/machine-learning/feature-store/online-feature-stores.html) | [Azure doc](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/online-feature-stores)) to store secrets in the Databricks secret manager with the scope below. 
# MAGIC 
# MAGIC Important note for Azure users: If you are doing this demo in our Azure workspace, please make sure you have installed [Azure Cosmos DB Apache Spark 3 OLTP Connector for API for NoSQL](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/sdk-java-spark-v3) (i.e. `com.azure.cosmos.spark:azure-cosmos-spark_3-2_2-12:4.17.2`) to your non-UC 11.3+ML `Shared Compute` policy  cluster before running this demo.

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
from databricks.feature_store.online_store_spec import AzureCosmosDBSpec, AmazonDynamoDBSpec
fs = FeatureStoreClient()

if cloud_name == "azure":

  account_uri = account_uri_field_demo

  destination_location_online_store_spec = AzureCosmosDBSpec(
    account_uri=account_uri,
    write_secret_prefix = write_secret_prefix,
    read_secret_prefix = read_secret_prefix,
    database_name = database_name_cosmosdb,
   container_name = fs_online_destination_location_features
  )

  destination_online_store_spec = AzureCosmosDBSpec(
    account_uri = account_uri,
    write_secret_prefix = write_secret_prefix,
    read_secret_prefix = read_secret_prefix,
    database_name = database_name_cosmosdb,
    container_name = fs_online_destination_popularity_features
  )

  destination_availability_online_store_spec = AzureCosmosDBSpec(
    account_uri = account_uri,
    write_secret_prefix = write_secret_prefix,
    read_secret_prefix = read_secret_prefix,
    database_name = database_name_cosmosdb,
    container_name = fs_online_destination_availability_features
  )

  user_online_store_spec = AzureCosmosDBSpec(
    account_uri = account_uri,
    write_secret_prefix = write_secret_prefix,
    read_secret_prefix = read_secret_prefix,
    database_name = database_name_cosmosdb,
    container_name = fs_online_user_features
  )
  
elif cloud_name == "aws":
  
  destination_location_online_store_spec = AmazonDynamoDBSpec(
    region=aws_region,
    write_secret_prefix=write_secret_prefix, 
    read_secret_prefix=read_secret_prefix,
    table_name = fs_online_destination_location_features
  )

  destination_online_store_spec = AmazonDynamoDBSpec(
    region=aws_region,
    write_secret_prefix=write_secret_prefix,
    read_secret_prefix=read_secret_prefix,
    table_name = fs_online_destination_popularity_features  
  )

  destination_availability_online_store_spec = AmazonDynamoDBSpec(
    region=aws_region,
    write_secret_prefix=write_secret_prefix,
    read_secret_prefix=read_secret_prefix,
    table_name = fs_online_destination_availability_features  
  )

  user_online_store_spec = AmazonDynamoDBSpec(
    region=aws_region,
    write_secret_prefix=write_secret_prefix,
    read_secret_prefix=read_secret_prefix,
    table_name = fs_online_user_features
  )

# COMMAND ----------

fs.publish_table(fs_table_user_features , user_online_store_spec) # Publish the offline fs table to online

fs.publish_table(fs_table_destination_location_features, destination_location_online_store_spec) 

fs.publish_table(fs_table_destination_popularity_features, destination_online_store_spec) 

# COMMAND ----------

# Push features to Online Store through Spark Structured streaming
query2 = fs.publish_table(
    fs_table_destination_availability_features,
    destination_availability_online_store_spec,
    streaming=True, 
    checkpoint_location=fs_destination_availability_features_online_checkpoint
)

# COMMAND ----------

# MAGIC %md # Realtime model inference

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enable model inference via API call
# MAGIC 
# MAGIC After calling `log_model`, a new version of the model is saved. To provision a serving endpoint, follow the steps below.
# MAGIC 
# MAGIC 1. Click **Serving** in the left sidebar. If you don't see it, switch to the Machine Learning Persona ([AWS](https://docs.databricks.com/workspace/index.html#use-the-sidebar)|[Azure](https://docs.microsoft.com/azure/databricks//workspace/index#use-the-sidebar)).
# MAGIC 2. Enable serving for your model. See the Databricks documentation for details ([AWS](https://docs.databricks.com/machine-learning/model-inference/serverless/create-manage-serverless-endpoints.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-inference/serverless/create-manage-serverless-endpoints)).
# MAGIC 
# MAGIC The code below automatically creates a model serving endpoint for you.

# COMMAND ----------

# Provide both a token for the API, which can be obtained from the notebook.

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
# With the token, we can create our authorization header for our subsequent REST calls
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
  }

# Next we need an enpoint at which to execute our request which we can get from the Notebook's tags collection
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()

# This object comes from the Java CM - Convert the Java Map opject to a Python dictionary
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)

# Lastly, extract the databricks instance (domain name) from the dictionary
instance = tags["browserHostName"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set up configurations for Serverless model serving endpoint:
# MAGIC * Create the serving endpoint if it does not exist yet
# MAGIC * Or update the configuration of the model serving endpoint if it already exists


# COMMAND ----------

import requests

my_json = {
  "name": model_serving_endpoint_name,
  "config": {
   "served_models": [{
     "model_name": model_name,
     "model_version": get_latest_model_version(model_name=model_name),
     "workload_size": "Small",
     "scale_to_zero_enabled": True
   }]
 }
}

def func_create_endpoint(model_serving_endpoint_name: str):
  #get endpoint status
  endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
  url = f"{endpoint_url}/{model_serving_endpoint_name}"
  r = requests.get(url, headers=headers)
  if "RESOURCE_DOES_NOT_EXIST" in r.text:  
    print("Creating this new endpoint: ", f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations")
    re = requests.post(endpoint_url, headers=headers, json=my_json)
   
  else:
    new_model_version = (my_json['config'])['served_models'][0]['model_version']
    print("This endpoint existed previously! We are updating it to a new config with new model version: ", new_model_version)
    # update config
    url = f"{endpoint_url}/{model_serving_endpoint_name}/config"
    re = requests.put(url, headers=headers, json=my_json['config']) 
    # wait till new config file in place
    import time,json
    #get endpoint status
    url = f"https://{instance}/api/2.0/serving-endpoints/{model_serving_endpoint_name}"
    retry = True
    total_wait = 0
    while retry:
      r = requests.get(url, headers=headers)
      assert r.status_code == 200, f"Expected an HTTP 200 response when accessing endpoint info, received {r.status_code}"
      endpoint = json.loads(r.text)
      if "pending_config" in endpoint.keys():
        seconds = 10
        print("New config still pending")
        if total_wait < 6000:
          #if less the 10 mins waiting, keep waiting
          print(f"Wait for {seconds} seconds")
          print(f"Total waiting time so far: {total_wait} seconds")
          time.sleep(10)
          total_wait += seconds
        else:
          print(f"Stopping,  waited for {total_wait} seconds")
          retry = False  
      else:
        print("New config in place now!")
        retry = False
  assert re.status_code == 200, f"Expected an HTTP 200 response, received {re.status_code}"
  
  
func_create_endpoint(model_serving_endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wait for endpoint to be ready

# COMMAND ----------

import time, mlflow
def wait_for_endpoint():
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    while True:
        url =  f"{endpoint_url}/{model_serving_endpoint_name}" 
        response = requests.get(url, headers=headers)
        assert response.status_code == 200, f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"

        status = response.json().get("state", {}).get("ready", {})
        if status == "READY": print(status); print("-"*80); return
        else: print(f"Endpoint not ready ({status}), waiting 10 seconds"); time.sleep(10) # Wait 10 seconds
        
api_url = mlflow.utils.databricks_utils.get_webapp_url()
wait_for_endpoint()
# Give the system just a couple extra seconds to transition
time.sleep(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Send payloads via REST call
# MAGIC 
# MAGIC With Databricks's Serverless Model Serving, the endpoint takes a different score format.
# MAGIC You can see that users in New York can see high scores for Florida, whereas usersers in California can see high scores for Hawaii.
# MAGIC 
# MAGIC ```
# MAGIC {
# MAGIC   "dataframe_records": [
# MAGIC     {"user_id": 4, "booking_date": "2022-12-22", "destination_id": 16, "user_latitude": 40.71277, "user_longitude": -74.005974}, 
# MAGIC     {"user_id": 39, "booking_date": "2022-12-22", "destination_id": 1, "user_latitude": 37.77493, "user_longitude": -122.41942}
# MAGIC   ]
# MAGIC }
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Create wrapper function
import requests

def score_model(data_json: dict):
    url =  f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations"
    response = requests.request(method="POST", headers=headers, url=url, json=data_json)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    return response.json()
  
payload_json = {
  "dataframe_records": [
    # Users in New York, see high scores for Florida 
    {"user_id": 4, "booking_date": "2022-12-22", "destination_id": 16, "user_latitude": 40.71277, "user_longitude": -74.005974}, 
    # Users in California, see high scores for Hawaii 
    {"user_id": 39, "booking_date": "2022-12-22", "destination_id": 1, "user_latitude": 37.77493, "user_longitude": -122.41942} 
  ]
}

# COMMAND ----------

print(score_model(payload_json))

# COMMAND ----------

# MAGIC %md
# MAGIC Once your serving endpoint is ready, your previous cell's `score_model` code should give you the model inference result. 
# MAGIC 
# MAGIC (**OPTIONAL**) To visualize the UI for model serving or to manually create a model serving endpoint, you could click the **"Serving"** tab on the left, 
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/product/feature_store/notebook4_ff_model_serving_screenshot2_1.png" alt="step12" width="1500"/>
# MAGIC and then select your model to enable the model serving/check the enabled serving model as shown in screenshot below:
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/product/feature_store/notebook4_ff_model_serving_screenshot2_2.png" alt="step12" width="1500"/>
# MAGIC  You can try the "Query endpoint" option on the upper right corner of the screen to manually input the payload_json, and check the inference result. 
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/product/feature_store/notebook4_ff_model_serving_screenshot2_3.png" alt="step12" width="1500"/>

# COMMAND ----------

# MAGIC %md ## Cleanup 
# MAGIC Please run the cell below after your demo. This cleanup code
# MAGIC 1. stops the serving endpoint by visiting models tab or serving tab on the left.  
# MAGIC 2. run the `cleanup` function for dropping the created demo database, the offline/online feature tables/containers, models etc. 
# MAGIC 3. stops the streaming writes to feature table and online store.

# COMMAND ----------

cleanup(query, query2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stop the entire notebook
# MAGIC <img src="https://files.training.databricks.com/images/icon_warn_24.png"/> Please also click the `Interrupt` button on top right to **stop the entire notebook** after finish running the above cleanup cell.

# COMMAND ----------


