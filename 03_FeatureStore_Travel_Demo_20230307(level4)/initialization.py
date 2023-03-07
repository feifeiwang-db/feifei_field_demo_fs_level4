# Databricks notebook source
# MAGIC %pip install geopy

# COMMAND ----------

from databricks import feature_store
import boto3
import requests

# Below are initialization related functions 
def get_cloud_name():
  return spark.conf.get("spark.databricks.clusterUsageTags.cloudProvider").lower()

def getUsername() -> str: # Get the user's username
  return (
    spark
      .sql("SELECT current_user()")
      .first()[0]
      .lower()
      .split("@")[0]
      .replace(".", "_")
  )
  
def get_request_headers():
  return {
    "Authorization": f"""Bearer {
      dbutils
        .notebook
        .entry_point
        .getDbutils()
        .notebook()
        .getContext()
        .apiToken()
        .getOrElse(None)
      }"""
  }

def get_instance():
  # Next we need an enpoint at which to execute our request which we can get from the Notebook's tags collection
  java_tags =(
    dbutils
      .notebook
      .entry_point
      .getDbutils()
      .notebook()
      .getContext()
      .tags()
   )
    # This ojbect comes from the Java CM - Convert the Java Map opject to a Python dictionary
  tags = (
    sc
      ._jvm.scala
      .collection
      .JavaConversions
      .mapAsJavaMap(java_tags)
  )
    # Lastly, extract the databricks instance (domain name) from the dictionary
  return tags["browserHostName"]  

def func_print_variable (variable):
  '''
  print out variable's string name and its variable value
  '''
  str_variable = [name for name in globals() if globals()[name] is variable][0]
  print(f"\t", "\"%s\""% str_variable, "populated as: ", variable)
  
def func_print_all_variables (list_variables):
  for var in list_variables:
    func_print_variable (var)

def func_todaysdate():
  '''
  return todays date in string format in EST time zone
  '''
  from datetime import datetime, timedelta
  now_EST = datetime.today() - timedelta(hours=5)
  todaysdate = now_EST.strftime('%Y_%m_%d')
  todaysdate = todaysdate.replace("_", "")
  return todaysdate

# COMMAND ----------

# DBTITLE 1,Variables for initialization
cloud_name = get_cloud_name()
suffix = func_todaysdate()
cleaned_username = getUsername()
database_name = f"travel_demo_{cleaned_username}_{suffix}"
fs_table_name_destinations = f"travel_demo_{cleaned_username}.destination_popularity_features_{suffix}"
model_name = f"model_travel_demo_{cleaned_username}_{suffix}"
model_serving_endpoint_name = f"endpoint_travel_demo_{cleaned_username}_{suffix}"


read_secret_prefix="field-all-users-feature-store-example-read/field-eng"
write_secret_prefix="field-all-users-feature-store-example-write/field-eng"
account_uri_field_demo="https://field-demo.documents.azure.com:443/"
database_name_cosmosdb="field_demos"

resquest_headers = get_request_headers()
current_instance = get_instance()
aws_region = "us-west-2"



# COMMAND ----------

# Below are cleanup related functions   

def delete_dynamodb_table(table_name):
  client = boto3.client(
    'dynamodb', 
    aws_access_key_id=dbutils.secrets.get(scope="field-all-users-feature-store-example-write", key="dynamo-access-key-id"),
    aws_secret_access_key=dbutils.secrets.get(scope="field-all-users-feature-store-example-write", key="dynamo-secret-access-key"),
    region_name="us-west-2")
  client.delete_table(
    TableName=table_name
  )
  waiter = client.get_waiter('table_not_exists')
  waiter.wait(
    TableName=table_name
  )
  print(f"table: '{table_name}' was deleted")

from azure.cosmos import cosmos_client
import azure.cosmos.exceptions as exceptions
def delete_Container(db, id):
    try:
        db.delete_container(id)
        print('Container with id \'{0}\' was deleted'.format(id))
    except exceptions.CosmosResourceNotFoundError:
        print('A container with id \'{0}\' does not exist'.format(id))
          
def delete_cosmosdb_container(container_name, account_uri_field_demo):
  URL = account_uri_field_demo
  KEY = dbutils.secrets.get(scope="feature-store-example-write", key="field-eng-authorization-key") ###Amine El Lou is maintaining those, hard code here
  DATABASE_NAME = "field_demos" ###Amine El Lou is maintaining those, hard code here
  CONTAINER_NAME = container_name
  client = cosmos_client.CosmosClient(URL, credential=KEY)
  database = client.get_database_client(DATABASE_NAME)
  container = database.get_container_client(CONTAINER_NAME)
  delete_Container(database, container)
        
def func_delete_model_serving_endpoint(model_serving_endpoint_name):
  endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
  url =  f"{endpoint_url}/{model_serving_endpoint_name}" 
  response = requests.delete(url, headers=headers)
  if response.status_code != 200:
    raise Exception(f"Request failed with status {response.status_code}, {response.text}")
  else:
    print(model_serving_endpoint_name, "endpoint is deleted!")
  #return response.json()
  
def delete_model(name):
  client = MlflowClient()
  active_models = [
    model.version 
    for model 
    in  client.get_latest_versions(name) 
    if model.current_stage not in ['Archived']
  ]

  for model_version in active_models:
    client.transition_model_version_stage(
      name=name, 
      version=model_version, 
      stage='Archived'
    )

  client.delete_registered_model(name=name) 
  
  print(f"model: {name} was deleted")  

def drop_database(name):  
  spark.sql(f"DROP DATABASE {name}")
  print(f"Database: '{name}' was dropped")

# COMMAND ----------

# DBTITLE 1,Variables that will be cleaned up, and print all populated vars
fs_table_user_features = f"{database_name}.user_features_{suffix}"
fs_table_destination_popularity_features = f"{database_name}.destination_popularity_features_{suffix}"
fs_table_destination_location_features = f"{database_name}.destination_location_features_{suffix}"
fs_table_destination_availability_features = f"{database_name}.destination_availability_features_{suffix}"
fs_destination_availability_features_delta_checkpoint = f"/Shared/fs_realtime/checkpoints/{cleaned_username}/destination_availability_features_delta_{suffix}/"
fs_online_user_features = f"feature_store_travel_demo_user_features_{cleaned_username}_{suffix}" # Due to set policy, this online table name must begin with "feature_store"
fs_online_destination_popularity_features =  f"feature_store_travel_demo_destination_popularity_features_{cleaned_username}_{suffix}"
fs_online_destination_location_features = f"feature_store_travel_demo_destination_location_features_{cleaned_username}_{suffix}"
fs_online_destination_availability_features = f"feature_store_travel_demo_destination_availability_{cleaned_username}_{suffix}"
fs_destination_availability_features_online_checkpoint = f"/Shared/fs_realtime/checkpoints/{cleaned_username}/destination_availability_features_online_{suffix}/"

print("populating demo variables: ")
list_vars = [
    cloud_name,
    suffix, 
    cleaned_username, 
    database_name, 
    fs_table_name_destinations, 
    model_name, 
    model_serving_endpoint_name,
    write_secret_prefix, 
    read_secret_prefix, 
    account_uri_field_demo,
    database_name_cosmosdb,
    resquest_headers, 
    current_instance, 
    aws_region, 
 
    fs_table_user_features, 
    fs_table_destination_popularity_features, 
    fs_table_destination_location_features, 
    fs_table_destination_availability_features, 
    fs_destination_availability_features_delta_checkpoint, 
    fs_online_user_features, 
    fs_online_destination_popularity_features, 
    fs_online_destination_location_features, 
    fs_online_destination_availability_features, 
    fs_destination_availability_features_online_checkpoint
]
func_print_all_variables(list_vars)

# COMMAND ----------

def func_stop_streaming_query(query):
  import time
  while len(query.recentProgress) == 0 or query.status["isDataAvailable"]:
    print("waiting for stream to process all data")
    print(query.status)
    time.sleep(10)
  query.stop() 
  print("Just stopped one of the streaming queries.")
  

from mlflow.tracking.client import MlflowClient
def get_latest_model_version(model_name: str):
  client = MlflowClient()
  models = client.get_latest_versions(model_name, stages=["None"])
  for m in models:
    new_model_version = m.version
  return new_model_version


def cleanup(query, query2):
  func_stop_streaming_query(query)
  func_stop_streaming_query(query2)
  
  func_delete_model_serving_endpoint(model_serving_endpoint_name)

  fs_table_names = [
    fs_table_destination_popularity_features,
    fs_table_destination_location_features,
    fs_table_destination_availability_features,
    fs_table_user_features
  ]
  fs_online_table_names = [
    fs_online_destination_popularity_features,
    fs_online_destination_location_features,
    fs_online_destination_availability_features,
    fs_online_user_features
  ]
  db_name = database_name
  delta_checkpoint = fs_destination_availability_features_delta_checkpoint
  online_checkpoint = fs_destination_availability_features_online_checkpoint
  model = model_name

  fs = feature_store.FeatureStoreClient()

  for table_name in fs_table_names:
    try:
      fs.drop_table(name=table_name)
      print(table_name, "is dropped!")
    except Exception as ex:
      print(ex)

  try:
    drop_database(db_name)
  except Exception as ex:
    print(ex)


  for container_name in fs_online_table_names:
    try:
      print("currentlly working on this online table/container dropping: ", container_name)
      if cloud_name == "azure":
        delete_cosmosdb_container(container_name, account_uri_field_demo)
        print("\n")
      elif cloud_name == "aws":
        delete_dynamodb_table(table_name=container_name)
        print("\n")
    except Exception as ex:
      print(ex)

  try:    
    dbutils.fs.rm(
      delta_checkpoint, True
    )
  except Exception as ex:
      print(ex) 

  try:    
    dbutils.fs.rm(
      online_checkpoint, True
    )
  except Exception as ex:
    print(ex)

  try:
    delete_model(model_name)
  except Exception as ex:
    print(ex)  
    
print("all cells are finished now !!!")  