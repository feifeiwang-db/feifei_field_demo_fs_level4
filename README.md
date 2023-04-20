# feifei_field_demo_fs_level4
This is a temporary repo for users to test my level4 demo. Official demo will eventually be published to field-demo repo and dbdemos [here](https://www.dbdemos.ai/demo.html?demoName=feature-store).

There are 2 notebooks, initialization notebook and main notebook. Please connect code to repo, and run the main demo notebook in either `e2-field-eng-west` [AWS workspace](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#ml/dashboard) or `field-eng-east` [Azure workspace](https://adb-984752964297111.11.azuredatabricks.net/?o=984752964297111#), since the secret scopes for accessing dynamoDB/cosmosDB are set up for all field eng users. 

Please **READ** the instructions in the main notebook, especially about cluster settings. DBR versions 11.3ML,12.1ML, and 12.2ML clusters have been tested for now. Azure users may need additional spark connector installations steps and use a shared-compute non-UC cluster as shown in the notebook. 

Please provide feedback to this google sheet: https://docs.google.com/spreadsheets/d/19WGJjAD426GUkUDgOqBbChLDkJstUWh9F3uq0kxbCNk/edit#gid=0 

Thank you for testing!
