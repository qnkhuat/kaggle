https://www.kaggle.com/c/ga-customer-revenue-prediction



Data note : 

The data has been updated with version2 with 30gbs of data. So downloaded a pre-convert data at : https://www.kaggle.com/utmost/files-without-hits-and-flatten-json-fields/output

Named \_v2\_flat in /data


Run order: 
1. files-without-hits-and-flatten-json-fields.py - to convert the data from 30gb to 1gb . Get the \_flat.csv data
2. clean raw data.ipynb to get a na filled data
