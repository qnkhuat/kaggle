import gc
import os
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import json
import time
from ast import literal_eval



def load_df(file_name = 'train_v2.csv', nrows = None):
    """Read csv and convert json columns."""


    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    df = pd.read_csv('../input/{}'.format(file_name),
                     parse_dates=['date'],
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        
    # Normalize customDimensions
    df['customDimensions']=df['customDimensions'].apply(literal_eval)
    df['customDimensions']=df['customDimensions'].str[0]
    df['customDimensions']=df['customDimensions'].apply(lambda x: {'index':np.NaN,'value':np.NaN} if pd.isnull(x) else x)

    column_as_df = json_normalize(df['customDimensions'])
    column_as_df.columns = [f"customDimensions_{subcolumn}" for subcolumn in column_as_df.columns]
    df = df.drop('customDimensions', axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df


def to_na(df):
    # Each type of columns that need to replace with the right na values
    to_NA_cols = ['trafficSource_adContent','trafficSource_adwordsClickInfo.adNetworkType',
                'trafficSource_adwordsClickInfo.slot','trafficSource_adwordsClickInfo.gclId',
                'trafficSource_keyword','trafficSource_referralPath']

    to_0_cols = ['totals_transactionRevenue','trafficSource_adwordsClickInfo.page','totals_sessionQualityDim','totals_bounces',
                 'totals_timeOnSite','totals_newVisits','totals_pageviews']
    to_0_cols_test = ['trafficSource_adwordsClickInfo.page','totals_sessionQualityDim','totals_bounces',
                 'totals_timeOnSite','totals_newVisits','totals_pageviews']

    to_true_cols = ['trafficSource_adwordsClickInfo.isVideoAd']
    to_false_cols = ['trafficSource_isTrueDirect']
    
    
    df[to_NA_cols] = df[to_NA_cols].fillna('NA')
    df[to_0_cols] = df[to_0_cols].fillna(0)
    df[to_true_cols] = df[to_true_cols].fillna(True)
    df[to_false_cols] = df[to_false_cols].fillna(False)
    
    return df
    
    
def encode_date(df):
    fld = pd.to_datetime(df['date'], infer_datetime_format=True)
    
    attrs = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
        'Is_month_end', 'Is_month_start', 'Is_quarter_end', 
        'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        
    for attr in attrs:
        df['Date_'+attr] = getattr(fld.dt,attr.lower())
        
    return df

    
def weird_na(df):
    cols_to_replace = {
        'socialEngagementType' : 'Not Socially Engaged',
        'device_browserSize' : 'not available in demo dataset', 
        'device_flashVersion' : 'not available in demo dataset', 
        'device_browserVersion' : 'not available in demo dataset', 
        'device_language' : 'not available in demo dataset',
        'device_mobileDeviceBranding' : 'not available in demo dataset',
        'device_mobileDeviceInfo' : 'not available in demo dataset',
        'device_mobileDeviceMarketingName' : 'not available in demo dataset',
        'device_mobileDeviceModel' : 'not available in demo dataset',
        'device_mobileInputSelector' : 'not available in demo dataset',
        'device_operatingSystemVersion' : 'not available in demo dataset',
        'device_screenColors' : 'not available in demo dataset',
        'device_screenResolution' : 'not available in demo dataset',
        'geoNetwork_city' : 'not available in demo dataset',
        'geoNetwork_cityId' : 'not available in demo dataset',
        'geoNetwork_latitude' : 'not available in demo dataset',
        'geoNetwork_longitude' : 'not available in demo dataset',
        'geoNetwork_metro' : ['not available in demo dataset', '(not set)'], 
        'geoNetwork_networkDomain' : 'unknown.unknown', 
        'geoNetwork_networkLocation' : 'not available in demo dataset',
        'geoNetwork_region' : 'not available in demo dataset',
        'trafficSource_adwordsClickInfo.criteriaParameters' : 'not available in demo dataset',
        'trafficSource_campaign' : '(not set)', 
        'trafficSource_keyword' : '(not provided)'
    }
    df = df.replace(cols_to_replace,'NA')
    return df

def pipeline(nrows=None):
    print('Load data')
    df_train = load_df('train_v2.csv',nrows)
    df_test = load_df('test_v2.csv',nrows)
    
    print('Raw data')
    print(f'Train {df_train.shape}')
    print(f'Test {df_test.shape}')
    
    print('Handle Na values')
    df_train = weird_na(df_train)
    df_test  = weird_na(df_test)
    
    df_train = to_na(df_train)
    df_test  = to_na(df_test)
    
    print('Encode date data')
    df_train = encode_date(df_train)
    df_test  = encode_date(df_test)
    
    print('Drop constant columns')
    const_col = []
    for col in df_train.columns:
        if df_train[col].nunique() == 1 and df_train[col].isnull().sum()==0 :
            const_col.append(col)
            
    df_train.drop(const_col,axis=1,inplace=True)
    df_test.drop(const_col,axis=1,inplace=True)
    
    print('Ouput data')
    print(f'Train {df_train.shape}')
    print(f'Test  {df_test.shape}')
    
    return df_train,df_test
    
    
    
df_train,df_test = pipeline()

df_train.to_hdf('train_flatten.h5','data')
df_test.to_hdf('test_flatten.h5','data')
