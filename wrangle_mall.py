######################### IMPORTS #########################

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# import preprocessing
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

from env import user, password, host



######################### ACQUIRE DATA #########################

def get_db_url(db):

    '''
    This function calls the username, password, and host from env file and provides database argument for SQL
    '''

    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
#------------------------- MALL DATA FROM SQL -------------------------

def new_mall_data():

    '''
    This function reads the mall data from the Codeup database based on defined query argument and returns a DataFrame.
    '''

    # Create SQL query.
    query = "SELECT * FROM customers"
    
    # Read in DataFrame from Codeup db using defined arguments.
    df = pd.read_sql(query, get_db_url('mall_customers'))

    return df

def get_mall_data():

    '''
    This function checks for a local file and reads it in as a Datafile.  
    If the csv file does not exist, it calls the new_mall_data function then writes the data to a csv file.
    '''

    # Checks for csv file existence
    if os.path.isfile('mall_data.csv'):
        
        # If csv file exists, reads in data from the csv file.
        df = pd.read_csv('mall_data.csv', index_col=0)
        
    else:
        
        # If csv file does not exist, uses new_wrangle_zillow_2017 function to read fresh data from telco db into a DataFrame
        df = new_mall_data()
        
        # Cache data into a new csv file
        df.to_csv('mall_data.csv')
        
    return pd.read_csv('mall_data.csv', index_col=0)

#------------------------ ONE WRANGLE FILE TO RUN THEM ALL ------------------------

def wrangle_mall_data():
    """
    This function is used to run all Acquire and Prepare functions.
    """
    df = get_mall_data()
    df = clean_mall_data(df)
    return df



######################### PREPARE DATA #########################

def clean_mall_data(df):
    df = encode(df, 'gender')
    return df

def encode(df, column_name):
    """Use pandas dummies to pivot features with more than two string values
    into multiple columns with binary int values that can be read as boolean
    drop_first = False in draft for human readability; Final will have it set to True.
    """
    
    dummy_df = pd.get_dummies(data=df[[column_name]], drop_first=False)
    
    # Assign dummies to DataFrame
    df = pd.concat([df, dummy_df], axis=1)
    
    # Drop dummy Columns 
    df = df.drop(columns=column_name)
    
    return df



######################### EXPLORE DATA #########################

def check_column_outliers(df, column_name):
    """
    Creates a DataFrame to show outlier records based on  InterQuartile Range (IQR)
    """
    
    name = column_name
    name_q1, name_q3 = df[column_name].quantile([0.25, 0.75])
    
    name_iqr = name_q3 - name_q1
    
    name_upper = name_q3 + (name_iqr * 1.5)
    name_lower = name_q1 - (name_iqr * 1.5)
    
    return df[df[column_name] > name_upper]

def check_whitespace(df):
    """
    This Function checks the DataFrame argument for whitespace,
    replaces any that exist with NaN, then returns report.
    
    Imports Needed:
    import numpy as np
    """
    
    # Calculate count of Whitespace
    row_count = df.shape[0]
    column_list = df.columns
    row_value_count = df[column_list].value_counts().sum()
    whitespace_count = row_count - row_value_count

    # Collect Row count of isnull before cleaning whiitespace    
    isnull_before = df.dropna().shape[0]
    
    # Clean the Whitespace
    if whitespace_count > 0:
        df = df.replace(r'^\s*$', np.NaN, regex=True)

    # Collect Row count of isnull after cleaning whiitespace    
    isnull_after = df.dropna().shape[0]
    
    # Report of Whitespace affect on NULL/NaN row count
    print (f'Cleaning {whitespace_count} Whitespace characters found and replaced with NULL/NaN.')
    print(f'Resulting in {isnull_before - isnull_after} additional rows containing NULL/NaN.')
    print()
    print()
    
    # set temporary conditions for this instance of code    
    with pd.option_context('display.max_rows', None):
        # print count of nulls by column
        print('COUNT OF NULL/NaN PER COLUMN:')
        print(df.isnull().sum().sort_values(ascending=False))

def null_stats(df):
    """
    This Function will display the DataFrame row count, 
    the NULL/NaN row count, and the 
    percent of rows that would be dropped.
    """

    print('COUNT OF NULL/NaN PER COLUMN:')
    # set temporary conditions for this instance of code
    with pd.option_context('display.max_rows', None):
        # print count of nulls by column
        print(df.isnull().sum().sort_values(ascending=False))
    print()
    print(f'     DataFrame Row Count: {df.shape[0]}')
    print(f'      NULL/NaN Row Count: {df.shape[0] - df.dropna().shape[0]}')
    
    if df.shape[0] == df.dropna().shape[0]:
        print()
        print('Row Counts are the same')
        print('Drop NULL/NaN cannot be run')
    
    elif df.dropna().shape[0] == 0:
        print()
        print('This will remove all records from your DataFrame')
        print('Drop NULL/NaN cannot be run')
    
    else:
        print()
        print(f'  DataFrame Percent kept: {round((df.dropna().shape[0] / df.shape[0]), 4)}')
        print(f'NULL/NaN Percent dropped: {round(1 - (df.dropna().shape[0] / df.shape[0]), 4)}')




    ######################### SPLIT DATA #########################

def split(df, stratify=False, target=None):
    """
    This Function splits the DataFrame into train, validate, and test
    then prints a graphic representation and a mini report showing the shape of the original DataFrame
    compared to the shape of the train, validate, and test DataFrames.
    """
    
    # Do NOT stratify on continuous data
    if stratify:
        # Split df into train and test using sklearn
        train, test = train_test_split(df, test_size=.2, random_state=1992, stratify=df[target])
        # Split train_df into train and validate using sklearn
        train, validate = train_test_split(train, test_size=.25, random_state=1992, stratify=df[target])
        
    else:
        train, test = train_test_split(df, test_size=.2, random_state=1992)
        train, validate = train_test_split(train, test_size=.25, random_state=1992)
    
    # reset index for train validate and test
    train.reset_index(drop=True, inplace=True)
    validate.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    train_prcnt = round((train.shape[0] / df.shape[0]), 2)*100
    validate_prcnt = round((validate.shape[0] / df.shape[0]), 2)*100
    test_prcnt = round((test.shape[0] / df.shape[0]), 2)*100
    
    print('________________________________________________________________')
    print('|                              DF                              |')
    print('|--------------------:--------------------:--------------------|')
    print('|        Train       |      Validate      |        Test        |')
    print(':--------------------------------------------------------------:')
    print()
    print()
    print(f'Prepared df: {df.shape}')
    print()
    print(f'      Train: {train.shape} - {train_prcnt}%')
    print(f'   Validate: {validate.shape} - {validate_prcnt}%')
    print(f'       Test: {test.shape} - {test_prcnt}%')
 
    
    return train, validate, test


def Xy_split(feature_cols, target, train, validate, test):
    """
    This function will split the train, validate, and test data by the Feature Columns selected and the Target.
    
    Imports Needed:
    from sklearn.model_selection import train_test_split
    
    Arguments Taken:
       feature_cols: list['1','2','3'] the feature columns you want to run your model against.
             target: list the target feature that you will try to predict
              train: Assign the name of your train DataFrame
           validate: Assign the name of your validate DataFrame
               test: Assign the name of your test DataFrame
    """
    
    print('_______________________________________________________________')
    print('|                              DF                             |')
    print('|-------------------:-------------------:---------------------|')
    print('|       Train       |       Validate    |          Test       |')
    print('|-------------------:-------------------:---------------------|')
    print('| x_train | y_train |   x_val  |  y_val |   x_test  |  y_test |')
    print(':-------------------------------------------------------------:')
    
    X_train, y_train = train[feature_cols], train[target]
    X_validate, y_validate = validate[feature_cols], validate[target]
    X_test, y_test = test[feature_cols], test[target]

    print()
    print()
    print(f'   X_train: {X_train.shape}   {X_train.columns}')
    print(f'   y_train: {y_train.shape}     Index({target})')
    print()
    print(f'X_validate: {X_validate.shape}   {X_validate.columns}')
    print(f'y_validate: {y_validate.shape}     Index({target})')
    print()
    print(f'    X_test: {X_test.shape}   {X_test.columns}')
    print(f'    y_test: {y_test.shape}     Index({target})')
    
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test
    # When I run this it returns the OUT but  gives me a name not defined error when I try to call Variables.
    # NameError: name 'X_train' is not defined
    # NameError: name 'y_train' is not defined



######################### SCALE SPLIT #########################

def scale_data(train, 
               validate, 
               test, 
               columns_to_scale,
               scaler,
               return_scaler = False):
    
    """
    Scales the 3 data splits. 
    Takes in train, validate, and test data 
    splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    
    Imports Needed:
    from sklearn.preprocessing import MinMaxScaler 
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import QuantileTransformer
    
    Arguments Taken:
               train = Assign the train DataFrame
            validate = Assign the validate DataFrame 
                test = Assign the test DataFrame
    columns_to_scale = Assign the Columns that you want to scale
              scaler = Assign the scaler to use MinMaxScaler(),
                                                StandardScaler(), 
                                                RobustScaler(), or 
                                                QuantileTransformer()
       return_scaler = False by default and will not return scaler data
                       True will return the scaler data before displaying the _scaled data
    """
    
    # make copies of our original data so we dont corrupt original split
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    # fit the scaled data
    scaler.fit(train[columns_to_scale])
    
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    
    else:
        return train_scaled, validate_scaled, test_scaled


    
######################### DATA SCALE VISUALIZATION #########################

# Function Stolen from Codeup Instructor Andrew King
def visualize_scaler(scaler, df, columns_to_scale, bins=10):
    """
    This Function takes input arguments, 
    creates a copy of the df argument, 
    scales it according to the scaler argument, 
    then displays subplots of the columns_to_scale argument 
    before and after scaling.
    """    

    fig, axs = plt.subplots(len(columns_to_scale), 2, figsize=(16,9))
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    for (ax1, ax2), col in zip(axs, columns_to_scale):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    plt.tight_layout()
    #return df_scaled.head().T
    #return fig, axs
