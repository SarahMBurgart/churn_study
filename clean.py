import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, precision_score, recall_score

from datetime import datetime

def active_flag(df, last_trip_date_columns):
    # calucate preceding data
    df['pulled_data'] = datetime.strptime('2014-7-1', "%Y-%m-%d") # temp pulled_data columns
    df['preceding_data'] = df['pulled_data'] - pd.to_datetime(df[last_trip_date_columns])
    df['preceding_data'] = df['preceding_data'].astype(str).str[:-24].astype(int)
    
    
    # if active or not
    df['active_flag'] = np.where(df['preceding_data'] <= 30, 1, 0)
    
    # drop temp columns
    df.drop('pulled_data',axis=1)
    return df




def clean_data(df):
    drop_columns = ['pulled_data', 'last_trip_date','signup_date','phone', 'preceding_data']
    df['signup_period'] = df['pulled_data'] - pd.to_datetime(df["signup_date"])
    df['signup_period'] = df['signup_period'].astype(str).str[:-24].astype(int)
    df.replace("King's Landing", 1, inplace=True)
    df.replace("Astapor", 2, inplace=True)
    df.replace("Winterfell", 3, inplace=True)
    df.luxury_car_user.replace(True,1)
    df.luxury_car_user.replace(False, 0)
    df = df.fillna(df.mean())
    return df.drop(drop_columns,axis=1)

