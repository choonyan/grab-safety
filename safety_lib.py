# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 08:51:00 2019

@author: nxf47752
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#using IQR to filter outlier (accuracy)
def reject_outliers_IQR(data):
    accuracy_data = data['Accuracy']
    Q1 = accuracy_data.quantile(0.25)
    Q3 = accuracy_data.quantile(0.75)
    IQR = Q3 - Q1
    # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
    filtered_value = accuracy_data[(accuracy_data > Q1-1.5*IQR) & (accuracy_data < Q3+1.5*IQR)]
    min_value = min(filtered_value)
    max_value = max(filtered_value)
    result = data[(data['Accuracy']<=max_value) & (data['Accuracy']>= min_value)]
    return result

#calculate different of bearing
def find_bearing_diff(data):
    data = data.sort_values(by=['bookingID','second'])
    data['Bearing_prev'] = data['Bearing'].shift(periods = 1)
    data['Bearing_diff'] = data['Bearing'] - data['Bearing_prev']
    data['Bearing_diff'] = data['Bearing'] - data['Bearing_prev']
    data['Bearing_diff'] = data['Bearing_diff'].fillna(value=0)
    data.reset_index(inplace=True)
    result = data.drop(['index','Bearing_prev'],axis=1)
    result.loc[(result['Bearing_diff']>180),'Bearing_diff'] = result['Bearing_diff']-360
    result.loc[(result['Bearing_diff']<-180),'Bearing_diff'] = result['Bearing_diff']+360
    return result

#use mean to check for signal offset during speed and bearing_diff = 0
def offset_signal(data, column):
    data_ref = data[(data['Speed']<=1) & (data['Bearing_diff']==0)]
    for cn, cid in enumerate (column):
        sub = data_ref.groupby(['bookingID']).agg({cid: ["mean"]})
        sub.columns = [''.join(col).strip() for col in sub.columns.values] 
#        main = pd.concat([main, sub], axis=1, sort=False)
        data = pd.merge(data, sub,left_on = ['bookingID'],right_on =['bookingID'] , how = 'left')
        data[cid+'_offset'] = data[cid]- data[sub.columns.item()]
        data = data.drop([sub.columns.item()], axis=1)
    return data

#use lpf to remove spike data
def butter_lowpass_filter(data,fc=0.6,fs=1,order=2,column=['ax_offset']):
    for bn, bid in enumerate (set(data['bookingID'])):
        for cn, cid in enumerate (column):
            x = data[data['bookingID']==bid][cid]
#            print (x)
            low = fc/fs
            b,a = butter(order, low, btype='low')
            lpf_result = lfilter(b,a,x) 
            col_name = cid + "_lpf"
            data.loc[(data['bookingID']==bid),col_name] = lpf_result            
    return data

def create_data(data,features_result, category_by):
    features_result.columns = [''.join(col).strip() for col in features_result.columns.values] 
    #fill up na data
    features_result = features_result.fillna(features_result.mean())
    # Create a list of feature names
#    features_label = list(features_result.columns)
    # Create X from the features
    X = features_result.values
    # Create y from output 
    features_result.reset_index(level=0,inplace=True)
    y_data = pd.merge(data, features_result,left_on = ['bookingID'],right_on =['bookingID'] , how = 'right')
    y = np.array(y_data['label'])
    return X, y

def Train_Test(X,y):
    # Split the data into 20% test and 80% training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    # Train the classifier
    clf.fit(X_train, y_train)    
    # Print the name and gini importance of each feature
    for feature in zip(features_result, clf.feature_importances_):
        print(feature)        
    # Apply The Full Featured Classifier To The Test Data
    y_pred = clf.predict(X_test)    
    # View The Accuracy Of Our Full Feature Model
    cmatrix = confusion_matrix(y_test,y_pred) 
    classification = classification_report(y_test,y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return cmatrix, classification, accuracy

def features(data, category_by):
    result = data.groupby(
    [category_by]
    ).agg(
        {
            'Speed': [max ,"std","var"],
            'Bearing_diff': [max,"var"],  
            'acceleration_x_offset': ["sum","var","std","mean",max],
            'acceleration_y_offset': ["sum","var","std","mean",max],
            'acceleration_z_offset': ["sum","var","std","mean",max],
            'gyro_x': ["sum","var","std","mean",max],
            'gyro_y': ["sum","var","std","mean",max],
            'gyro_z': ["sum","var","std","mean",max],
            'a_mag': ["sum","var","std","mean",max],
            'g_mag': ["sum","var","std","mean",max] 
        }
    )
    return result