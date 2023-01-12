# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 08:07:20 2022

@author: imen2
"""

import pickle
import pandas as pd
import numpy as np
from  lightgbm import LGBMClassifier
from flask import Flask,jsonify,request
import json
from sklearn.neighbors import KDTree


app=Flask(__name__)
app.config["DEBUG"] = True
#load the model
infile1=open('LGBMClassifier_f2score_is_unbalance.pkl','rb')
model=pickle.load(infile1)
#load the original data test
test=pd.read_csv('appli_test_git.csv')
columns=test.columns

#load the test dataset after preprocessing and feature engineeing (categorical data is onehot encoded)
data_test=pd.read_csv('test_git_data.csv')
data_test.set_index('SK_ID_CURR' ,inplace=True)


data=data_test.drop(['TARGET'] ,axis= 1)
        
cols_infos = ['CODE_GENDER','NAME_FAMILY_STATUS', 
               'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE',
               'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'NAME_CONTRACT_TYPE'
             ]

infos=data[cols_infos]
# DEF KDTREE

df_vois = pd.get_dummies(infos.iloc[:,:6])
tree = KDTree(df_vois)

#Home page
@app.route('/')
def home():
    return "API pour le Dashboard \'Prêt à dépenser\' "

# GENERAL INFORMATIONS
@app.route('/info', methods=['GET'])
def get_infos():        
    # Converting the pd.DataFrame to JSON
    info_json = json.loads(infos.to_json())
    return jsonify({ 'data' : info_json})    

# GENERAL INFORMATIONS ON SELECTED CLIENT
@app.route('/info/<int:id_client>', methods=['GET'])
def get_info_id(id_client):
    
    info_client_select = infos.loc[id_client,:]
    
    # Converting the pd.Series to JSON
    data_client_json = json.loads(info_client_select.to_json())

    return jsonify({ 'data' : data_client_json})

# PREDICTIONS ON SELECTED CLIENT
@app.route('/prediction/<int:id_client>', methods=['GET'])
def get_data_pred(id_client):
    
    data_client_select=data.loc[id_client]   
    
    # Converting the pd.Series to JSON
    #data_client_json=json.dumps(np.array(data_client_select).tolist())
    data_client_json = json.loads(data_client_select.to_json())    

    # Make prediction
    data_client_select=data_client_select.to_numpy()
    data_client_select=data_client_select.reshape(1, -1)
    prediction=model.predict_proba(data_client_select)[0][1]
    if prediction<0.525:
        decision='non default customer'
    else:
        decision='default customer'

    
    return jsonify({ 'data':data_client_json,
                    'decision':decision,
                    'prediction':prediction})
#PREDICTION ON CLIENT'S NEIGHBORS
@app.route('/voisins/<int:id_client>', methods=['GET'])
def voisins(id_client):
	# get indexes of 10 nearest neighbors
	idx_vois = tree.query(df_vois.loc[id_client:id_client], k=10)[1][0]
	# select processed data of neighbors
	data_vois = data.iloc[idx_vois]
	#make predictions
	predict_vois =model.predict_proba(data_vois).mean(axis=0)[1]
	# get mean of features for neighbors
	mean_vois = pd.DataFrame(data_vois.mean(), columns=['voisins']).T
	# Converting the pd.Series to JSON
	mean_vois_json = json.loads(mean_vois.to_json())

	return jsonify({ 'mean' : mean_vois_json,
					'prediction' : predict_vois})    
#Interpretations:Feature importance ON SELECTED CLIENT

@app.route('/interpretation/<int:id_client>', methods=['GET'])
        
def feat_imp(id_client):
    data_client_select = data.loc[id_client,:]
    data_client_json = json.loads(data_client_select.to_json())
    dictionnaire={'data':data_client_json
            }
    return jsonify(dictionnaire)


app.run()