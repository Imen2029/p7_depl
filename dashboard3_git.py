# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 08:06:40 2022

@author: imen2
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go
import pickle
from  lightgbm import LGBMClassifier
#import seaborn as sns
from matplotlib import pyplot as plt
import shap
import streamlit.components.v1 as components

API_URL='http://127.0.0.1:5000/'
#API_URL='http://0.0.0.0:$PORT/'

#load the model
infile1=open('LGBMClassifier_f2score_is_unbalance.pkl','rb')
model=pickle.load(infile1)

train=pd.read_csv('train_git_data.csv')

train=train.set_index('SK_ID_CURR')

train_data=train.drop(['TARGET'] ,axis= 1)
columns=train.columns
# LOAD DATA
#@st.cache(allow_output_mutation=True)
def load_data(url):
    response = requests.get(url)
    if response.status_code != 200:
        st.text(' connection error')
        print("Error from server: " + str(response.content))
    else:
        content = json.loads(response.content.decode('utf-8'))
        return pd.DataFrame(content['data'])
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
def main():
    data_load_state = st.text('Loading data...')
    infos = load_data(API_URL + 'info')
    infos = infos[['CODE_GENDER','NAME_FAMILY_STATUS', 
                   'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE',
                   'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'NAME_CONTRACT_TYPE']]
    data_load_state.text('')
    #-------------------------------------------------------------------------#
    # GENERAL INFORMATION
    st.title('Dashboard PRET A DEPENSER')

    # Select client
    client_id = st.sidebar.selectbox('Select ID Client :', infos.index)

    # Display general informations in sidebar
    st.sidebar.table(infos.loc[client_id][:6])
    
    # Plot data relative to income and credit amounts
    bar_cols = infos.columns[4:7]
    infos_amt_mean = infos[bar_cols].mean()
    #infos.at['Moyenne clients', bar_cols] = infos.loc[:,bar_cols].mean()
    fig = go.Figure(data=[
        go.Bar(name='Client sélectionné', x=bar_cols, y=infos.loc[client_id, bar_cols].values),
       go.Bar(name='Moyenne des clients', x=bar_cols, y=infos_amt_mean.loc[ bar_cols].values)
    ])
    fig.update_layout(title_text=f'Montants des revenus et du crédit demandé pour le client {client_id}')

    st.plotly_chart(fig, use_container_width=True)
    #-------------------------------------------------------------------------#
    # PREDICTIONS

    st.header('Risque de défaut')
    # Load data client :
    url_data_client = API_URL + 'prediction/' + str(client_id)
    req = requests.get(url_data_client)
    if req.status_code != 200:
        st.text(' connection to prediction  error')
        st.text(client_id)
        st.text("Error from server: " + str(req.content))
        req.content
    else:
        content = json.loads(req.content.decode('utf-8'))
        # Get predictions :
        prediction_client = content['prediction']
        data_client=content['data']
        
    # Get predictions for similar clients :
    url_voisins_client = API_URL + 'voisins/' + str(client_id)
    req = requests.get(url_voisins_client)
    content = json.loads(req.content.decode('utf-8'))
    prediction_voisins = content['prediction']

    # Plot gauge
    gauge = go.Figure(go.Indicator(
         mode = "gauge+delta+number",
         value = prediction_client,
         domain = {'x': [0, 1], 'y': [0, 1]},
         gauge = {'axis': {'range': [None, 100]},
                  'steps' : [
                      {'range': [0, 25], 'color': "lightgreen"},
                      {'range': [25, 50], 'color': "lightyellow"},
                      {'range': [50, 75], 'color': "orange"},
                      {'range': [75, 100], 'color': "red"},
                      ],
                  'threshold': {
                 'line': {'color': "black", 'width': 10},
                 'thickness': 0.8,
                 'value': prediction_client},

                  'bar': {'color': "black", 'thickness' : 0.2},
                 },
         delta = {'reference': prediction_voisins,
         'increasing': {'color': 'red'},
         'decreasing' : {'color' : 'green'}}
         ))

    st.plotly_chart(gauge)

    st.markdown('Score: client sélectionné : **{0:.3f}%**'.format(prediction_client))
    st.markdown('Score moyen: clients similaires : **{0:.3f}%** (critères de similarité : genre,\
          statut familial, éducation, profession)'.format(prediction_voisins))
        
    if prediction_client<0.525:
        st.text('Non default customer')
    else:
        st.text('Default customer')
    # ________________________________________________________
    # INTERPRETATION

    feature_desc = { 'EXT_SOURCE_2' : 'Score normalisé attribué par un organisme indépendant',
                    'EXT_SOURCE_3' :  'Score normalisé attribué par un organisme indépendant', 
                    'AMT_ANNUITY' : 'Montant des annuités', 
                    
                    'CREDIT_TO_INCOME_RATIO' : 'Crédit demandé par rapport aux revenus', 
                    'DAYS_EMPLOYED' : 'Années travaillées en pourcentage' }
    st.header('Interprétation du résultat')
    url_feat_imp = API_URL + 'interpretation/' + str(client_id)
    req1 = requests.get(url_feat_imp)
    
    content1 = json.loads(req1.content.decode('utf-8'))
    indices = np.argsort(model.feature_importances_)[::-1]    
    features = []
    columns=list(train_data.columns)
    for i in range(30):
        features.append(columns[indices[i]]) 

    fig = go.Figure(data=[
        go.Bar(name='fe', x=features, y=model.feature_importances_[indices[range(30)]]),
       
    ])
    fig.update_layout(title_text='Global features importance')

    st.plotly_chart(fig, use_container_width=True)
    st.write('Feature importance of the selected customer')
    explainer = shap.TreeExplainer(model)
    #shap_values = explainer.shap_values(train_data)
    
    df_data_client = pd.DataFrame([data_client])
    shap_vals = explainer.shap_values(df_data_client)

  
    #st_shap(shap.force_plot(shap_vals[0]),400)
    st_shap(shap.force_plot(explainer.expected_value[1], shap_vals[0], df_data_client.loc[:,:],figsize=(40,10),text_rotation=30),800)
    #st.write(df_data_client)
    #st.write(df_data_client.shape)
    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    #st_shap(shap.force_plot(explainer.expected_value, shap_vals[0], df_data_client.iloc[0,:]))
    # visualize the training set predictions
    #st_shap(shap.force_plot(explainer.expected_value, shap_values, train_data), 400)
    #explainer = shap.TreeExplainer(model)
# =============================================================================
#     shap_values = explainer.shap_values(train_data)[0:30]
#     shap_imp_pd=pd.DataFrame(
#     index=train_data.columns[0:30], 
#     data = np.mean(np.absolute(shap_values), axis = 0), 
#     columns=["Importance"])
#     shap_imp_pd.sort_values(
#     by=['Importance']).plot.barh(y='Importance')
# =============================================================================
    #features = content1['features']
    #shap_values = content1['feat_imp']
    #shap_values = shap.TreeExplainer(model).shap_values(data)
    #shap.summary_plot(shap_values, X_encoded, plot_type='bar',feature_names=feature_after_transformer(preprocessor))
# =============================================================================
#     indices = np.argsort(model.feature_importances_)[::-1]
#     features = []
#     for i in range(30):
#         features.append(columns[indices[i]]) 
# 
#     sns.barplot(x=features, y=model.feature_importances_[indices[range(30)]], color=("orange"))
# =============================================================================
if __name__== '__main__':
    main()