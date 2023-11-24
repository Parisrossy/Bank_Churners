import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('BankChurners.csv')
# data

# data.isnull().sum()
# # FEATURE ENGINEERING

df = data.copy()
# df['Income Category'] = df['Income_Category'].str.extract(r'(\d+)') # .......... extract all numericals from the column and save it to a new column
# df['Income Category'] = df['Income Category'].astype(float) # .................. Turn the new column to a numerical datatype
# df.drop(['CLIENTNUM', 'Income_Category'], axis = 1, inplace = True) # .......... Drop the columns we dont need
# #df.head()


# # Check if the DEPENDENTS column has been identified into its right datatype
# categoricals = df.select_dtypes(include = ['object', 'category'])
# numericals = df.select_dtypes(include = 'number')

# print(f"\t\tCategorical Columns")
# print(f"\n\t\tNumerical Columns")

# # Clean the newly created column
# df['Income Category'].fillna(df['Income Category'].median(), inplace = True)

# # PREPROCESSSING
# # Standardization
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# scaler = StandardScaler()
# encoder = LabelEncoder()

# for i in numericals.columns: # ................................................. Select all numerical columns
#     if i in df.columns: # ...................................................... If the selected column is found in the general dataframe
#         df[i] = scaler.fit_transform(df[[i]]) # ................................ Scale it

# for i in categoricals.columns: # ............................................... Select all categorical columns
#     if i in df.columns: # ...................................................... If the selected columns are found in the general dataframe
#         df[i] = encoder.fit_transform(df[i])# .................................. encode it

# # df.head()
# # FEATURE SELECTION

# x = df.drop('Attrition_Flag', axis = 1)
# y = df['Attrition_Flag']

# sel_cols = ['Total_Trans_Amt', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Customer_Age', 'Total_Revolving_Bal', 'Months_on_book']
# x = df[sel_cols]
# # x.head()

# # UnderSampling The Majority Class
# dx = pd.concat([x, y], axis =1) # .............................................. create a new dataframe with x and y
# class1 = dx.loc[dx['Attrition_Flag'] == 1] # ................................... select attrition flag that is only 1
# class0 = dx.loc[dx['Attrition_Flag'] == 0] # ................................... select attrition flag that is only 0

# class1_3000 = class1.sample(2800) # ............................................ randomly select 2800 rows from majority class 1

# new_dataframe = pd.concat([class1_3000, class0], axis = 0) # ................... join the new data of class 1 and class 0 together along the rows
# (new_dataframe)
# sns.countplot(x = new_dataframe['Attrition_Flag'])

# #modelling
# x = new_dataframe.drop('Attrition_Flag', axis = 1)
# y = new_dataframe['Attrition_Flag']

# from sklearn.model_selection import train_test_split
# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20, random_state = 9, stratify = y)

# algorithms = ['RandomForestClassifier', 'DecisionTreeClassifier', 'SVMClassifier', 'XGBoostClassifier', 'LogisticRegression']

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from xgboost import XGBClassifier

# # Modelling
# model = RandomForestClassifier() 
# model.fit(xtrain, ytrain) 
# cross_validation = model.predict(xtrain)
# pred = model.predict(xtest) 

# model = pickle.dump(model, open('BankChurners.pkl', 'wb'))
# print('\nModel is saved\n')


#-------Streamlit development------
model = pickle.load(open('BankChurners.pkl', "rb"))

st.markdown("<h1 style = 'color: #B31312; text-align: center;font-family: Arial, Helvetica, sans-serif; '>BANK CHURNERS PREDICTION</h1>", unsafe_allow_html= True)
st.markdown("<h3 style = 'margin: -25px; color: #2B2A4C; text-align: center;font-family: Arial, Helvetica, sans-serif; '>BY OLUWAYOMI ROSEMARY</h3>", unsafe_allow_html= True)
st.image('pngwing.com (20).png', width = 600)
st.markdown("<h2 style = 'color: #2B2A4C; text-align: center;font-family: Arial, Helvetica, sans-serif; '>BACKGROUND OF STUDY </h2>", unsafe_allow_html= True)

st.markdown('<br>', unsafe_allow_html= True)
st.markdown("<p>Predicting customer churn in banking involves analyzing various factors that contribute to customers leaving or discontinuing their services. Here's a breakdown of steps for a background study.</p>",unsafe_allow_html= True)

st.sidebar.image('pngwing.com (21).png')
st.markdown('<br>', unsafe_allow_html= True)

input_type = st.sidebar.radio("Select Your Preferred Input Style", ["Slider", "Number Input"])
st.markdown('<br>', unsafe_allow_html= True)

if input_type == 'Slider':
    Total_Trans_Amt = st.sidebar.slider('Total_Trans_Amt', df['Total_Trans_Amt'].min(), df['Total_Trans_Amt'].max())
    Total_Amt_Chng_Q4_Q1 = st.sidebar.slider('Total_Amt_Chng_Q4_Q1', df['Total_Amt_Chng_Q4_Q1'].min(), df['Total_Amt_Chng_Q4_Q1'].max())
    Total_Trans_Ct = st.sidebar.slider('Total_Trans_Ct', df['Total_Trans_Ct'].min(), df['Total_Trans_Ct'].max())
    Total_Ct_Chng_Q4_Q1 = st.sidebar.slider('Total_Ct_Chng_Q4_Q1', df['Total_Ct_Chng_Q4_Q1'].min(), df['Total_Ct_Chng_Q4_Q1'].max())
    Customer_Age = st.sidebar.slider('Customer_Age', df['Customer_Age'].min(), df['Customer_Age'].max())
    Total_Revolving_Bal = st.sidebar.slider('Total_Revolving_Bal', df['Total_Revolving_Bal'].min(), df['Total_Revolving_Bal'].max())
    Months_on_book = st.sidebar.slider('Months_on_book', df['Months_on_book'].min(), df['Months_on_book'].max())
else:
    Total_Trans_Amt = st.sidebar.number_input('Total_Trans_Amt', df['Total_Trans_Amt'].min(), df['Total_Trans_Amt'].max())
    Total_Amt_Chng_Q4_Q1 = st.sidebar.number_input('Total_Amt_Chng_Q4_Q1', df['Total_Amt_Chng_Q4_Q1'].min(), df['Total_Amt_Chng_Q4_Q1'].max())
    Total_Trans_Ct = st.sidebar.number_input('Total_Trans_Ct', df['Total_Trans_Ct'].min(), df['Total_Trans_Ct'].max())
    Total_Ct_Chng_Q4_Q1 = st.sidebar.number_input('Total_Ct_Chng_Q4_Q1', df['Total_Ct_Chng_Q4_Q1'].min(), df['Total_Ct_Chng_Q4_Q1'].max())
    Customer_Age = st.sidebar.number_input('Customer_Age', df['Customer_Age'].min(), df['Customer_Age'].max())
    Total_Revolving_Bal = st.sidebar.number_input('Total_Revolving_Bal', df['Total_Revolving_Bal'].min(), df['Total_Revolving_Bal'].max())
    Months_on_book = st.sidebar.number_input('Months_on_book', df['Months_on_book'].min(), df['Months_on_book'].max())

st.header('Input Values')
# Bring all the inputs into a dataframe
input_variable = pd.DataFrame([{'Total_Trans_Amt':Total_Trans_Amt, 'Total_Amt_Chng_Q4_Q1': Total_Amt_Chng_Q4_Q1, 'Total_Trans_Ct': Total_Trans_Ct, 'Total_Ct_Chng_Q4_Q1':Total_Ct_Chng_Q4_Q1, 'Customer_Age':Customer_Age, 'Total_Revolving_Bal': Total_Revolving_Bal, 'Months_on_book':Months_on_book}])

st.write(input_variable)

# Standard Scale the Input Variable.
for i in input_variable.columns:
    input_variable[i] = StandardScaler().fit_transform(input_variable[[i]])

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("<h2 style = 'color: #0A2647; text-align: left; font-family: helvetica '>Model Report</h2>", unsafe_allow_html = True)

if st.button('Press To Predict'):
    predicted = model.predict(input_variable)
    st.toast('Attrition Flag Predicted')
    st.image('pngwing.com (22).png', width = 200)
    st.success(f'{predicted} Predicted')

    if predicted == 0:
        st.success('This person is not an existing customer')
    else:
        st.success('This person is an existing customer')