import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Boston House Price Prediction App
This app predicts the *Boston House Price*!
""")
st.write('---')

# Loads the Boston House Price Dataset
os.chdir('C:\\Users\\Shubham\\Downloads')
column_names = ['CRIME_RATE', 'PROP_OF_LAND_OVER_25000_SQKM', 'PROP_OF_NON_RETAIL_BUSINESS_ACRE_PER_TOWN', 'CHARLES_RIVER', 'NITRIC_OXIDE_CONC', 'AVG_NO_OF_ROOMS', 'AGE', 'DISTANCE_TO_BUSINESS_CENTERS', 'INDEX_OF_ACCESS_TO_HIGHWAYS', 'TAX', 'PUPIL_TEACHER_RATIO'
,'PROP_OF_BLACK', 'LOWER_STATUS_PERCENTAGE_OF_POPULATION', 'MEDIAN_VALUE_OF_HOUSE']
boston = pd.read_csv('housing_data.csv', header=None, delimiter=r"\s+", names=column_names)
X = pd.DataFrame(data = boston, columns = ['CRIME_RATE', 'PROP_OF_LAND_OVER_25000_SQKM', 'PROP_OF_NON_RETAIL_BUSINESS_ACRE_PER_TOWN', 'CHARLES_RIVER', 'NITRIC_OXIDE_CONC', 'AVG_NO_OF_ROOMS', 'AGE', 'DISTANCE_TO_BUSINESS_CENTERS', 'INDEX_OF_ACCESS_TO_HIGHWAYS', 'TAX', 'PUPIL_TEACHER_RATIO'
, 'PROP_OF_BLACK', 'LOWER_STATUS_PERCENTAGE_OF_POPULATION'] )
Y = pd.DataFrame(data = boston, columns=["MEDIAN_VALUE_OF_HOUSE"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIME_RATE = st.sidebar.slider('CRIME_RATE', float(X.CRIME_RATE.min()), float(X.CRIME_RATE.max()), float(X.CRIME_RATE.mean()))
    PROP_OF_LAND_OVER_25000_SQKM = st.sidebar.slider('PROP_OF_LAND_OVER_25000_SQKM', X.PROP_OF_LAND_OVER_25000_SQKM.min(), X.PROP_OF_LAND_OVER_25000_SQKM.max(), float(X.PROP_OF_LAND_OVER_25000_SQKM.mean()))
    PROP_OF_NON_RETAIL_BUSINESS_ACRE_PER_TOWN = st.sidebar.slider('PROP_OF_NON_RETAIL_BUSINESS_ACRE_PER_TOWN', X.PROP_OF_NON_RETAIL_BUSINESS_ACRE_PER_TOWN.min(), X.PROP_OF_NON_RETAIL_BUSINESS_ACRE_PER_TOWN.max(), float(X.PROP_OF_NON_RETAIL_BUSINESS_ACRE_PER_TOWN.mean()))
    CHARLES_RIVER = st.sidebar.slider('CHARLES_RIVER', float(X.CHARLES_RIVER.min()), float(X.CHARLES_RIVER.max()), float(X.CHARLES_RIVER.mean()))
    NITRIC_OXIDE_CONC = st.sidebar.slider('NITRIC_OXIDE_CONC', X.NITRIC_OXIDE_CONC.min(), X.NITRIC_OXIDE_CONC.max(), float(X.NITRIC_OXIDE_CONC.mean()))
    AVG_NO_OF_ROOMS = st.sidebar.slider('AVG_NO_OF_ROOMS', X.AVG_NO_OF_ROOMS.min(), X.AVG_NO_OF_ROOMS.max(), float(X.AVG_NO_OF_ROOMS.mean()))
    AGE = st.sidebar.slider('AGE', X.AGE.min(), X.AGE.max(), float(X.AGE.mean()))
    DISTANCE_TO_BUSINESS_CENTERS = st.sidebar.slider('DISTANCE_TO_BUSINESS_CENTERS', X.DISTANCE_TO_BUSINESS_CENTERS.min(), X.DISTANCE_TO_BUSINESS_CENTERS.max(), float(X.DISTANCE_TO_BUSINESS_CENTERS.mean()))
    INDEX_OF_ACCESS_TO_HIGHWAYS = st.sidebar.slider('INDEX_OF_ACCESS_TO_HIGHWAYS', float(X.INDEX_OF_ACCESS_TO_HIGHWAYS.min()), float(X.INDEX_OF_ACCESS_TO_HIGHWAYS.max()), float(X.INDEX_OF_ACCESS_TO_HIGHWAYS.mean()))
    TAX = st.sidebar.slider('TAX', X.TAX.min(), X.TAX.max(), float(X.TAX.mean()))
    PUPIL_TEACHER_RATIO = st.sidebar.slider('PUPIL_TEACHER_RATIO', X.PUPIL_TEACHER_RATIO.min(), X.PUPIL_TEACHER_RATIO.max(), float(X.PUPIL_TEACHER_RATIO.mean()))
    PROP_OF_BLACK = st.sidebar.slider('PROP_OF_BLACK', X.PROP_OF_BLACK.min(), X.PROP_OF_BLACK.max(), float(X.PROP_OF_BLACK.mean()))
    LOWER_STATUS_PERCENTAGE_OF_POPULATION = st.sidebar.slider('LOWER_STATUS_PERCENTAGE_OF_POPULATION', X.LOWER_STATUS_PERCENTAGE_OF_POPULATION.min(), X.LOWER_STATUS_PERCENTAGE_OF_POPULATION.max(), float(X.LOWER_STATUS_PERCENTAGE_OF_POPULATION.mean()))
    data = {'CRIME_RATE': CRIME_RATE,
            'PROP_OF_LAND_OVER_25000_SQKM': PROP_OF_LAND_OVER_25000_SQKM,
            'PROP_OF_NON_RETAIL_BUSINESS_ACRE_PER_TOWN': PROP_OF_NON_RETAIL_BUSINESS_ACRE_PER_TOWN,
            'CHARLES_RIVER': CHARLES_RIVER,
            'NITRIC_OXIDE_CONC': NITRIC_OXIDE_CONC,
            'AVG_NO_OF_ROOMS': AVG_NO_OF_ROOMS,
            'AGE': AGE,
            'DISTANCE_TO_BUSINESS_CENTERS': DISTANCE_TO_BUSINESS_CENTERS,
            'INDEX_OF_ACCESS_TO_HIGHWAYS': INDEX_OF_ACCESS_TO_HIGHWAYS,
            'TAX': TAX,
            'PUPIL_TEACHER_RATIO': PUPIL_TEACHER_RATIO,
            'PROP_OF_BLACK': PROP_OF_BLACK,
            'LOWER_STATUS_PERCENTAGE_OF_POPULATION': LOWER_STATUS_PERCENTAGE_OF_POPULATION}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')