import numpy as np 
import pandas as pd 
import streamlit as st

from sklearn.preprocessing import StandardScaler 
from src.pipeline.predict_pipeline import CustomData, PredictPipeline 

st.markdown("""
# Student Performance Indicator App """)
st.header("User Input Features")


def user_input_features():
    """
    Function to take inputs from user and returns the predicted value
    """

    data = CustomData(gender = st.selectbox('gender',('male', 'female')),
        race_ethnicity = st.selectbox('race_ethnicity',('group A', 'group B', 'group C', 'group D', 'group E')),
        parental_level_of_education = st.selectbox('parental_level_of_education',("associate's degree", "high school", "master's degree", 
                                                                                        "some college", "some high school")),
        lunch = st.selectbox('lunch',("free/reduced", "standard")),
        test_preparation_course = st.selectbox('test_preparation_course',("none", "completed")),
        reading_score = int(st.slider('reading_score', 0, 100, 50)),
        writing_score = int(st.slider('writing_score', 0, 100, 50))
        )

    pred_df = data.get_data_as_data_frame()

    # initializaing prediction pipeline 
    pred_pipeline = PredictPipeline()
    preds = pred_pipeline.predict(pred_df)

    if st.button("Predict"):
        st.balloons()
        st.success(f"Maths Score of a Student: {int(preds)}")

results = user_input_features()




