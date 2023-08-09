"""
A lot of the models from John Snow use pyspark
Each model can be implemented as another module here
"""
import pandas as pd
import streamlit as st


def predict(model_name, df: pd.DataFrame):
    result = None
    just_text = df["text"].tolist()
    if model_name == "classifierdl_use_cyberbullying":
        with st.spinner("Downloading/Loading model... this may take awhile! ⏳"):
            import spark_models.cyberbullying as cyberbullying
        with st.spinner("Predicting..."):
            result = cyberbullying.predict(just_text)

    # Appending the results as the last column of the dataframe
    df["result"] = result
    return df
