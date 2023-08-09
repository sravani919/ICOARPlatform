"""
A lot of the models from John Snow use pyspark
Each model can be implemented as another module here
"""

def predict(model_name, df):
    result = None
    just_text = df["text"].tolist()
    if model_name == "classifierdl_use_cyberbullying":
        import spark_models.cyberbullying as cyberbullying
        result = cyberbullying.predict(just_text)

    # Appending the results as the last column of the dataframe
    df["result"] = result
    return df


