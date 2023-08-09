"""
When this module is imported, it will download the models if they are not already downloaded
Only import this module once the user has selected that they want to use this model to avoid unnecessary downloads
"""

import json
import pandas as pd
import numpy as np

import sparknlp
import pyspark.sql.functions as F

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.pretrained import PretrainedPipeline
from pyspark.sql.types import StringType, IntegerType

spark = sparknlp.start()

spark.sparkContext.setLogLevel("ERROR")

MODEL_NAME = "classifierdl_use_cyberbullying"

documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

# Used to create universal sentence encodings (use)
use = (
    UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en")
    .setInputCols(["document"])
    .setOutputCol("sentence_embeddings")
)

# Takes the sentence embeddings and outputs the sentiment
sentimentdl = (
    ClassifierDLModel.pretrained(name=MODEL_NAME).setInputCols(["sentence_embeddings"]).setOutputCol("sentiment")
)

# Links the above stages together
nlpPipeline = Pipeline(stages=[documentAssembler, use, sentimentdl])


def predict(text_list: list[str]):
    df = spark.createDataFrame(text_list, StringType()).toDF("text")
    result = nlpPipeline.fit(df).transform(df)

    # Convert the result to a simple list of strings
    result = result.select("sentiment.result").collect()
    result = [row.result for row in result]
    print(result)


if __name__ == "__main__":
    # For testing purposes
    predict(["I love you", "I hate you"])
