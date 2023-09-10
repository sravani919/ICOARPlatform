import streamlit as st
import numpy as np
from model.df.classifiers import *
from model.df.pipeline import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def df_detection():
    st.write('df')