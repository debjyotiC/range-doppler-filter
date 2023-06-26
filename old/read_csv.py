import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

data = "range_doppler_home_data.csv"

df = pd.read_csv(data)

k_value = df["kurtosis"]
s_value = df["skew"]
labels = df["Ground truth"]






