import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin
import pywt
import pandas as pd

data_path = "data/umbc_new_2.npz"  # change with desired NPZ file

range_doppler_labels = np.load(data_path, allow_pickle=True)
range_doppler, labels = range_doppler_labels["out_x"], range_doppler_labels["out_y"]

configParameters = {'numDopplerBins': 16, 'numRangeBins': 256, 'rangeResolutionMeters': 0.146484375,
                    'rangeIdxToMeters': 0.146484375, 'dopplerResolutionMps': 0.1252347734553042, 'maxRange': 33.75,
                    'maxVelocity': 1.0018781876424336}


def highlight_peaks(matrix, threshold):
    rows, cols = matrix.shape
    peaks = []

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if matrix[i, j] >= threshold:
                neighbors = matrix[i - 1:i + 2, j - 1:j + 2]
                if matrix[i, j] == np.max(neighbors):
                    peaks.append((i, j))

    return peaks


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

avg_value = []
peaks_matrix = []

mask = np.ones((configParameters["numDopplerBins"], configParameters["numRangeBins"]))
mask[8] = 0  # make central frequencies zero


rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
dopplerArray = np.multiply(np.arange(-configParameters["numDopplerBins"] / 2, configParameters["numDopplerBins"] / 2),
                           configParameters["dopplerResolutionMps"])

for count, frame in enumerate(range_doppler):

    highlighted_peaks = highlight_peaks(frame, threshold=70.0)
    highlighted_peaks_array = np.array(highlighted_peaks)
    classes_values = ["Human_Present", "No_Human_detected"]

    try:
        picked_elements = rangeArray[highlighted_peaks_array[:, 1]].round(2)[:4]  # select only 4 detected objects
    except IndexError:
        picked_elements = [0.01, 0.01]  # push dummy data

    stacked_arr = np.vstack((picked_elements[:2],) * 5)  # Fist 2 elements of the detected range array stacked 5 times

    if np.any(stacked_arr > 1.0):
        predicted_class = classes_values[0]
    else:
        predicted_class = classes_values[1]

    print(predicted_class)
