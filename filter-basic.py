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


def wavelet_denoising(data, wavelet='db4', value=0.5):
    # Perform the wavelet transform.
    coefficients = pywt.wavedec2(data, wavelet)

    # Threshold the coefficients.
    threshold = pywt.threshold(coefficients[0], value=value)
    coefficients[0] = pywt.threshold(coefficients[0], threshold)

    # Inverse wavelet transform.
    denoised_data = pywt.waverec2(coefficients, wavelet)

    return denoised_data


def pulse_doppler_filter(radar_data):
    # Radar data dimensions: [range_bins, doppler_bins]
    range_bins, doppler_bins = radar_data.shape

    # Doppler filter length
    filter_length = 11

    # Generate filter coefficients using FIR filter design
    filter_coeffs = firwin(filter_length, cutoff=0.3, window='hamming', fs=2.5)

    # Output filtered data
    filtered_data = np.zeros((range_bins, doppler_bins))

    # Apply the pulse Doppler filter
    for i in range(doppler_bins):
        filtered_data[:, i] = np.convolve(radar_data[:, i], filter_coeffs, mode='same')

    return filtered_data


def create_peak_matrix(matrix, threshold):
    peak_matrix = np.zeros_like(matrix)

    rows, cols = matrix.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            current_value = matrix[i, j]
            neighbors = [
                matrix[i - 1, j],  # top
                matrix[i + 1, j],  # bottom
                matrix[i, j - 1],  # left
                matrix[i, j + 1]  # right
            ]

            if current_value >= np.max(neighbors) and current_value >= threshold:
                peak_matrix[i, j] = 1

    return peak_matrix




avg_value = []
peaks_matrix = []

mask = np.ones((configParameters["numDopplerBins"], configParameters["numRangeBins"]))
mask[8] = 0  # make central frequencies zero

frame = range_doppler[1]

denoised_frame = wavelet_denoising(frame, wavelet='haar', value=0.5)
filtered_frame = pulse_doppler_filter(denoised_frame)

peaks = create_peak_matrix(filtered_frame, threshold=0.9)

masked_peaks = peaks * mask

rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
dopplerArray = np.multiply(
        np.arange(-configParameters["numDopplerBins"] / 2, configParameters["numDopplerBins"] / 2),
        configParameters["dopplerResolutionMps"])



# for count, frame in enumerate(range_doppler):
#     plt.cla()
#
#     frame = wavelet_denoising(frame, wavelet='haar', value=0.5)
#     filtered_frame = pulse_doppler_filter(frame)
#     peaks = create_peak_matrix(filtered_frame, threshold=0.9)
#     peaks_matrix.append(peaks)
#
#     avg = np.std(peaks)
#
#     classification = "human_present" if avg > 0 else "no_human_present"
#
#     print(f"Got avg. {avg} for {classification} ground truth {labels[count]}")
#
#     plt.title(f"{labels[count]}")
#
#     avg_value.append(avg)
#
#     rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
#     dopplerArray = np.multiply(
#         np.arange(-configParameters["numDopplerBins"] / 2, configParameters["numDopplerBins"] / 2),
#         configParameters["dopplerResolutionMps"])
#
#     X, Y = np.meshgrid(rangeArray, dopplerArray)
#
#     ax.plot_surface(X, Y, peaks, cmap='viridis')
#     ax.set_xlabel('Range (m)', labelpad=10, fontsize=14, fontweight='bold')
#     ax.set_ylabel('Doppler (m/s)', labelpad=10, fontsize=14, fontweight='bold')
#     ax.set_zlabel('PDF', labelpad=10, fontsize=14, fontweight='bold')
#     plt.pause(5)

fig = plt.figure(dpi=600)
ax = fig.add_subplot(111, projection='3d')


X, Y = np.meshgrid(rangeArray, dopplerArray)

ax.plot_surface(X, Y, peaks, cmap='viridis')
ax.set_xlabel('Range (m)', labelpad=10, fontsize=14, fontweight='bold')
ax.set_ylabel('Doppler (m/s)', labelpad=10, fontsize=14, fontweight='bold')
ax.set_zlabel('PDF', labelpad=10, fontsize=14, fontweight='bold')
plt.show()