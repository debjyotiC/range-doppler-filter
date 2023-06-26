import numpy as np
from scipy.signal import firwin
import matplotlib.pyplot as plt

range_doppler_labels = np.load("data/range_doppler_home_data.npz", allow_pickle=True)
range_doppler, labels = range_doppler_labels["out_x"], range_doppler_labels["out_y"]


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
    filter_coeffs = firwin(filter_length, cutoff=0.2, window='hamming', fs=1.0)

    # Output filtered data
    filtered_data = np.zeros((range_bins, doppler_bins))

    # Apply the pulse Doppler filter
    for i in range(doppler_bins):
        filtered_data[:, i] = np.convolve(radar_data[:, i], filter_coeffs, mode='same')

    return filtered_data


def generate_threshold(range_doppler_data, window_size, threshold_factor):
    # Clutter power estimation using cell averaging
    clutter_power = np.zeros_like(range_doppler_data)
    range_bins, doppler_bins = range_doppler_data.shape

    for i in range(range_bins):
        for j in range(doppler_bins):
            # Estimate clutter power in a local window around (i, j)
            i_start = max(0, i - window_size // 2)
            i_end = min(range_bins, i + window_size // 2 + 1)
            j_start = max(0, j - window_size // 2)
            j_end = min(doppler_bins, j + window_size // 2 + 1)
            local_window = range_doppler_data[i_start:i_end, j_start:j_end]
            clutter_power[i, j] = np.mean(np.abs(local_window) ** 2)

    # Compute threshold value
    clutter_std = np.std(clutter_power)
    threshold = threshold_factor * clutter_std

    return threshold


def zero_velocity_filter(data):
    # Calculate the Doppler spectrum by taking the FFT along the range axis
    doppler_spectrum = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)

    # Set the zero Doppler frequency component and its neighbors to zero
    num_range_bins = data.shape[0]
    zero_doppler_index = num_range_bins // 2
    doppler_spectrum[zero_doppler_index - 1:zero_doppler_index + 2, :] = 0

    # Reconstruct the filtered data by taking the inverse FFT
    filtered_data = np.fft.ifft(np.fft.ifftshift(doppler_spectrum, axes=0), axis=0)

    return np.abs(filtered_data)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for count, frame in enumerate(range_doppler):
    plt.cla()
    denoise_frame = wavelet_denoising(frame)
    pdf_data = pulse_doppler_filter(frame)

    x = np.arange(pdf_data.shape[1])
    y = np.arange(pdf_data.shape[0])
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, pdf_data, cmap='viridis')
    ax.set_xlabel('Range')
    ax.set_ylabel('Doppler')
    ax.set_zlabel('PDF')

    plt.pause(0.1)

plt.show()
