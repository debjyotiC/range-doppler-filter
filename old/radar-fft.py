import numpy as np
import matplotlib.pyplot as plt

# Assuming you have your range doppler data stored in a variable called 'data'
data = np.random.rand(16, 128)  # Example random data, replace with your actual data

# Perform 1D-FFT on each range bin
fft_result = np.fft.fft(data, axis=1)

# Calculate the magnitude of the FFT values along the range axis
fft_magnitude = np.abs(fft_result)

# Create the x-axis values for the plot
x = np.arange(data.shape[1])  # Range axis

# Plot the FFT magnitude over the range axis
plt.figure()
for i in range(data.shape[0]):
    plt.plot(x, fft_magnitude[i, :], label=f'Range Bin {i+1}')

plt.xlabel('Range')
plt.ylabel('Magnitude')
plt.title('FFT Magnitude over Range')
plt.legend()

# Show the plot
plt.show()
