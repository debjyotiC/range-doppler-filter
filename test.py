import numpy as np
import matplotlib.pyplot as plt

# Create the mask
mask = np.ones((16, 256))
mask[8] = 0  # make central frequencies zero
configParameters = {'numDopplerBins': 16, 'numRangeBins': 256, 'rangeResolutionMeters': 0.146484375,
                    'rangeIdxToMeters': 0.146484375, 'dopplerResolutionMps': 0.1252347734553042, 'maxRange': 33.75,
                    'maxVelocity': 1.0018781876424336}

rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
dopplerArray = np.multiply(
    np.arange(-configParameters["numDopplerBins"] / 2, configParameters["numDopplerBins"] / 2),
    configParameters["dopplerResolutionMps"])

# Display the matrix using a heatmap
plt.contourf(rangeArray, dopplerArray, mask)

plt.xlabel('Range (m)')
plt.ylabel('Doppler (m/s)')
plt.savefig('mask.png', dpi=600)
plt.show()