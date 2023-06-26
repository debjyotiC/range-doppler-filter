import numpy as np
np.set_printoptions(precision=3, suppress=False, linewidth=120)
data = np.load("data/peaks.npz", allow_pickle=True)

peaks, labels = data['out_x'], data['out_y']

peaks_2 = peaks[labels == 2]

matrix = np.ones((16, 128))
matrix[8] = 0

print(*matrix, sep="\n")
