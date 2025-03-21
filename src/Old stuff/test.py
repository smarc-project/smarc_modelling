from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

# Define two vectors
x = np.array([0, 1, 2, 3, 4])
y1 = np.array([0, 1, 4, 9, 16])
y2 = np.array([0, 2, 8, 18, 32])

# Define the points at which to interpolate
x_new = np.linspace(0, 4, 10, endpoint=False)
print("x_new:", x_new)

# Create interpolation functions
f1 = interp1d(x, y1, kind='cubic')
f2 = interp1d(x, y2, kind='cubic')

# Perform interpolation
y1_interp = f1(x_new)
y2_interp = f2(x_new)

print("Interpolated y1:", y1_interp)
print("Interpolated y2:", y2_interp)

# Plot the results
plt.figure()
plt.plot(x, y1, 'o', label='y1')
plt.plot(x, y2, 'o', label='y2')
plt.plot(x_new, y1_interp, '-', label='y1 interp')
plt.plot(x_new, y2_interp, '-', label='y2 interp')
plt.legend()
plt.show()