import matplotlib.pyplot as plt
import numpy as np

# Example data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
plt.plot(x, y)

# Add Greek letter psi to the title, xlabel, and ylabel
plt.title(r'Plot of $\ddot\psi$ vs. x', fontsize=14)
plt.xlabel(r'$\psi$ (x)', fontsize=12)
plt.ylabel(r'$\dot\theta$', fontsize=12)

# Show the plot
plt.show()
