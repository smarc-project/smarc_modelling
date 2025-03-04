
import sys
import os
import re 
# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import numpy as np
import matplotlib.pyplot as plt
from smarc_modelling.lib import *


# Function to extract arrays from the RTF content
def extract_arrays_from_rtf(rtf_file_path):
    with open(rtf_file_path, "r") as file:
        rtf_content = file.read()

    # Regex pattern to match the arrays in the RTF content
    array_pattern = r'\[array\((.*?)\)\]'
    array_pattern = r'\ array\((.*?)\)'

    
    # Find all arrays in the RTF content
    arrays = re.findall(array_pattern, rtf_content, re.DOTALL)
    
    # Convert arrays to numpy arrays
    numpy_arrays = []
    for array_str in arrays:
        # Convert string to a numpy array
        # Remove unwanted characters like newline and extra spaces
        array_str = array_str.replace("\n", "").replace(" ", "")
        array_str = array_str.replace("\\", "").replace(" ", "")
        array_str = array_str.strip("[]")
        number_str_list = array_str.split(",")
        
        # Convert string to a list of floats
        array_list = np.array([float(i) for i in number_str_list])

        # Convert list to numpy array
        numpy_arrays.append(np.array(array_list))
    return numpy_arrays

rtf_file_path = "/home/admin/smarc_modelling/src/smarc_modelling/sam_example_trajectory.rtf"  # Replace with your actual file path
# Extract numpy arrays from the RTF content
arrays = extract_arrays_from_rtf(rtf_file_path)

# Print the numpy arrays
reshaped_arrays = np.array([row.reshape(19, 1) for row in arrays])


print(reshaped_arrays[0,:3])
print(arrays[-1])
print(arrays[-1]-arrays[0])

plt.figure()
plt.subplot(2,2,1)
plt.plot(reshaped_arrays[:,0])
plt.plot(reshaped_arrays[:,1])
plt.plot(reshaped_arrays[:,2])
plt.legend(["x", "y", "z"])
plt.ylabel("Position [m]")
plt.grid()

n   = len(reshaped_arrays)
psi = np.zeros(n)
theta = np.zeros(n)
phi = np.zeros(n)
for i in range(n):
        quat = reshaped_arrays[i]
        quat = quat.flatten()
        print(f"heheh: {quat[3]}")
        print(quat)
        q = [quat[3], quat[4], quat[5], quat[6]]
        psi[i], theta[i], phi[i] = gnc.quaternion_to_angles(q)

plt.subplot(2,2,2)
plt.plot(np.rad2deg(phi))
plt.plot(np.rad2deg(theta))
plt.plot(np.rad2deg(psi))
plt.legend(["roll", "pitch", "yaw"])
plt.ylabel("Angle [deg]")
plt.grid()

plt.subplot(2,2,3)
plt.plot(reshaped_arrays[:,7])
plt.plot(reshaped_arrays[:,8])
plt.plot(reshaped_arrays[:,9])
plt.legend(["x", "y", "z"])
plt.ylabel("Velocity [m/s]")
plt.grid()

plt.subplot(2,2,4)
plt.plot(reshaped_arrays[:,10])
plt.plot(reshaped_arrays[:,11])
plt.plot(reshaped_arrays[:,12])
plt.legend(["x", "y", "z"])
plt.ylabel("Velocity [m/s]")
plt.grid()
plt.show()
