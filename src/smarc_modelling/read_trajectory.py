
import re
import numpy as np

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
i = 0
for array in arrays:
    print(i)
    print(array)
    i +=1
