# Python Vehicle Simulator
smarc_modelling contains the dynamical models of the SMaRC vehicles and a simple simulator for basic testing.
This repository was originally a fork of the [Python Vehicle Simulator](https://github.com/cybergalactic/PythonVehicleSimulator.git) by T.I. Fossen.

## Install:
To run the main program ```main.py``` the following modules must be installed:

    numpy           https://numpy.org/install/
    matplotlib      https://matplotlib.org/stable/users/installing.html
    pytest          https://docs.pytest.org

The Python packages are automatically installed by using the command

```pip install <path>```

where ```<path>``` is the path of the downloaded PythonVehicleSimulator repository.

After downloading the repo, run:

1. ```cd smarc_modelling ```
2. ```python3 -m pip install -e .```

Note that the -e option is needed to update and change the files. If omitted, you can only run the program.

Then, create your own fork and a folder under ```src/smarc_modelling``` where you'll add your own scripts. I.e "control".

## Run
To get started with SAM, check this [readme](/src/smarc_modelling/README.md)
