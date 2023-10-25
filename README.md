# TSDM_WS23_34

Use `git clone` to clone this repository.

## Environment
Cautions: The python version must be lower than 3.12.0, otherwise the package `cartopy` will not be installed successfully.
Create a conda environment with the following command:
```conda create -n TSDM-env python=3.9```

Activate the environment with the following command:
```conda activate TSDM-env```

Install the required packages with the following command:
```conda install --channel conda-forge cartopy```
```conda install --file requirements.txt```