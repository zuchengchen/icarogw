# ICAROGW

This is a python package to infer cosmology and astrophysics with observations of gravitational waves (ICAROGW).

For a quick tour about the icarogw functionality, please refer to the notebook under the `example` foder.

If you use this code for one of your studies, please cite [S. Mastrogiovanni et al, PRD 062009 (2021)](https://ui.adsabs.harvard.edu/abs/2021PhRvD.104f2009M/exportcitation)

## Installation instructions

You will need a conda distribution to install icarogw. The easiest way to install icarogw is the following

* [Download](https://anaconda.org/simone.mastrogiovanni/icarogw/2021.11.04.101404/download/env_creator.yml) the environment creator for icarogw.
* Run from your terminal ` conda env create -f env_creator.yml`. This will create a virtual environment where icarogw is installed.
* Activate the environment with `conda activate icarogw`.
* If you want to install the environment in your jupyter notebook. With the icarogw environment activated just run `python -m ipykernel install --user --name icarogw --display-name "icarogw"`.
* If you modify the icarogw source code and you want to re-install it, run from the icarogw folder `python setup.py install --force`.
