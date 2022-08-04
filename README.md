# rSNAPed

<img src="./docs/images/logo/rSNAPed_Logo.png" width="200" />

rSNAPed : RNA Sequence to NAscent Protein Experiment Designer.

Authors: Luis U. Aguilera, William Raymond, Brooke Silagy, Brian Munsky.

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Description

This library is intended to generate simulated single-molecule gene expression experiments to test machine learning pipelines. 

The code generates simulated intensity translation spots using [rSNAPsim](https://github.com/MunskyGroup/rSNAPsim). The code uses [Cellpose](https://github.com/MouseLand/cellpose) to segment the cell in the image. Then, it uses [Trackpy](http://soft-matter.github.io/trackpy/dev/index.html) to detect spots inside the mask. 

If you use `rSNAPed`, please make sure you properly cite `cellpose`, `trackpy` and `rSNAPsim`.

## Usage

* Tracking for single-molecule translation (RNA + nascent protein) spots.
* Tracking for single-molecule RNA spots.
* Simulating the single-molecule translation for any gene.
* Design of single-molecule gene expression experiments.

## Simulating translation

The code generates videos with the simulated cell and a data frame containing spot and intensity positions. This simulation can be used to train new algorithms.

## Local installation using PIP

* To create a virtual environment using:

```bash
    conda create -n rsnaped_env python=3.8.5 -y
    source activate rsnaped_env
```

* Open the terminal and use [pip](https://pip.pypa.io/en/stable/) for the installation:
```bash
    pip install rsnaped
```

## Local installation from the Github repository

* To create a virtual environment navigate to the location of the requirements file, and use:
```bash
    conda create -n rsnaped_env python=3.8.5 -y
    source activate rsnaped_env
```
* To install GPU for Cellpose (Optional step). Only for **Linux and Windows users** check the specific version for your computer on this [link]( https://pytorch.org/get-started/locally/) :
```
    conda install pytorch cudatoolkit=10.2 -c pytorch -y
```
* To install CPU for Cellpose (Optional step). Only for **Mac users** check the specific version for your computer on this [link]( https://pytorch.org/get-started/locally/) :
```
    conda install pytorch -c pytorch
```
* To include the rest of requirements use:
```
    pip install -r requirements.txt
```
Additional steps to deactivate or remove the environment from the computer:
* To deactivate the environment use
```
    conda deactivate
```
* To remove the environment use:
```
    conda env remove -n rsnaped_env
```

## References for main dependencies

- [rSNAPsim](https://github.com/MunskyGroup/rSNAPsim):
 Aguilera, Luis U., et al. "Computational design and interpretation of single-RNA translation experiments." PLoS computational biology 15.10 (2019): e1007425.

- [Trackpy](http://soft-matter.github.io/trackpy/dev/index.html):
 Dan Allan, et al. (2019, October 16). soft-matter/trackpy: Trackpy v0.4.2 (Version v0.4.2). Zenodo. http://doi.org/10.5281/zenodo.3492186

- [Cellpose](https://github.com/MouseLand/cellpose):
 Stringer, Carsen, et al. "Cellpose: a generalist algorithm for cellular segmentation." Nature Methods 18.1 (2021): 100-106.

## Licenses for dependencies

**For a complete list containing the complete licenses for the dependencies, check file:  [Licenses_Dependencies.txt](https://github.com/MunskyGroup/rsnaped/blob/master/Licenses_Dependencies.txt).**

- License for [rSNAPsim](https://github.com/MunskyGroup/rSNAPsim): MIT. Copyright © 2018 Dr. Luis Aguilera, William Raymond
- License for [Trackpy](http://soft-matter.github.io/trackpy/dev/index.html): BSD-3-Clause. Copyright © 2013-2014 trackpy contributors https://github.com/soft-matter/trackpy. All rights reserved.
- License for [Cellpose](https://github.com/MouseLand/cellpose): BSD 3-Clause. Copyright © 2020 Howard Hughes Medical Institute
