# rSNAPed

<img src="./docs/images/logo/rSNAPed_Logo.png" width="200" />

rSNAPed : RNA Sequence to NAscent Protein Experiment Designer.

Authors: Luis U. Aguilera, William Raymond, Brooke Silagy, Brian Munsky.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> :warning: **This software is in a very early and experimental stage**: at this point, it is intended to be used for testing and debugging purposes!

## Description

This library is intended to quantify single-molecule gene expression experiments. Specifically, the code uses [Cellpose](https://github.com/MouseLand/cellpose) to segment the cell in the image. Then, it uses [Trackpy](http://soft-matter.github.io/trackpy/dev/index.html) to detect spots inside the mask. Finally, it uses the spot position to quantify the spot intensity. The code also generates simulated data using [rSNAPsim](https://github.com/MunskyGroup/rSNAPsim).

## Usage

* Tracking for single-molecule translation (RNA + nascent protein) spots.
* Tracking for single-molecule RNA spots.
* RNA detection spots for FISH images.
* Simulating the single-molecule translation for any gene.
* Design of single-molecule gene expression experiments.

## Simulating translation

The code is intended to simulated single-molecule translation. A  video with the simulated cell and a data frame containing spot and intensity positions are generated. This simulation can be used to train new algorithms or for teaching new students.

## Installation

Open the terminal and use [pip](https://pip.pypa.io/en/stable/) for the installation:
```bash
pip install rsnaped
```
## Dependencies
cellpose (0.5.1) <br />
trackpy (0.4.2) <br />
matplotlib (3.2.2) <br />
numpy (1.20.1) <br />
pandas (1.0.5) <br />
scikit-image (0.18.0) <br />
joblib (0.16.0) <br />
bqplot (0.12.17) <br />
scipy (1.5.0) <br />
pyfiglet (0.8.post1) <br />
tifffile (2020.10.1) <br />
opencv-python (4.4.0.42) <br />
ipywidgets (7.5.1) <br />

## References

- [rSNAPsim](https://github.com/MunskyGroup/rSNAPsim):
 Aguilera, Luis U., et al. "Computational design and interpretation of single-RNA translation experiments." PLoS computational biology 15.10 (2019): e1007425.

- [Trackpy](http://soft-matter.github.io/trackpy/dev/index.html):
 Dan Allan, et al. (2019, October 16). soft-matter/trackpy: Trackpy v0.4.2 (Version v0.4.2). Zenodo. http://doi.org/10.5281/zenodo.3492186

- [Cellpose](https://github.com/MouseLand/cellpose):
 Stringer, Carsen, et al. "Cellpose: a generalist algorithm for cellular segmentation." Nature Methods 18.1 (2021): 100-106.

## Licenses for dependencies
- License for [rSNAPsim](https://github.com/MunskyGroup/rSNAPsim): MIT. Copyright © 2018 Dr. Luis Aguilera, William Raymond
- License for [Trackpy](http://soft-matter.github.io/trackpy/dev/index.html): BSD-3-Clause.
Copyright © 2013-2014 trackpy contributors
   https://github.com/soft-matter/trackpy
   All rights reserved.
- License for [Cellpose](https://github.com/MouseLand/cellpose): BSD 3-Clause.
Copyright © 2020 Howard Hughes Medical Institute
