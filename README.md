# rSNAPed
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Screenshot](.docs/images/rSNAPed_Logo.png)

rSNAPed a Python library for single-molecule image processing.

Author Luis U. Aguilera, William Raymond, Brooke Silagy, Brian Munsky.

## Description

The code is intended to automatically track single-molecules from single-cell videos. The code calculates spot position and extract intensity values.

Compendium of libraries including `cellpose`, `rsnapsim`, `trackpy`.

![Screenshot](https://github.com/MunskyGroup/image_processing_toolbox/blob/master/rSNAPsim_IP/General_Documents/Images_for_github/rSNAPsimIP_Pipeline.png)


## Usage

* **Single-molecule translation**
* **Single-molecule transcription**
* **FISH**

Example using real data.

![Screenshot](https://github.com/MunskyGroup/image_processing_toolbox/blob/master/rSNAPsim_IP/General_Documents/Images_for_github/screenshot_3.png)

## Simulating translation

The code is intended to simulated single-
molecule translation. A  video with the simulated cell and a data frame containing spot and intensity positions are generated. This simulation can be used to train new algorithms or for training new students.

![Screenshot](https://github.com/MunskyGroup/image_processing_toolbox/blob/master/rSNAPsim_IP/Simulated_Cell/Development/Gifs/output.gif)

## Installation

First make sure that you have installed the following packages. For this, you can use the package manager [pip](https://pip.pypa.io/en/stable/).
```bash
pip install rsnaped
```
## dependencies.

matplotlib (3.2.2) <br />
numpy (1.20.1) <br />
pandas (1.0.5) <br />
scikit-image (0.18.0) <br />
joblib (0.16.0) <br />
bqplot (0.12.17) <br />
scipy (1.5.0) <br />
cellpose (0.5.1) <br />
pyfiglet (0.8.post1) <br />
tifffile (2020.10.1) <br />
opencv-python (4.4.0.42) <br />
trackpy (0.4.2) <br />
ipywidgets (7.5.1) <br />

## To install all packages please open installation.ipynb and run the notebook.
![Screenshot](https://github.com/MunskyGroup/image_processing_toolbox/blob/master/rSNAPsim_IP/General_Documents/Images_for_github/screenshot_4.png)


## If you want to manually install the dependencies run the following lines on the terminal

```bash
pip3 install scikit-image
pip3 install ipywidgets
pip3 install bqplot
pip3 install scipy
pip3 install pyfiglet
pip3 install opencv-python
pip3 install cellpose
pip3 install trackpy
pip3 install ipywidgets
```




## License
[MIT](https://choosealicense.com/licenses/mit/)
