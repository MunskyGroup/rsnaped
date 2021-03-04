from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "rsnaped",
    version = "0.0.3",
    author = "Luis Aguilera, William Raymond, Brian Munsky",
    author_email = "luisubald@gmail.com",
    description = ("Python module for single-molecule image processing."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license = "MIT",
    keywords = "single-molecule image processing",
    url = "https://github.com/MunskyGroup/rsnaped",
    package_dir ={'':'rsnaped'},
    packages=find_packages(where="rsnaped"),
    install_requires=['matplotlib','numpy>=1.20.1','numba>=0.52.0','pandas','scikit-image','joblib','bqplot','scipy','cellpose>=0.6.1', 'pyfiglet','tifffile','opencv-python','trackpy','ipywidgets','tqdm','rSANPsim','rsnapsim-ssa-cpp'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8"
    ],
    python_requires='>=3.7',
)
