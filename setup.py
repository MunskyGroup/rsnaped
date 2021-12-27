from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# package requirements
with open("requirements.txt", encoding='utf-8') as f:
    requirements = [l.strip() for l in f.readlines() if l]

# package version
__version__ = None
with open('rsnaped/__init__.py', encoding='utf-8') as f:
    for row in f:
        if row.startswith('__version__'):
            __version__ = row.strip().split()[-1][1:-1]
            break

setup(
    name = "rsnaped",
    version = __version__,
    author = "Luis Aguilera, William Raymond, Brooke Silagy, Brian Munsky",
    author_email = "luisubald@gmail.com",
    description = ("Python module for single-molecule image processing."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license = "MIT",
    keywords = "single-molecule image processing",
    url = "https://github.com/MunskyGroup/rsnaped",
    package_dir = {'rsnaped':'rsnaped'},
    packages=find_packages(exclude=['docs','database','notebooks','__pycache__','.gitignore','.vscode','TODO.md']),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8"
    ],
    python_requires='>=3.8',
)
