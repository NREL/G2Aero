import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()


setuptools.setup(
    name='g2aero',
    version='0.1.0',
    packages=setuptools.find_packages(where="src"),
    project_urls={
        "Bug Tracker": "https://github.com/NREL/G2Aero",
    },
    license='BSD 3-Clause License',
    author='Olga Doronina',
    author_email='olga.doronina@nrel.gov',
    description='Separable shape tensors for aerodynamic applications',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        "License :: OSI Approved :: BSD 3-Clause License"
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=install_requires
)
