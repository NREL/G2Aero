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
    version='0.0.1',
    packages=setuptools.find_packages(where="src"),
    project_urls={
        "Bug Tracker": "https://github.com/NREL/G2Aero",
    },
    license='GPLv3',
    author='Olga Doronina',
    author_email='olga.doronina@nrel.gov',
    description='Grassmannian shape representation for Aerodynamic Applications',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=install_requires
)
