[![test](https://github.com/NREL/G2Aero/actions/workflows/python-package.yml/badge.svg)](https://github.com/NREL/G2Aero/actions/workflows/python-package.yml)
[![paper](https://github.com/NREL/G2Aero/actions/workflows/draft-pdf.yml/badge.svg)](https://github.com/NREL/G2Aero/actions/workflows/draft-pdf.yml)
[![Documentation Status](https://readthedocs.org/projects/g2aero/badge/?version=latest)](https://g2aero.readthedocs.io/en/latest/?badge=latest)

# G2Aero: Separable shape tensors for aerodynamic design
 `G2Aero` is a flexible and practical tool for design and deformation of 2D airfoils and 3D blades using data-driven approaches. `G2Aero` utilizes the geometry of matrix manifolds&mdash;specifically the Grassmannian&mdash;to build a novel framework for representing physics-based separable deformations of shapes. `G2Aero` offers the flexibility to generate perturbations in a customizable way over any portion of the blade. The `G2Aero` framework utilizes data-driven methods based on a curated database of physically relevant airfoils. Specific tools include: 
 
 -  principal geodesic analysis over normal coordinate neighborhoods of matrix manifolds; 
 -  a variety of data-regularized deformations to nominal 2D airfoil shapes; 
 -  Riemannian interpolation connecting a sequence of airfoil cross-sections to build 3D blades from 2D data; 
 -  consistent perturbations over the span of interpolated 3D blades based on dominant modes from the data-driven analysis. 

 More details can be found in the [G2Aero documentation](https://g2aero.readthedocs.io/en/latest/index.html).

## Installation

Install `G2Aero` from sources with Python3.x:

```bash
git clone https://github.com/NREL/G2Aero.git
cd G2Aero
python setup.py install
```

Installing via `conda-forge`
```bash
conda install -c conda-forge g2aero
```
## Testing
You can run the tests from the root `g2aero` folder (once you installed pytest):
```bash
pip install pytest
pytest
```

## Usage

<!-- ```python

``` -->
## Example 
Grassmannian interpolation combined with parametrized affine deformations:
<img src="https://github.com/NREL/G2Aero/blob/main/data/animations/animation.gif" alt="blade gif" title="gif" width="500"/>

## Contributing

Contributions are always welcome! See `contributing.md` for ways to get started.
Please adhere to this project's `code of conduct`.

## Citations
If you use this software in your research or publications, please cite the following paper:

    @article{Doronina_JOSS_2023, 
      author = {Olga A. Doronina and Zachary J. Grey and Andrew Glaws}, 
      title = {G2Aero: A Python package for separable shape tensors}, 
      journal = {Journal of Open Source Software}, 
      publisher = {The Open Journal}, 
      year = {2023}, 
      volume = {8}, 
      number = {89}, 
      pages = {5408},
      doi = {10.21105/joss.05408}, 
      url = {https://doi.org/10.21105/joss.05408}, 
    }

    @article{GreyJCDE2023,
      author = {Grey, Zachary J and Doronina, Olga A and Glaws, Andrew},
      title = "{Separable shape tensors for aerodynamic design}",
      journal = {Journal of Computational Design and Engineering},
      volume = {10},
      number = {1},
      pages = {468-487},
      year = {2023},
      month = {01},
      doi = {10.1093/jcde/qwac140},
      url = {https://doi.org/10.1093/jcde/qwac140},
    }

    @inproceedings{grassmannian2022,
       title={Grassmannian Shape Representations for Aerodynamic Applications},
       author={Olga Doronina and Zachary Grey and Andrew Glaws},
       booktitle={AAAI 2022 Workshop on AI for Design and Manufacturing (ADAM)},
       year={2022},
       url={https://openreview.net/forum?id=1RRU6ud9YC}
    }

<!-- ## License -->






