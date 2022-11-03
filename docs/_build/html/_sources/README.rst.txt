# G2Aero Documentation

## Organization

The documentation is broken up into three separate folders:

1. How-To Guides: These are walkthroughs intended to help understand key steps in the setup and run process.
2. Explanation: Information that provides theory and background and other topics where appropriate.
3. Technical Reference: Programming details for the G2Aero modules, mostly auto-generated from module docstrings.

## Building Documentation

For developers interested in building the documentation locally for testing purposes, carry out the following steps:

1. In order to build the documentation, you will need a version of Sphinx installed in your Conda environment (run `conda install -c conda-forge sphinx=4.2.0` followed by `pip install sphinx-rtd-theme`).
2. Change directory to `G2Aero/docs`
3. Run `make html`
4. Open the file `_build/html/index.html`

The philisophical organization of the documentation is shown above.  From a file system perspective, this means storing documentation using the following directory structure:

```
docs
|  README.md
|  index.rst
|  conf.py
|  make.bat
|  Makefile
|  ...
|
|--explanation
|  |  index.rst
|
|--how_to_guides
|  |  index.rst
|  |  guide_1.rst
|  |  guide_2.rst
|  |  ...
|
|--technical_reference
|  |  index.rst
|  |  module_1.rst
|  |  module_1.rst
|  |  ...
|
|--_build
|  |  doctrees
|  |  html
|  |  |  index.html
|  |  |  ...
|

```

where the `_build` directory and the resulting html files are the result of Step 3.  Each `index.rst` file enumerates the guides/pages in that section, while the parent `index.rst` file at the `docs/` level contains information to be shown on the home page and a higher-level presentation of the individual sections.