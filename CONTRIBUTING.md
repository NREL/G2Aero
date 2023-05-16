Feedback, Support, and Contributions
====================================

Contributions are always welcome! 
To contribute to G2aero, report an issue, or seek support, please initiate a pull request or issue through the [project GitHub](https://github.com/NREL/G2Aero).

## Setup development environment
### Install package in development mode
To setup your development environment you can install the latest version directly from the most up-to-date version 
of the source code by cloning/forking the GitHub repository

```bash
git clone https://github.com/NREL/G2Aero.git
```

Once you have the source, you can build G2Aero in development mode by executing

```bash
python setup.py develop
```
or

```bash
pip install -e .
```

in the top-level directory. The required Python packages will automatically be
installed as well.

### Testing
To test that the package is working correctly, run

```bash
pytest
```
from the root directory of the package.
This will run a basic test problem.

## Report an issue
Issues can be submitted at https://github.com/NREL/G2Aero/issues
### Reporting a bug
Please report bugs by submitting an issue.

If you are reporting a bug, please include the following information:

- A quick summary and/or background.
- Your operating system name and version.
- Details about your local setup that might be helpful in troubleshooting e.g. python version, library versions
- Detailed steps to reproduce the bug.
- What you expected to happen.
- What actually happens.

### Proposing a new feature
The best way to propose a new feature is by submitting an issue.

To propose a feature please include:

- Describe in detail how the new feature would work.
- Explain the use case of the new feature.
- Please keep the scope as narrow and specific as possible, to make it easier to implement.

### Request for documentation
The latest documentation for the `G2Aero` library is available at https://g2aero.readthedocs.io/en/latest/index.html

If any documentation is unclear or requires correction, please submit an issue.

## Submitting changes
To submit your code when fixing bugs, documentation, or implementing new features, please follow the steps below.

1. Fork the `G2Aero` repository on GitHub.
2. Clone your fork locally:
```bash
git clone https://github.com/NREL/G2Aero.git
```
3. Create a branch for local development:
```bash
git checkout -b name-of-your-bugfix-or-feature
```
4. Make your desired changes on your local branch.
5. Commit your changes and push your branch to GitHub:
``` bush
git add .
git commit -m "Your detailed description of your changes."
git push origin name-of-your-bugfix-or-feature
```
6. Submit a pull request through GitHub.

## Support
If you are having any issues with the code or the `G2Aero` library in general, 
please don’t hesitate to reach out at: [olga.doronina@nrel.gov](mailto:olga.doronina@nrel.gov)