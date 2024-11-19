# Quantum-Optimized Machine Learning of Boolean Networks

The following repository concerns the project of the _Quintic Shock_ team of the
_2024 GTG Hackathon_. In general, it is a Python library for performing
optimizations on the training process of networks of elementary boolean
functions using neutral-atom quantum computers.

## Contributing

### Workstation Requirements

To develop, run, and test this application locally one must ensure that they
have met the following developer workstation requirements:

1. Installed Python v3.11
2. Installed [Poetry](https://python-poetry.org/)
3. Installed Jupyter Desktop.

### Environment Setup

We use [Poetry](https://python-poetry.org/) to manage Python dependencies and
virtual environments. Once the repository has been cloned, one needs to first
direct Poetry to create a Python virtual environment based on Python 3.11 like
so:

```bash
poetry env use python3.11
```

From there, the project's dependencies can be downloaded and installed into the
virtual environment with:

```bash
poetry install
```
