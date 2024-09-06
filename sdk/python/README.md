# Kubeflow Traning-Operator SDK APIs Documentation!

Python SDK for Training Operator

## Requirements.

Python >= 3.8

Training Python SDK follows [Python release cycle](https://devguide.python.org/versions/#python-release-cycle)
for supported Python versions.

## Installation & Usage

### pip install

```sh
pip install kubeflow-training
```

Then import the package:

```python
from kubeflow import training
```

### Setuptools

Install via [Setuptools](http://pypi.python.org/pypi/setuptools).

```sh
python setup.py install --user
```

(or `sudo python setup.py install` to install the package for all users)

## Getting Started

Please follow the [Getting Started guide](https://www.kubeflow.org/docs/components/training/overview/#getting-started)
or check Training Operator [examples](../../examples).

## Building conformance tests

Run

```
docker build . -f Dockerfile.conformance -t <tag>
```
