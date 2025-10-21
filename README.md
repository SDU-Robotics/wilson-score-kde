# Wilson Score KDE

## Installation Notes

### Installing the UV Package Manager

The [UV Package Manager](https://docs.astral.sh/uv/) helps keeping track of the needed python packages.

UV can be installed as [Standalone](https://docs.astral.sh/uv/getting-started/installation/) by:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Example

This repository includes a simple 1D example which showcases WSKDE.
In the example the WSKDE confidence bounds are calculated and plotted.
The example can be executued by cloning the repository, installing the
required python packages using uv sync, and running the example as
follows:

```bash
git clone https://github.com/SDU-Robotics/wilson-score-kde.git
cd wilson-score-kde
uv sync
uv run examples/simple_1D/simple_1D.py
```

The example demonstrates how to:

1. Define a bandwidth matrix H
1. Initialize WSKDE with H
1. Set the supervised dataset used for inference
1. Compute the z-confidence bounds as $p\pm\sigma$

(see examples/simple_1D.py for full example)

```python
from wskde.wskde import WSKDE
...
# Load x_train and y_train
# Define x_test
h = 0.5
H = torch.diag(torch.Tensor([h]))
wskde = WSKDE(H)
wskde.set_training_samples(x_train, y_train)
p_wskde, sigma_wskde = wskde(x_test, z=1.96)
```

## C++

Below shows how to build and install the C++ Wrapper for WSKDE Python implementation

### Install

Install the following packages:

```bash
sudo apt install python3-dev pybind11-dev
```

Navigate into the following folder while standing in the root folder of this repository:

```bash
cd cpp/Wrapper/
mkdir build
cd build/
source ../../../.venv/bin/activate
cmake ..
make
```

To see if the bindings to python works correctly execute the test:
```bash
./WSKDEWrapperTest
```

Afterwards, install the WSKDE Wrapper lib:
```bash
sudo make install
```

To see if the Wrapper lib was installed correctly, build and execute the example:
```bash
cd ../../example/
mkdir build
cd build/
cmake ..
make
./WSKDEWrapperLibExample
```

## Including WS-KDE in external projects

### Using Package from Remote

The WS-KDE package can be added directly to your uv project over https by:

- HTTPS: ```uv add git+https://github.com/SDU-Robotics/wilson-score-kde.git```

### Using Package from Local

To clone the python package and install it with pip, do the following:

1. Clone project from git
1. Activate your virtual python environment
1. cd to the root of the wilson-score-kde folder
1. ```pip install .```

## Citations
If you found this repository useful, please cite:

```
@inproceedings{iversen2025global,
  title={Global Optimization of Stochastic Black-Box Functions with Arbitrary Noise Distributions using Wilson Score Kernel Density Estimation},
  author={Iversen, Thorbj{\o}rn Mosekj{\ae}r and S{\o}rensen, Lars Car{\o}e and Mathiesen, Simon Faarvang and Petersen, Henrik Gordon}, 
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2025},
  organization={IEEE}
}
```

For reference to the original Wilson Score Kernel Density Estimation paper, please cite:

```
@inproceedings{sorensen2020wilson,
  title={Wilson score kernel density estimation for bernoulli trials},
  author={S{\o}rensen, Lars Car{\o}e and Mathiesen, Simon and Kraft, Dirk and Petersen, Henrik Gordon},
  booktitle={17th International Conference on Informatics in Control, Automation and Robotics (ICINCO)},
  pages={305--313},
  year={2020},
  organization={SCITEPRESS Digital Library}
}
```
