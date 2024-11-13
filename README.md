# netam

Neural NETworks for antibody Affinity Maturation.

## pip installation

Netam is available on PyPI, and works with Python 3.8 through 3.11.

```
pip install netam
```

This will allow you to use the models.

However, if you wish to interact with the models on a more detailed level, you will want to do a developer installation (see below).


## Pretrained models

Pretrained models will be downloaded on demand, so you will not need to install them separately.

The models are named according to the following convention:

    ModeltypeSpeciesVXX-YY

where:

* `Modeltype` is the type of model, such as `Thrifty` for the "thrifty" SHM model
* `Species` is the species, such as `Hum` for human
* `XX` is the version of the model
* `YY` is any model-specific information, such as the number of parameters

If you need to clear out the cache of pretrained models, you can use the command-line call:

    netam clear_model_cache


## Usage

See the examples in the `notebooks` directory.


## Developer installation

From a clone of this repository, install using:

    python3.11 -m venv .venv
    source .venv/bin/activate
    make install

Note that you should be fine with an earlier version of Python.
We target Python 3.9, but 3.11 is faster.


## Experiments

If you are running one of the experiment repos, such as:

* [thrifty-experiments-1](https://github.com/matsengrp/thrifty-experiments-1/)
* [dnsm-experiments-1](https://github.com/matsengrp/dnsm-experiments-1/)

you will want to visit those repos and follow the installation instructions there.
