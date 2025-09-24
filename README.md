# netam

Neural NETworks for antibody Affinity Maturation.

## pip installation

Netam is available on PyPI, and works with Python 3.9 through 3.11.

```
pip install netam
```

This will allow you to use the models.

However, if you wish to interact with the models on a more detailed level, you will want to do a developer installation (see below).

## Models

### Thrifty model of somatic hypermutation

This model is described in:

Sung, Johnson, Dumm, Simon, Haddox, Fukuyama, Matsen IV. [Thrifty wide-context models of B cell receptor somatic hypermutation](https://elifesciences.org/reviewed-preprints/105471). *eLife*. 2025 Mar. doi: 10.7554/elife.105471.1

The corresponding reproducible experiments are at [matsengrp/thrifty-experiments-1](https://github.com/matsengrp/thrifty-experiments-1/); see that repo's README for additional dependencies.


### Deep Natural Selection Model (DNSM)

This model is described in:

Matsen IV, Sung, Johnson, Dumm, Rich, Starr, Song, Bradley, Fukuyama, Haddox. [A sitewise model of natural selection on individual antibodies via a transformer-encoder](https://academic.oup.com/mbe/advance-article/doi/10.1093/molbev/msaf186/8222712). *Mol Biol Evol*. 2025 Jul;42(8):msaf186. doi: 10.1093/molbev/msaf186

The corresponding reproducible experiments are at [matsengrp/dnsm-experiments-mbe](https://github.com/matsengrp/dnsm-experiments-mbe/); see that repo's README for additional dependencies.


## Pretrained models

Pretrained models will be downloaded on demand, so you will not need to install them separately.

The models are named according to the following convention:

    ModeltypeSpeciesVXX-YY

where:

* `Modeltype` is the type of model, such as `Thrifty` for the "thrifty" SHM model or `DNSM` for Deep Natural Selection Models
* `Species` is the species, such as `Hum` for human
* `XX` is the version of the model
* `YY` is any model-specific information, such as the number of parameters

### Available Models

**Thrifty Models:**
- `ThriftyHumV0.2-20`, `ThriftyHumV0.2-45`, `ThriftyHumV0.2-59`: SHM models trained on human data

**DNSM Models:**
- `DNSMHumV1.0-1M`: 1M parameter Deep Natural Selection Model trained on human data
- `DNSMHumV1.0-4M`: 4M parameter Deep Natural Selection Model trained on human data

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



## Troubleshooting
* On some machines, pip may install a version of numpy that is too new for the
    available version of pytorch, returning an error such as `A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.2 as it may crash.` The solution is to downgrade to `numpy<2`:
    ```console
    pip install --force-reinstall "numpy<2"
    ```
