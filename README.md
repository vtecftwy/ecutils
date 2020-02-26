# ecutils package
Collection of my utility functions usable across several projects and applications, both locally and on hosted VMs.
This is provided as it is, feel free to use it but no support is provided.

## Installation:
Two options:
- install in `develop` mode from local source:
    - `pip install -e .` from the project folder, or
    - `pip install -e "path to local source code directory"`
- install in from github for hosted VMs:
    - `pip install git+git:https://github.com/vtecftwy/ecutils.git@master`
    - `pip install git+git:https://github.com/vtecftwy/ecutils.git@develop`

## Modules:
### General use:
- general_utils
- ipython_utils

### Data Science and Machine learning:
- eda_stats_utils
- fastai_utils (WIP)

### Handling of images
- image_utils