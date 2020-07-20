# Installation

Please install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) before proceeding to the next steps.

## Install packages

There are two sets of packages in the [requirements.txt](requirements.txt). Firstly, we have the packages required to train/evaluate the models and the second set is for the visualization module. Activate the python environment and install the required packages as:

```
pip install -r install/requirements.txt
```

## Build widgets and qgrid

We use widgets in Jupyter Notebook for the [Visualization](../Visualize.ipynb). Build the ipywidgets and qgrid for jupyterlab as:

```
conda install -c conda-forge nodejs
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install qgrid2
```

**NOTE**: If you do not wish to use conda and install nodejs with pip, it will not work. The nodejs version in the pip wheel is outdated. If you only wish to use jupyter notebook, you can enable them with:

```
jupyter nbextension enable --py widgetsnbextension
jupyter nbextension enable --py --sys-prefix qgrid
```

More details can be found under:
1. https://ipywidgets.readthedocs.io/en/latest/user_install.html
2. https://github.com/quantopian/qgrid