[![DOI](https://zenodo.org/badge/351403602.svg)]()

# Introduction
This project consists of two notebooks that are using the [Julia](https://julialang.org) programming language.
Some physics background is provided in the first notebook.

In the notebook [fbs.ipynb](fbs.ipynb) we want to discuss a way to determine a specific parameter for numerical simulation.
This procedure has been used to determine numerical parameters for simulations with a bigger Julia program.

That program can be found in the folder [EdgeImp](EdgeImp).
Usage is explained in the readme there.

The results of those simulations are a large number of .csv files and in the data folder there is a small sample set of data created with the TopoImp program.
The notebook [evaluation.ipynb](evaluation.ipynb) explaines how to read these data files and perform a typical analysis.

The full results are currently being used to prepare a manuscript.

# Running the notebooks
If you don't want to install anything you can simply use the following binder link: [![Binder](https://mybinder.org/badge_logo.svg)]()

To run it locally on your computer simply download this repo and unzip it.
I'm assuming you have a conda installation and can run jupyter notebook.
Install Julia (download it from the [official website](https://julialang.org/downloads/) and follow the installation instructions.), if you haven't already and make sure to add the kernel to jupyter notebook (instructions for example [here](https://github.com/JuliaLang/IJulia.jl)).

Open a jupyter notebook instance as usual and whenever you start one of the notebooks the Project.toml (and Manifest.toml) should be detected automatically (corresponds to environments in python).

In any case now you can either run the cells sequentially or choose to run all by going to 'Run->Run all cells' in the dropdown menues. 

# Used Julia Packages
* [Plots](http://docs.juliaplots.org): for plotting, similar to matplotlib in python (has several backends e.g. GR, PyPlot, plotly and several more)
* [StatsPlots](https://github.com/JuliaPlots/StatsPlots.jl): extends the Plots package for common statistics plots and contains plot recepies for DataFrames
* [Interact](https://juliagizmos.github.io/Interact.jl/latest/): for interactive plot widgets, sliders, etc.
* [WebIO](https://github.com/JuliaGizmos/WebIO.jl): not explicitly loaded, but needed for interactive plots
* [LaTeXStrings](https://github.com/stevengj/LaTeXStrings.jl): for LaTeX type setting in the plot labels
* [CSV](https://csv.juliadata.org/stable/): library to handle .csv files
* [DataFrames](https://dataframes.juliadata.org/stable/): package for handling datasets (similar to pandas for python)
* [Statistics](https://docs.julialang.org/en/v1/stdlib/Statistics/): standard library for statistics operations like for example mean values
* [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/): standard library for linear algebra operations like for example eigenvalues
