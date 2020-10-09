# FYS-STK4155: Project 1
Investigating the effects of different regression and resampling methods on the Franke function and terrain data.

Folder `src` contains the python-files used in this project
  - `main.py` : main file to run, has different run modes depending on whether you want to perform OLS, Ridge or Lasso, and which data set
  - `unit_test.py` : contains a test to see if the bootstrap implementation functions as intended

Folder `benchmarks` contains a set of benchmark files for each section of the code. Setting `benchmark = True` in `main.py` should reproduce these results

Folder `report` contains the report for the project

Folder `figures` contains all the figures used in the report

Folder `datafiles` contain the data files used for the terrain data, as well as saved results from running the code
