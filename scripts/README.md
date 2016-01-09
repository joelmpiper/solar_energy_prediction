This directory contains all of the scripts used for the General Assembly project on the old Kaggle Solar Energy competition. I started out with the beat-the-benchmark code from the forums (not uploaded to github). The other files in this directory are described below.

- explore\_training.py borrows much of the benchmark code to send back dataframes associated with the input and output data. I also attempted to convert the entire input to a single dataframe, but that turned out to be about a factor of 2 too large for my computer.

- dfsubset.py creates dataframes given subsetting parameters on the X and/or y values. Currently, one can select on the station names considered, latitude, longitude, and elevation on just the y values.

TODO: Build in subsetting for the X values using latitude, longitude, elevation, time, model, aggregations?, variables, station name grids?. After this, maybe automate a few simple plots and the analysis framework (heavy lift is getting variables into the form for the analysis, which is very different from that needed for the dataframes.
