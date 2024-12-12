# NFL Data Visualization and Predictive Modeling with FCNs and CNNS

## for Physics 5680 - Big Data Analytics & Machine Learning (Final Project)

### by Connor Fricke

This project makes use of the data available [here](https://github.com/ryurko/nflscrapR-data). By training neural network models on this data,
one can hope to make accurate predictions of game outcomes or player performances. Such models are highly desirable in the world of sports betting, as 
the best performing models can generate income while limiting financial risk.

This project uses Python scripts and Jupyter notebooks with the Tensorflow machine learning package in order to train and test fully-connected and
convolutional neural networks. Various hyperparameter sets will be tested for both CNNs and FCNs. An accuracy of >75% is a desirable outcome of the
training process. 

Data must be explored, visualized, cleaned, engineered, then split into training and validation subsets prior to training the models. The current method
utilizes K-Fold validation over each year of available data (2009-2019). Play-by-play data is pre-processed and normalized
to produce "images" which are pixelized representations of each football game in the dataset. Each year of data provides about
330 games, labeled either 1 or 0 based on whether the home team won or lost.

The files `FCN_training.py` and `CNN_training.py` are script versions of the `prep_data.ipynb` notebook. In this notebook and the corresponding
scripts, the data is prepped for input to the models, the models are trained, then K-fold validation is used to obtain accuracy and loss metrics on
all 11 years of validation data.

Several hyperparameter sets have been tested using these scripts, by repeating the training process several times with each
set of hyperparameters to get an idea of the effect of the complexity of the model on its performance.

Data is cleaned using `pbp_cleaning.ipynb`. Some preliminary visuals (plots) are included in `game_visuals.ipynb` and `pbp_visuals.ipynb`.
The data obtained directly from the `nflscrapR-data` repo is in the `data` directory, while a version of the data that has been cleaned
is in the `cleaned_data` directory. The trained models along with printouts of the accuracy and loss metrics for each are stored in the `models` directory.
