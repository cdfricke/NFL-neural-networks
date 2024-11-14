# NFL Data Visualization and Predictive Modeling with FCNs and CNNS

## for Physics 5680 - Big Data Analytics & Machine Learning (Final Project)

### by Connor Fricke

This project makes use of the data available [here](https://github.com/ryurko/nflscrapR-data). By training neural network models on this data,
one can hope to make accurate predictions of game outcomes or player performances. Such models are highly desirable in the world of sports betting, as 
the best performing models can generate income while limiting financial risk.

This project will use Python scripts and Jupyter notebooks with the Tensorflow machine learning package in order to train and test fully-connected and
convolutional neural networks. Various hyperparameter sets will be tested for both CNNs and FCNs. An accuracy of >75% is a desirable outcome of the
training process. 

Data must be explored, visualized, cleaned, engineered, then split into training and validation subsets prior to training the models. The current method
utilizes K-Fold validation over each year of available data (2009-2019).
