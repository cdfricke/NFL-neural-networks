# Programmer: Connor Fricke
# File: CNN_training.py
# Latest Revision: 5-Nov-2024
# Desc: Python script for training CNN models of various hyperparameter sets.
# This file follows along with the code done in the notebook "prep_data.ipynb",
# but implements the model training portion in a loop over various parameters.


import pandas as pd
from myFuncs import get_image
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout
from keras import models

years = [x for x in range(2009, 2020)]

# **********************
# ** READ IN PBP DATA **
# **********************
print("Reading cleaned PBP Data...")
preseason_pbp = {}
regseason_pbp = {}
pstseason_pbp = {}
for year in years:
    preseason_pbp[year] = pd.read_csv(f"../cleaned_data/pbp_data/pre_season/pre_pbp_{year}.csv", low_memory=False)
    regseason_pbp[year] = pd.read_csv(f"../cleaned_data/pbp_data/regular_season/reg_pbp_{year}.csv", low_memory=False)
    pstseason_pbp[year] = pd.read_csv(f"../cleaned_data/pbp_data/post_season/post_pbp_{year}.csv", low_memory=False)

all_pre_pbp = pd.concat(preseason_pbp)
all_reg_pbp = pd.concat(regseason_pbp)
all_pst_pbp = pd.concat(pstseason_pbp)
df_pbp = pd.concat([all_pre_pbp, all_reg_pbp, all_pst_pbp])

# ***********************
# ** READ IN GAME DATA **
# ***********************
print("Reading Game Data...")
preseason_games = {}
regseason_games = {}
pstseason_games = {}
for year in years:
    preseason_games[year] = pd.read_csv(f"../cleaned_data/games_data/pre_season/pre_games_{year}.csv")
    regseason_games[year] = pd.read_csv(f"../cleaned_data/games_data/regular_season/reg_games_{year}.csv")
    pstseason_games[year] = pd.read_csv(f"../cleaned_data/games_data/post_season/post_games_{year}.csv")

all_pre_games = pd.concat(preseason_games)
all_reg_games = pd.concat(regseason_games)
all_pst_games = pd.concat(pstseason_games)
df_games = pd.concat([all_pre_games, all_reg_games, all_pst_games])

# *************************
# ** PREPROCESS PBP DATA **
# *************************
print("\nPreprocessing PBP Data...")
print("INITIAL SHAPE:", df_pbp.shape)

print("Dropping duplicates, filling NaNs...")
df_pbp.drop(columns=['Unnamed: 0', 'play_id'], inplace=True)
df_pbp.fillna(value=0)
df_pbp.drop_duplicates(inplace=True)
df_pbp.reset_index(drop=True, inplace=True)

# Select plays where the team of possession is the home team
print("Isolating home team possession plays...")
df_pbp = df_pbp[df_pbp['posteam_type'] == 'home']
df_pbp.reset_index(drop=True, inplace=True)

# get NUMERIC features of pbp data
print("Isolating numeric features...")
df_pbp_numeric = df_pbp.select_dtypes(include="number")
numeric_features = df_pbp_numeric.columns.tolist()

# scale features and isolate image data
# WE WILL USE COLUMNS 29,30,33,37-88
print("Isolating image features and scaling them...")
scaler = MinMaxScaler()
image_features = numeric_features[29:31] + [numeric_features[33]] + numeric_features[37:89]
df_pbp[image_features] = scaler.fit_transform(df_pbp[image_features])

print("FINAL SHAPE:", df_pbp.shape)

# **************************
# ** PREPROCESS GAME DATA **
# **************************
print("\nPreprocessing Game Data...")
print("INITIAL SHAPE:", df_games.shape)

print("Dropping duplicates, filling NaNs...")
df_games.drop(columns=['Unnamed: 0'], inplace=True)
df_games.dropna(inplace=True)
df_games.drop_duplicates(inplace=True)
df_games.reset_index(drop=True, inplace=True)

# A bit of feature engineering
print("Running Feature Engineering...")
df_games['home_win'] = df_games['home_score'] > df_games['away_score']
df_games['home_win'] = df_games['home_win'].astype(int)

print("FINAL SHAPE:", df_pbp.shape)

# **********************
# ** TRAIN TEST SPLIT **
# **********************
print("\nSplitting data into test and train...")
VAL_YEAR = 2009
df_train = df_pbp[df_pbp['season'] != VAL_YEAR]
df_test = df_pbp[df_pbp['season'] == VAL_YEAR]
train_games = df_train['game_id'].unique()
test_games = df_test['game_id'].unique()
print("All PBP data shape:", df_pbp.shape)
print("Training data shape:", df_train.shape)
print("Testing data shape:", df_test.shape)
print("Games in training data:", len(train_games))
print("Games in testing data:", len(test_games))

# *******************
# ** BUILD X AND Y **
# *******************
print('\nPopulating X (images) for train and test...')
# This parameter sets the number of plays to capture from each game
IMAGE_H = 40
IMAGE_W = len(image_features)
IMAGE_SHAPE = (IMAGE_H, IMAGE_W, 1)
print("Image dimensions:", IMAGE_SHAPE)

X_train = []
X_test = []

for game_id in train_games:
    image = get_image(df=df_train, game_id=int(game_id), columns=image_features)
    X_train.append(image[:IMAGE_H].reshape(IMAGE_SHAPE))
for game_id in test_games:
    image = get_image(df=df_test, game_id=int(game_id), columns=image_features)
    X_test.append(image[:IMAGE_H].reshape(IMAGE_SHAPE))

print('\nPopulating Y (labels) for train and test...')
Y_train = []
Y_test = []

for game_id in train_games:
    for row in df_games.itertuples():
        if int(row.game_id) == int(game_id):
            Y_train.append(row.home_win)
            break
for game_id in test_games:
    for row in df_games.itertuples():
        if int(row.game_id) == int(game_id):
            Y_test.append(row.home_win)
            break

print("Converting to numpy arrays...")
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("Y_train:", Y_train.shape)
print("Y_test :", Y_test.shape)

# *****************
# ** BUILD MODEL **
# *****************
NUM_FILTERS = 25
FILTER_SIZE_0 = 5
FILTER_SIZE_1 = 3
POOLING_SIZE_0 = 2
POOLING_SIZE_1 = 2
HIDDEN_SIZE = 64
DROPOUT = 0.2
model_hyp = {'NUM_FILTERS': NUM_FILTERS,
             'FILTER_SIZE_0': FILTER_SIZE_0,
             'FILTER_SIZE_1': FILTER_SIZE_1,
             'POOLING_SIZE_0': POOLING_SIZE_0,
             'POOLING_SIZE_1': POOLING_SIZE_1,
             'HIDDEN_SIZE': HIDDEN_SIZE,
             'DROPOUT': DROPOUT
             }

print("\nBuilding CNN model...")
print(model_hyp)
model = models.Sequential(layers=[
    Input(IMAGE_SHAPE),
    Conv2D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE_0, activation='relu'),  # first conv layer
    MaxPooling2D(pool_size=POOLING_SIZE_0),                                     # first pooling layer
    Conv2D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE_1),                     # second conv layer
    MaxPooling2D(pool_size=POOLING_SIZE_1),                                     # second pooling layer
    Flatten(),                                                                  # flatten for input to FCN
    Dense(units=HIDDEN_SIZE, activation='relu'),                                # Hidden layer of FCN
    Dropout(DROPOUT),                                                           # Dropout layer to avoid overfitting
    Dense(units=1, activation='sigmoid')                                        # Output layer of FCN (1 or 0)
])

LOSS_FUNC = 'binary_crossentropy'
OPTIMIZER = keras.optimizers.Adam(learning_rate=0.0001)
print(f"Compiling model... (loss: {LOSS_FUNC}, opt: {OPTIMIZER})")
model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNC, metrics=['accuracy'])

# *****************
# ** TRAIN MODEL **
# *****************
PATIENCE = 10
CALLBACKS = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE),
             keras.callbacks.ModelCheckpoint('../models/best_CNN_model.keras', save_best_only=True, verbose=1)]
EPOCHS = 50
BATCH_SIZE = 32

print(f'Training model... (epochs: {EPOCHS}, batch size:{BATCH_SIZE})')
training_results = model.fit(
    x=X_train,
    y=Y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=CALLBACKS,
    validation_data=(X_test, Y_test),
    verbose=1
)
history = training_results.history
best_acc = history['accuracy'][-PATIENCE-1]
best_val_acc = history['val_accuracy'][-PATIENCE-1]
best_loss = history['loss'][-PATIENCE-1]
best_val_loss = history['val_loss'][-PATIENCE-1]
print(f"Training Complete! (val accuracy: {round(best_val_acc, 4)}, val loss: {round(best_val_loss, 4)})")
