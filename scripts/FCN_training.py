# Programmer: Connor Fricke
# File: FCN_training.py
# Latest Revision: 5-Nov-2024
# Desc: Python script for training FCN models of various hyperparameter sets.
# This file follows CNN_training.py, but designs FCN models as opposed to CNNs.


import pandas as pd
from myFuncs import get_image
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
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

# scale features and isolate image data from numeric data
print("Isolating image features and scaling them...")
scaler = MinMaxScaler()
# image_features = numeric_features[29:31] + [numeric_features[33]] + numeric_features[37:89]
image_features = numeric_features[5:]
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

# ***********************
# ** LOOP OVER K-FOLDS **
# ***********************
for VAL_YEAR in years:
    print(f"\n*** CURRENT K FOLD: {VAL_YEAR} ***")
    # **********************
    # ** TRAIN TEST SPLIT **
    # **********************
    print("\nSplitting data into test and train...")
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
    MODEL_DIRECTORY = f'../models/val_year_{VAL_YEAR}/'  # directory where models and training results will be saved
    # map an ID to a set of model hyperparams
    set0 = {'DL0': 16,
            'DL1': 16,
            'DL2': 16,
            'ACT': 'tanh',
            'DROP': 0.5,
            'BATCH': 32
            }
    set1 = {'DL0': 16,
            'DL1': 32,
            'DL2': 16,
            'ACT': 'tanh',
            'DROP': 0.5,
            'BATCH': 32
            }
    set2 = {'DL0': 32,
            'DL1': 32,
            'DL2': 32,
            'ACT': 'tanh',
            'DROP': 0.5,
            'BATCH': 32
            }
    set3 = {'DL0': 16,
            'DL1': 16,
            'DL2': 16,
            'ACT': 'relu',
            'DROP': 0.5,
            'BATCH': 32
            }
    set4 = {'DL0': 16,
            'DL1': 32,
            'DL2': 16,
            'ACT': 'relu',
            'DROP': 0.5,
            'BATCH': 32
            }
    set5 = {'DL0': 32,
            'DL1': 32,
            'DL2': 32,
            'ACT': 'relu',
            'DROP': 0.5,
            'BATCH': 32
            }
    SET_IDS = [0, 1, 2, 3, 4, 5]
    hp_sets = [set0, set1, set2, set3, set4, set5]
    acc_loss_sets = []

    for ID in SET_IDS:
        print(f"\nBuilding FCN model... (hyp set {ID})")
        param_set = hp_sets[ID]
        print(param_set)
        model = models.Sequential(layers=[
            Input(IMAGE_SHAPE),
            Flatten(),
            Dense(units=param_set['DL0'], activation=param_set['ACT']),
            Dense(units=param_set['DL1'], activation=param_set['ACT']),
            Dense(units=param_set['DL2'], activation=param_set['ACT']),
            Dropout(param_set['DROP']),
            Dense(units=1, activation='sigmoid')
        ])

        LOSS_FUNC = 'binary_crossentropy'
        OPTIMIZER = keras.optimizers.Adam(learning_rate=0.0005)
        print(f"Compiling model... (loss: {LOSS_FUNC}, opt: adam)")
        model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNC, metrics=['accuracy'])

        # *****************
        # ** TRAIN MODEL **
        # *****************
        MODEL_PATH = MODEL_DIRECTORY + f'best_FCN_model_set{ID}.keras'
        PATIENCE = 5
        CALLBACKS = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE),
                     keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=0)]
        EPOCHS = 25
        BATCH_SIZE = param_set['BATCH']

        print(f'Training model... (epochs: {EPOCHS}, batch size: {BATCH_SIZE})')
        training_results = model.fit(
            x=X_train,
            y=Y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=CALLBACKS,
            validation_data=(X_test, Y_test),
            verbose=0
        )
        # Get performance metrics
        hist = training_results.history
        best_val_acc = hist['val_accuracy'][-PATIENCE-1]
        best_val_loss = hist['val_loss'][-PATIENCE-1]
        print(f"Training Complete! (val accuracy: {round(best_val_acc, 4)}, val loss: {round(best_val_loss, 4)})")
        acc_loss_sets.append((best_val_acc, best_val_loss))

    # *******************************
    # ** WRITE RESULTS TO TXT FILE **
    # *******************************
    DETAILS_PATH = MODEL_DIRECTORY + 'FCN_results.txt'
    file = open(DETAILS_PATH, "w")
    file.write("** MODEL RESULTS **\n")
    for ID in SET_IDS:
        (val_acc, val_loss) = acc_loss_sets[ID]
        file.write(f"** Set ID: {ID}\n")
        file.write(f"** Validation Accuracy: {round(val_acc, 4)}, Validation Loss: {round(val_loss, 4)}\n\n")

    best_acc = np.min([x[0] for x in acc_loss_sets])
    best_acc_id = np.argmin([x[0] for x in acc_loss_sets])
    file.write(f"Best Accuracy: {round(best_acc, 4)} for hyperparameter set: {best_acc_id}\n")
    file.close()
