import os

import keras.backend
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as k
from keras.layers import Dense
import sys
import numpy
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load in X data, appropriate Y data for Fox, where
# Jigglypuff = 15, Fox = 1. Recall that format is
# "id#-y0#-y1#-x/y0/y1.npy." This data is way more overwhelmingly Fox v. Fox
# than I was expecting, this might not go well against other characters.
#
# With the data we have, it doesn't make sense to do one stage as originally
# planned, so os.walk will read in all of them.
load_directory = "D:\\smashdataset\\parseddata\\"

for directory, subdirectory, data_files in os.walk(load_directory):
    x_games = []
    y_games = []
    #enclosing_folder = ""
    # Train per directory. I don't have enough memory for all of these at once, even with
    # memory mapping enabled.
    for file in tqdm(data_files):
        #enclosing_folder = os.path.basename(directory)
        # There were three crashes throughout the course of the data writing process.
        # That may mean incomplete data. Accidentally having some arrays a few entries
        # off from others would defeat the purpose of this exercise, hence all this.
        #
        # Split components of the filenames:
        file_breakdown = file.split("-")
        # Start with the X data files:
        if file_breakdown[3] == "x.npy":
            # Ensure that the X file has a matching y0 and y1 file:
            if (
                    f"{file_breakdown[0]}-{file_breakdown[1]}-{file_breakdown[2]}-y0.npy" in data_files
                    and
                    f"{file_breakdown[0]}-{file_breakdown[1]}-{file_breakdown[2]}-y1.npy" in data_files
            ):
                # If p1 is a Fox, load in the X data and the Y0 data.
                if file_breakdown[1] == "1":
                    path = os.path.join(directory, file)
                    x_games.append(numpy.load(path, mmap_mode='r+'))
                    y_games.append(numpy.load(f"{path[:-5]}y0.npy", mmap_mode='r+'))
                # If p1 not a Fox but p2 is, load in the X data and the Y1 data.
                elif file_breakdown[2] == "1":
                    path = os.path.join(directory, file)
                    x_games.append(numpy.load(path, mmap_mode='r+'))
                    y_games.append(numpy.load(f"{path[:-5]}y1.npy", mmap_mode='r+'))
    # I forgot to normalize stages in data collection so I initially did that here. It's already been done
    # once, though, and the files overwritten, so this will sit dormant now.
    # for x in tqdm(x_games):
    #     new_x = int(enclosing_folder)/26
    #     x[:, 0] = new_x

    if len(x_games) == 0:
        continue

    # Flatten these arrays. Our goal is to predict the Y entry for a given X entry, it doesn't matter which
    # particular game it came from now that we know we've loaded in the particular games we want.
    #
    # Thank you, Bing, for this code.
    x_games = numpy.concatenate(x_games, axis=0)
    y_games = numpy.concatenate(y_games, axis=0)
    print("Concatenated.")

    # Convert to dtype=float32 to speed up training.
    x_games = x_games.astype(numpy.float32, copy=False)
    y_games = y_games.astype(numpy.float32, copy=False)
    print("Casted.")

    num_entries = tf.shape(x_games)[0]
    if num_entries != tf.shape(y_games)[0]:
        print("ERROR, Y entries not match X entries.")
        sys.exit(1)

    # Divide into training/testing sets using scikitlearn, 20% test an 80% train.

    x_train, x_test, y_train, y_test = train_test_split(x_games, y_games, test_size=0.2, random_state=20)

    if os.path.isdir(f"saved_symmetric_melee_model"):
        bc_model = keras.models.load_model(f"saved_symmetric_melee_model")
        print("Loaded model.")

    else:
        # Define model:
        bc_model = keras.Sequential([
            keras.Input(shape=23),
            # The examples I've found tend to scale down, but I'm guessing this is a more complex problem.
            # I'm scaling up instead. I'm scaling up to something that should be nice and symmetrical because
            # that seems like a good idea. I'm not sure why. It just does.
            #
            # I've gone back to relu because I've seen a few people suggest it's better for backpropagation. I
            # do not want to figure out the logic behind backpropagation so I'm just trusting the internet here.
            # Asymmetric was layers.Dense(230, activation='relu'),
            layers.Dense(115, activation='relu'),
            layers.Dense(115, activation='relu'),
            # Was not there for Asymmetric.
            layers.Dense(115, activation='relu'),
            # Asymmetric was layers.Dense(9, activation='sigmoid'),
            layers.Dense(9, activation='relu'),

        ])

        bc_model.compile(
            # From this source: https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
            # Evidently you shouldn't change from MSE unless you know what you're doing. Given that the
            # error function from the demo NN I wrote using that video tutorial is apparently incompatible
            # with our dataset rn, I'll be using it.
            loss='mean_squared_error',
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"]
        )

    # Train and save
    # I have the memory for a higher batch size but evidently higher batch size means worse
    # general performance. This should take a very long time and I'm satisfied with that.
    bc_model.fit(x_train, y_train, batch_size=512, epochs = 50, verbose = 2, shuffle=True)
    bc_model.evaluate(x_test, y_test, batch_size=512, verbose=2)

    bc_model.save(f"saved_symmetric_melee_model/")

    # IMPORTANT NOTE: You will need to translate game inputs into a SUBARRAY and you will get a
    #                 SUBARRAY as output. Don't ask it to predict [data], ask it to predict
    #                 [[data]].
    #
    # IMPORTANT NOTE 2: I'm struggling to figure out how to appropriately round output values. This
    #                   would undoubtedly help accuracy but since I can't figure it out, we'll have
    #                   to do this in our player. DO NOT assume values are between 0 or 1 when doing
    #                   this.

    # Delete arrays to make space for next loop.
    del(x_games)
    del(y_games)

    # Free VRAM
    keras.backend.clear_session()