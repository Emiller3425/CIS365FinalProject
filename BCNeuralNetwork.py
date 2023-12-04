# This file trained the neural network that we used for our Behavioral Cloning agent. It reads in the data
# saved from the BCNetDataGenerator file, modifies it as needed, then trains on all of it.

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

load_directory = "D:\\smashdataset\\parseddata\\"

# Loop over every file in the directory where we saved all our data.
for directory, subdirectory, data_files in os.walk(load_directory):
    # This will contain all of the gamestate data for the games in a directory.
    x_games = []
    # This will contain all of the controller data for the player that the NN is playing as.
    y_games = []
    # This line was only used once, hence it being commented out, but this is used to normalize stage
    # data because I forgot to do that in gathering the data.
    #enclosing_folder = ""

    # We train per stage directory rather than loading in all the files at once. It turns out that even with
    # memory mapped arrays, we didn't have enough memory to do everything at once.
    for file in tqdm(data_files):
        # Again, this is used for normalization.
        #enclosing_folder = os.path.basename(directory)

        # There were three crashes throughout the course of the data writing process.
        # That may mean incomplete data. Accidentally having some arrays a few entries
        # off from others would defeat the purpose of this exercise, hence all of this code.
        #
        # Split components of the filenames:
        file_breakdown = file.split("-")
        # Start with the gamestate data files:
        if file_breakdown[3] == "x.npy":
            # Ensure that the X file has a matching y0 and y1 file:
            if (
                    f"{file_breakdown[0]}-{file_breakdown[1]}-{file_breakdown[2]}-y0.npy" in data_files
                    and
                    f"{file_breakdown[0]}-{file_breakdown[1]}-{file_breakdown[2]}-y1.npy" in data_files
            ):
                # If p1 is a Fox player, load in the X data and the Y0 data. The data proved to be far more Fox heavy
                # than Jigglypuff heavy, so we chose to disregard any files that didn't have Fox in them.
                if file_breakdown[1] == "1":
                    path = os.path.join(directory, file)
                    # Note that we're loading these in memory mapped. This is because there is not enough memory to load the entire
                    # set in in most cases. We have >100GB of data.
                    x_games.append(numpy.load(path, mmap_mode='r+'))
                    y_games.append(numpy.load(f"{path[:-5]}y0.npy", mmap_mode='r+'))
                # If p1 is not a Fox player but p2 is, load in the X data and the Y1 data.
                elif file_breakdown[2] == "1":
                    path = os.path.join(directory, file)
                    x_games.append(numpy.load(path, mmap_mode='r+'))
                    y_games.append(numpy.load(f"{path[:-5]}y1.npy", mmap_mode='r+'))

    # This is the part that actually normalizes the stage data in each of these arrays. Once this was done once, it was
    # saved to disk, so we didn't have to do it all again. Note that we divide by 26 rather than 29 because the stage represented
    # by 29 had zero replays, so we divided by the next highest value.
    # for x in tqdm(x_games):
    #     new_x = int(enclosing_folder)/26
    #     x[:, 0] = new_x

    # If there's nothing in one of the subdirectories, go to the next one.
    if len(x_games) == 0:
        continue

    # Now that we've loaded in all of the game data and we know that input frames map to output frames, we don't care about which
    # games they came from. This code will flatten out the arrays such that they are treated as one continuous set of
    # input data/output data pairs.
    x_games = numpy.concatenate(x_games, axis=0)
    y_games = numpy.concatenate(y_games, axis=0)
    print("Concatenated.")

    # Convert to dtype=float32 to speed up training. These are stored as float64, and since things took awhile as-is I can only imagine how bad
    # it would've been had we not done this.
    x_games = x_games.astype(numpy.float32, copy=False)
    y_games = y_games.astype(numpy.float32, copy=False)
    print("Casted.")

    # Ensure that things were done right with the pruning up above. Want to make sure that every gamestate has a matching controller input.
    num_entries = tf.shape(x_games)[0]
    if num_entries != tf.shape(y_games)[0]:
        print("ERROR, Y entries not match X entries.")
        sys.exit(1)

    # Divide into training/testing sets using scikitlearn, 20% test and 80% train.
    x_train, x_test, y_train, y_test = train_test_split(x_games, y_games, test_size=0.2, random_state=20)

    # If this isn't the first loop, load in the model, otherwise we'll create it.
    if os.path.isdir(f"saved_symmetric_melee_model"):
        bc_model = keras.models.load_model(f"saved_symmetric_melee_model")
        print("Loaded model.")

    else:
        # Define model. The details of our various parameter changes and experiments are outlined in our writeup doc, but we tried several different
        # configurations here. Generally, though, this was patterned off of the model in an "Intro to TensorFlow" video tutorial series we found on
        # YouTube. The particular video that helped us out the most can be found here: https://youtu.be/pAhPiF3yiXI
        bc_model = keras.Sequential([
            # Input layer.
            keras.Input(shape=23),
            # This is one of our symmetric models, with the extra hidden layer.
            layers.Dense(115, activation='relu'),
            layers.Dense(115, activation='relu'),
            layers.Dense(115, activation='relu'),
            # We also tried making this layer a sigmoid activation instead of ReLU, thinking it might give better output results.
            layers.Dense(9, activation='relu'),

        ])

        bc_model.compile(
            # The loss function used in the example video didn't work with our data set, so we dug around and decided MSE was the best option.
            # Evidently it's the default for a reason and you really shouldn't mess with it unless you're sure you need to. Here's one source
            # that explained this to us: https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
            loss='mean_squared_error',
            # Again, just what was in the video. We would've experimented with more possible values/optimizers had we had the time.
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            # Our goal is to be as close as possible to human inputs at any given state, which is why we use accuracy.
            metrics=["accuracy"]
        )

    # Train the network. The batch size was settled on because we wanted frequent updates but we also wanted the process to not take far too long.
    # Ultimately, we probably could've afforded to increase this a bit, but likely to even worse results. We probably would've toyed with things like batch
    # size and number of epochs more had we had more time to spend training.
    bc_model.fit(x_train, y_train, batch_size=512, epochs = 50, verbose = 2, shuffle=True)
    # Show test results at the end of each training step.
    bc_model.evaluate(x_test, y_test, batch_size=512, verbose=2)
    # Save the network so that the next loop can pick up here.
    bc_model.save(f"saved_symmetric_melee_model/")

    # Delete arrays to make space for next loop.
    del(x_games)
    del(y_games)

    # Free VRAM
    keras.backend.clear_session()#