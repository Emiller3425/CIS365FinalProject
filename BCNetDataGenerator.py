# Goal is to create a program that can parse through a collection of .slp replay files and better organize
# data to be used to feed into a behavioral cloning network (created using TensorFlow, since  things are going to take
# a very long time if we don't make use of all the CUDA cores that I have sitting here). We could probably do these things
# at the same time, but there's a *lot* of data and parsing it all once means that's less overhead for every epoch of
# a network.
#
# Data set used comes from an outline of a similar project, which you can find here:
# https://bycn.github.io/2022/08/19/project-nabla.html
#
# Includes 94484 high-level games. We only got through ~79% of them, or ~74240 of them.

import os
import sys
import melee
import numpy
# Thank you, ChatGPT, for this suggestion. I was just asking it how I would multithread this application (spoiler alert, that didn't
# end up working), but it helpfully suggested this library. Were it not for this, I think I would've killed the process at hour 5.
from tqdm import tqdm

# This function probably could've been its own file, but it's here, so oh well. This actually formats the data for each frame.
def store_data(gamestate: melee.GameState, controller_ports: list):
    # This is the game state data.
    x = []
    # This is the first player's controller data.
    y0 = []
    # This is the second player's controller data.
    y1 = []

    # Generate X data
    # Stage. Notice that we forgot to normalize this value. That ends up getting done in the NN training file instead.
    x.append(gamestate.stage.value)
    # P1 XY (this is deprecated but I can't get position to work for some reason)
    x.append(gamestate.players[controller_ports[0]].x/500)
    x.append(gamestate.players[controller_ports[0]].y/500)
    # P2 XY
    x.append(gamestate.players[controller_ports[1]].x/500)
    x.append(gamestate.players[controller_ports[1]].y/500)
    # P1 Percent
    x.append(gamestate.players[controller_ports[0]].percent/999)
    # P2 Percent
    x.append(gamestate.players[controller_ports[1]].percent/999)
    # P1 Action
    x.append(gamestate.players[controller_ports[0]].action.value/397)
    # P2 Action
    x.append(gamestate.players[controller_ports[1]].action.value/397)
    # P1 Action Frame
    x.append(gamestate.players[controller_ports[0]].action_frame/250)
    # P2 Action Frame
    x.append(gamestate.players[controller_ports[1]].action_frame/250)
    # P1 Facing
    x.append(int(gamestate.players[controller_ports[0]].facing))
    # P2 Facing
    x.append(int(gamestate.players[controller_ports[1]].facing))
    # P1 Jumps
    x.append(gamestate.players[controller_ports[0]].jumps_left/2)
    # P2 Jumps
    x.append(gamestate.players[controller_ports[1]].jumps_left/2)
    # P1 Inv
    x.append(int(gamestate.players[controller_ports[0]].invulnerable))
    # P2 Inv
    x.append(int(gamestate.players[controller_ports[1]].invulnerable))
    # P1 Ground
    x.append(int(gamestate.players[controller_ports[0]].on_ground))
    # P2 Ground
    x.append(int(gamestate.players[controller_ports[1]].on_ground))
    # P1 Off
    x.append(int(gamestate.players[controller_ports[0]].off_stage))
    # P2 Off
    x.append(int(gamestate.players[controller_ports[1]].off_stage))
    # P1 Shield
    x.append(round(gamestate.players[controller_ports[0]].shield_strength/60, 1))
    # P2 Shield
    x.append(round(gamestate.players[controller_ports[1]].shield_strength/60, 1))

    # Generate Y Data
    controller_0: melee.PlayerState.controller_state = gamestate.players[controller_ports[0]].controller_state
    controller_1: melee.PlayerState.controller_state = gamestate.players[controller_ports[1]].controller_state
    # A button
    y0.append(int(controller_0.button[melee.Button.BUTTON_A]))
    y1.append(int(controller_1.button[melee.Button.BUTTON_A]))
    # B button
    y0.append(int(controller_0.button[melee.Button.BUTTON_B]))
    y1.append(int(controller_1.button[melee.Button.BUTTON_B]))
    # X || Y, since both X and Y can be used to jump.
    y0.append(int( controller_0.button[melee.Button.BUTTON_X] or controller_0.button[melee.Button.BUTTON_Y] ))
    y1.append(int( controller_1.button[melee.Button.BUTTON_X] or controller_1.button[melee.Button.BUTTON_Y] ))
    # L||R || L_SH||R_SH. Both of these can be used to shield, and it seems like we can probably ignore the continuous
    # nature of the shield button since this rarely comes into play in an actual game. We'll reduce this to a 0/1
    # binary value.
    y0.append(int( controller_0.button[melee.Button.BUTTON_L] or controller_0.button[melee.Button.BUTTON_R] or
                  (controller_0.l_shoulder > 0) or (controller_0.r_shoulder > 0) ))
    y1.append(int( controller_1.button[melee.Button.BUTTON_L] or controller_1.button[melee.Button.BUTTON_R] or
                  (controller_1.l_shoulder > 0) or (controller_1.r_shoulder > 0) ))
    # Z button
    y0.append(int(controller_0.button[melee.Button.BUTTON_Z]))
    y1.append(int(controller_1.button[melee.Button.BUTTON_Z]))
    # Main Stick X value, reduced to ten possible states to simplify things.
    y0.append(round(controller_0.main_stick[0], 1))
    y1.append(round(controller_1.main_stick[0], 1))
    # Main Stick Y value, reduced to ten possible states to simplify things.
    y0.append(round(controller_0.main_stick[1], 1))
    y1.append(round(controller_1.main_stick[1], 1))
    # C stick X value
    y0.append(round(controller_0.c_stick[0], 1))
    y1.append(round(controller_1.c_stick[0], 1))
    # C stick Y value
    y0.append(round(controller_0.c_stick[1], 1))
    y1.append(round(controller_1.c_stick[1], 1))

    return x, y0, y1

input_path = "D:\\smashdataset\\smashdataset\\"#
output_path = "D:\\smashdataset\\parseddata\\"
# Unimportant for data, but we can't have identically named files. This will ensure unique filenames.
replay_num = 0

for replay in tqdm(os.listdir(input_path)):
    # Increment filenum
    replay_num += 1
    # There were some corrupt replay files in the set. When the program crashed here at replay 42016, this line let us pick up
    # right where we left off.
    if replay_num < 42017: 
        continue
    # Load the file in.
    console = melee.Console(path=input_path+replay, system="file", allow_old_version=True)
    console.connect()

    gamestate = console.step()
    # Ensure the game is valid to minimize crashes.
    if gamestate is None:
        continue
    # Ensure the data is valid as well. 1, 3, or 4 player games are meaningless if they're in there.
    if len(gamestate.players) != 2:
        continue

    # Since we now know the game data is good, this gets the controller port numbers to read controller state data from.
    # Players in real games could be connected to any controller port, after all.
    controller_ports = list(gamestate.players.keys())

    # We knew this was going to take a very long time to run through, so we simplified our methods to read data from only two characters.
    # Initially, we thought we might train a network on both of these characters, but when it became apparent just how many more games played
    # Fox than played Jigglypuff, we stuck with just the one.
    p1 = gamestate.players[controller_ports[0]].character
    p2 = gamestate.players[controller_ports[1]].character
    if p1 != melee.Character.FOX and p2 != melee.Character.FOX and p1 != melee.Character.JIGGLYPUFF and p2 != melee.Character.JIGGLYPUFF:
        continue

    # We have data to assign a filename with, so we do that. Files are named with the format "ID#-P1 character-P2 character-data set.npy"
    filename = f"{replay_num}-{gamestate.players[controller_ports[0]].character.value}-{gamestate.players[controller_ports[1]].character.value}-"
    # We thought maybe it would be worth training on only a single stage, but by the time we got to writing the network it became apparent that the more
    # data we had, the better things would probably turn out. This step ended up being redundant, then.
    stagename = gamestate.stage.value

    # Arrays in which to store each frame for a given replay.
    x_set = []
    y0_set = []
    y1_set = []
    
    while gamestate is not None:

        # Store the data for the current frame.
        x_data, y0_data, y1_data = store_data(gamestate, controller_ports)
        # Append to the game set.
        x_set.append(x_data)
        y0_set.append(y0_data)
        y1_set.append(y1_data)

        # Proceed to the next frame.
        gamestate = console.step()
    
    # Store the data as a numpy array, which will allow for easy reading later on.
    x_arr = numpy.array(x_set)
    y0_arr = numpy.array(y0_set)
    y1_arr = numpy.array(y1_set)
    
    # Save those arrays.
    numpy.save(f"{output_path}{stagename}\\{filename}x", x_arr)
    numpy.save(f"{output_path}{stagename}\\{filename}y0", y0_arr)
    numpy.save(f"{output_path}{stagename}\\{filename}y1", y1_arr)