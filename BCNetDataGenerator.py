# Goal is to create a program that can parse through a collection of .slp replay files and better organize
# data to be used to feed into a behavioral cloning network (created using TensorFlow, since that doesn't seem
# especially difficult to do and I have a lot of CUDA cores begging to be used here). We could probably read this
# data directly but I'm imagining this will make training significantly faster, and at the very least it will make
# it easier to write the NN because we'll have to do less processing there.
#
# Data set used comes from an outline of a similar project, which you can find here:
# https://bycn.github.io/2022/08/19/project-nabla.html
#
# Includes 94484 high-level games. We only got through ~79% of them, or ~74240 of them.
#
# Notes:
#    - Ensure everything is normalized to a numeric value between 0 and 1.
#    - Ensure the network only knows about X or Y, L or R. If either, then 1, else 0.
#         - Inconsistent between BUTTON_L/R and L/R_SHOULDER. Normalize all four to binary.
#    - Ignore D-Pad input. It'd be funny if it started taunting, but we'll get better
#      gameplay performance without it.
#    - Ignore start button input. Not useful, no sense in training it.
#    - Reduce stick precision to tenths place. Don't overtrain on precision here,
#      that's probably enough precision to execute all the required moves.
# For Y set:
#    - int(A)
#    - int(B)
#    - int(X || Y)
#    - int(BUTTON_L||BUTTON_R || L_SHOULDER>0 || R_SHOULDER>0)
#    - int(Z)
#    - Main Stick X, round to nearest tenth
#    - Main Stick Y, round to nearest tenth
#    - C X, round to nearest tenth
#    - C Y, round to nearest tenth
# For X set:
#    - Stage
#    - 2x, one for each character:
#        - Character playing as (Know index of this for both characters)
#        - Xpos (use position[0], not .x)
#        - Ypos (Not sure what the range is, let's say divide by 500, should be fine?)
#        - Percent (divide by 999)
#        - Action (divide by 397, nothing should ever be 65535 and if it is, oh well)
#        - Action frame (determined 250 is the most for this during the Gym solution, div by that)
#        - Facing (wrap in int)
#        - Jumps Left (divide by 2)
#        - Invulnerable (wrap in int)
#        - On Ground (wrap in int)
#        - Off Stage (wrap in int)
#        - Shield Strength/60, round to nearest tenth
# Storage:
#    - numpy will let us store bulk numeric data using np.save('file.npy', npyArrayName), but that's only
#      numeric/value data. Will need to encode other stuff in probably filenames.
#    - Store in output_path\\stage.value
#    - Name files as f"{countervalue}-{port1.character.value}-{port2.character.value}.
#    - Each array entry should contain three subarrays:
#         - X set data
#         - Y set data for port1
#         - Y set data for port2
#      in that order, such that we can recover it by reading filenames.

import os
import sys
import melee
import numpy
# Thank you, ChatGPT, for these suggestions:
from tqdm import tqdm

def store_data(gamestate: melee.GameState, controller_ports: list):
    """This function probably should've been in its own file, but we've already started, so... oops."""
    x = []
    y0 = []
    y1 = []

    # Generate X data
    # Stage
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
    # X || Y
    y0.append(int( controller_0.button[melee.Button.BUTTON_X] or controller_0.button[melee.Button.BUTTON_Y] ))
    y1.append(int( controller_1.button[melee.Button.BUTTON_X] or controller_1.button[melee.Button.BUTTON_Y] ))
    # L||R || L_SH||R_SH
    y0.append(int( controller_0.button[melee.Button.BUTTON_L] or controller_0.button[melee.Button.BUTTON_R] or
                  (controller_0.l_shoulder > 0) or (controller_0.r_shoulder > 0) ))
    y1.append(int( controller_1.button[melee.Button.BUTTON_L] or controller_1.button[melee.Button.BUTTON_R] or
                  (controller_1.l_shoulder > 0) or (controller_1.r_shoulder > 0) ))
    # Z button
    y0.append(int(controller_0.button[melee.Button.BUTTON_Z]))
    y1.append(int(controller_1.button[melee.Button.BUTTON_Z]))
    # Main Stick X
    y0.append(round(controller_0.main_stick[0], 1))
    y1.append(round(controller_1.main_stick[0], 1))
    # Main Stick Y
    y0.append(round(controller_0.main_stick[1], 1))
    y1.append(round(controller_1.main_stick[1], 1))
    # C X
    y0.append(round(controller_0.c_stick[0], 1))
    y1.append(round(controller_1.c_stick[0], 1))
    # C Y
    y0.append(round(controller_0.c_stick[1], 1))
    y1.append(round(controller_1.c_stick[1], 1))

    return x, y0, y1

input_path = "D:\\smashdataset\\smashdataset\\"#
output_path = "D:\\smashdataset\\parseddata\\"
# Unimportant for data, but we can't have identically named files.
replay_num = 0

for replay in tqdm(os.listdir(input_path)):
    # Increment filenum
    replay_num += 1
    if replay_num < 42017:
        continue
    # Load the file in.
    console = melee.Console(path=input_path+replay, system="file", allow_old_version=True)
    console.connect()

    gamestate = console.step()
    # Ensure the game is valid, I do *not* want this crashing on me.
    if gamestate is None:
        continue
    # Ensure the data is valid as well.
    if len(gamestate.players) != 2:
        continue

    # Game data is good, now figure out player port numbers to read Y state data from.
    controller_ports = list(gamestate.players.keys())

    # This is going to take a lot of time to run through, so we're only going to train two characters.
    p1 = gamestate.players[controller_ports[0]].character
    p2 = gamestate.players[controller_ports[1]].character
    if p1 != melee.Character.FOX and p2 != melee.Character.FOX and p1 != melee.Character.JIGGLYPUFF and p2 != melee.Character.JIGGLYPUFF:
        continue

    # We have data to assign a filename with now, so do that. When we get to actually saving these, we'll tack on an X/Y1/Y2
    filename = f"{replay_num}-{gamestate.players[controller_ports[0]].character.value}-{gamestate.players[controller_ports[1]].character.value}-"
    # For directory, the stagenum
    stagename = gamestate.stage.value

    # Arrays in which to store the data:
    x_set = []
    y0_set = []
    y1_set = []
    
    while gamestate is not None:

        # Store the data:
        x_data, y0_data, y1_data = store_data(gamestate, controller_ports)
        # Append to global set:
        x_set.append(x_data)
        y0_set.append(y0_data)
        y1_set.append(y1_data)

        # Increment the gamestate
        gamestate = console.step()
    
    # Save the data:
    x_arr = numpy.array(x_set)
    y0_arr = numpy.array(y0_set)
    y1_arr = numpy.array(y1_set)

    numpy.save(f"{output_path}{stagename}\\{filename}x", x_arr)
    numpy.save(f"{output_path}{stagename}\\{filename}y0", y0_arr)
    numpy.save(f"{output_path}{stagename}\\{filename}y1", y1_arr)