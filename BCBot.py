# This file wires up the BC network's outputs to controller inputs, allowing it to actually try playing the game rather than predicting
# static input/output matches.

import tensorflow as tf
from tensorflow import keras
import melee
import signal
import sys
import os
from Stop import stop

# Tells the program where the emulator is located.
homeDirectory = os.path.expanduser('~'+os.environ.get("USERNAME"))
console = melee.Console(path=homeDirectory+"\\AppData\\Roaming\\Slippi Launcher\\netplay", slippi_address="127.0.0.1")
# Configures a controller for the agent.
controller = melee.Controller(console=console, port=1)
# This controller will be used to select an in-game CPU opponent to play against.
opponentController = melee.Controller(console=console, port=2)

# Load in the model.
bc_model = keras.models.load_model('saved_symmetric_melee_model')

# This allows us to CTRL+C to close the emulator and kill the script rather than having to do those things separately.
def close(sig, frame):
    stop(console)
    sys.exit(0)
signal.signal(signal.SIGINT, close)

# Start the emulator and connect to it. Put the game in the same directory as this file for this to work.
# Rename the iso to match. I don't know that the /b is necessary anymore, but I've left it there because I don't want
# to take it out and accidentally break something.
console.run("./ssb.iso", environment_vars={"/b": "true"})
console.connect()

# Connect virtual controllers.
controller.connect()
opponentController.connect()

# This function takes all of the gamestate data and converts it into the format of the data that was fed into our NN.
def generate_state(gamestate: melee.GameState):
    # We have to nest the actual inputs within another array because of a quirk in how the model training data was 
    # formatted. Oops.
    outer_list = []
    inner_list = []

    # Normalized stage.
    inner_list.append(gamestate.stage.value/26)
    # P1 XY coordinates.
    inner_list.append(gamestate.players[1].x/500)
    inner_list.append(gamestate.players[1].y/500)
    # P2 XY coordinates.
    inner_list.append(gamestate.players[2].x/500)
    inner_list.append(gamestate.players[2].y/500)
    # Percent
    inner_list.append(gamestate.players[1].percent/999)
    inner_list.append(gamestate.players[2].percent/999)
    # Action
    inner_list.append(gamestate.players[1].action.value/397)
    inner_list.append(gamestate.players[2].action.value/397)
    # Action Frame
    inner_list.append(gamestate.players[1].action_frame/250)
    inner_list.append(gamestate.players[2].action_frame/250)
    # Facing
    inner_list.append(int(gamestate.players[1].facing))
    inner_list.append(int(gamestate.players[2].facing))
    # Jumps
    inner_list.append(gamestate.players[1].jumps_left/2)
    inner_list.append(gamestate.players[2].jumps_left/2)
    # Inv
    inner_list.append(int(gamestate.players[1].invulnerable))
    inner_list.append(int(gamestate.players[2].invulnerable))
    # On Ground
    inner_list.append(int(gamestate.players[1].on_ground))
    inner_list.append(int(gamestate.players[2].on_ground))
    # Offstage
    inner_list.append(int(gamestate.players[1].off_stage))
    inner_list.append(int(gamestate.players[2].off_stage))
    # Shield
    inner_list.append(round(gamestate.players[1].shield_strength/60, 1))
    inner_list.append(round(gamestate.players[2].shield_strength/60, 1))
    
    outer_list.append(inner_list)
    return outer_list

# Since there's no way for the bot to convey holding or pressing buttons, we use a simplified input scheme where
# the button it is most likely to press gets pressed on a given frame.
def send_simple_input(input, controller: melee.Controller):
    # Find most likely to be pressed button
    score = 0
    button = -1
    it = 0
    for x in input[0]:
        # This means we've gotten into stick position territory. Stop looking.
        if it > 4:
            break
        if score < x:
            score = x
            button = it
        it += 1

    # Conver the number to an actual button.
    if button == 0:
        button = melee.Button.BUTTON_A
    elif button == 1:
        button = melee.Button.BUTTON_B
    elif button == 2:
        button = melee.Button.BUTTON_X
    elif button == 3:
        button = melee.Button.BUTTON_L
    elif button == 4:
        button = melee.Button.BUTTON_Z
    else:
        button = None

    # Sends to the emulator a controller position with the desired stick positions, rounded to the nearest tenth just like the
    # training data was, as well as a button to be pressed.
    controller.simple_press(round(input[0][5], 1), round(input[0][6], 1), button)
    
# This value helps us get around a bug with melee.MenuHelper selecting CPU players. You'll see it in action momentarily.
cpu_select_counter = 100
# This value helps us deal with the network timing out. You'll see it in action momentarily.
frame_counter = 0

# Game loop.
while True:
    # Increment our frame counter.
    frame_counter += 1
    # Get next frame.
    gamestate = console.step()
    
    # If the game is active, make a move.
    if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        input = None
        # Only send a move every other frame, since our model is too slow to manage an input every frame.
        if frame_counter % 2 == 0:
            input = bc_model.predict(generate_state(gamestate))
            send_simple_input(input, controller)
    
    # Select the right game mode.
    elif gamestate.menu_state == melee.Menu.MAIN_MENU:
        melee.MenuHelper.choose_versus_mode(gamestate, controller)
    elif gamestate.menu_state == melee.Menu.CHARACTER_SELECT and cpu_select_counter > 0:
        # If our bot has selected a character, then player 2 should select their character and set the CPU difficulty.
        if gamestate.players[1].coin_down:
            melee.MenuHelper.choose_character(melee.Character.FOX,
                                              gamestate,
                                              opponentController,
                                              3,
                                              2,
                                              False,
                                              False)
        # Our bot should select their character.
        else:
            melee.MenuHelper.choose_character(melee.Character.FOX,
                                              gamestate,
                                              controller,
                                              0,
                                              0,
                                              False,
                                              False)
        # When player 2 has selected CPU difficulty, it gets stuck. While it's in the position where it's stuck, we'll tick down
        # the cpu_select_counter.
        if gamestate.players[2].cursor_x < -15.9 and gamestate.players[2].cursor_x > -16:
            if gamestate.players[2].cursor_y < -2.2 and gamestate.players[2].cursor_y > -2.3:
                cpu_select_counter -= 1
    # If that cpu_select_counter has reached 0 and we're still on the character select menu, then the opponent is probably ready. Let's
    # start the game.
    elif gamestate.menu_state == melee.Menu.CHARACTER_SELECT:
        controller.simple_press(0.5, 0.5, melee.Button.BUTTON_START)
        # The cursor has to be above the bottom half of the screen for the start button to do anything, and it gets stuck in the bottom half
        # of the screen due to this bug with MenuHelper. This code will move it upward.
        if gamestate.players[2].cursor_y < 3:
            opponentController.tilt_analog(melee.Button.BUTTON_MAIN, 0, 1)
        else:
            opponentController.simple_press(0.5, 0.5, melee.Button.BUTTON_START)
    # Select the stage to do battle on.
    elif gamestate.menu_state == melee.Menu.STAGE_SELECT:
        melee.MenuHelper.choose_stage(melee.Stage.YOSHIS_STORY, gamestate, controller)
    

        