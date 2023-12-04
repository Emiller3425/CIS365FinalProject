import argparse
import signal
import sys
import melee
import random
import os
import time
import psutil
from pathlib import Path
from Stop import stop


# Hard coded path to Slippi Dolphin
home_directory = "/Users/emiller3425"
slippi_path = "Library/Application Support/Slippi Launcher/netplay"
full_path = Path(os.path.expanduser(os.path.join(home_directory, slippi_path)))

# Emulator config.
console = melee.Console(path=str(full_path.resolve()), slippi_address="127.0.0.1")
# Bot controller config.
controller = melee.Controller(console=console, port=1)
# set opponent controller
opponentController = melee.Controller(console=console, port=2)

# Kill dolphin instance on windows
def close(sig, frame):
    stop(console)
    sys.exit(0)

signal.signal(signal.SIGINT, close)

# Start the emulator and connect to it.
console.run(iso_path="/Users/emiller3425/cis365/CIS365FinalProject/ssb.iso", exe_name="Slippi Dolphin")
console.connect()

# Connect virtual controller.
controller.connect()
opponentController.connect()

while True:
    # Get next frame.
    gamestate = console.step()
    if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        # Game is active

        #If agent is off stage
        if gamestate.player[1].off_stage == 1:
            controller.press_button(melee.Button.BUTTON_X)
            controller.release_button(melee.Button.BUTTON_X)
            controller.press_button(melee.Button.BUTTON_X)
            controller.release_button(melee.Button.BUTTON_X)
            gamestate = console.step()
        if gamestate.player[2].off_stage == 1:
            controller.release_all()
        #If opponant is off stage
        elif gamestate.distance < 6:
            if gamestate.player[1].y < gamestate.player[2].y:
                controller.press_button(melee.Button.BUTTON_X)
                controller.release_button(melee.Button.BUTTON_X)
            attackType = random.randint(0, 15)
            #random chance for heavy attack
            if attackType > 10:
                controller.press_button(melee.Button.BUTTON_B)
                controller.tilt_analog(melee.Button.BUTTON_MAIN, 0.5, 0)
            #random chance for light attack
            elif attackType > 5:
                controller.press_button(melee.Button.BUTTON_A)
                controller.tilt_analog(melee.Button.BUTTON_MAIN, 0.5, 0)
            #random chance for a light jump attack
            else:
                controller.tilt_analog(melee.Button.BUTTON_MAIN, int(onleft), 1)
                controller.press_button(melee.Button.BUTTON_X)
                controller.release_button(melee.Button.BUTTON_X)
                controller.press_button(melee.Button.BUTTON_X)
                controller.release_button(melee.Button.BUTTON_X)
                controller.press_button(melee.Button.BUTTON_A)
                controller.tilt_analog(melee.Button.BUTTON_MAIN, 0.5, 0)
        else:
            # Move towards opponent if outside attack range
            onleft = gamestate.player[1].x < gamestate.player[2].x
            controller.press_button(melee.Button.BUTTON_A)
            controller.tilt_analog(melee.Button.BUTTON_MAIN, int(onleft), 0.5)
            controller.release_button(melee.Button.BUTTON_A)
            #If below oppanant jump above them
            if gamestate.player[1].y < gamestate.player[2].y:
                controller.release_all()
                controller.tilt_analog(melee.Button.BUTTON_MAIN, int(onleft), 1)
                controller.press_button(melee.Button.BUTTON_X)
                controller.release_button(melee.Button.BUTTON_X)
                controller.press_button(melee.Button.BUTTON_X)
                controller.release_button(melee.Button.BUTTON_X)
                controller.press_button(melee.Button.BUTTON_X)
                controller.release_button(melee.Button.BUTTON_X)
            else:
                controller.release_button(melee.Button.BUTTON_X)

    else:
        # Navigate menus , select character and map.
        melee.MenuHelper.choose_character(melee.Character.FOX,
                                          gamestate,
                                          opponentController,
                                          cpu_level=9,
                                          costume=0,
                                          swag=False)
        melee.MenuHelper.menu_helper_simple(gamestate,
                                            controller,
                                            melee.Character.JIGGLYPUFF,
                                            melee.Stage.BATTLEFIELD,
                                            melee.gamestate.port_detector(gamestate, melee.Character.JIGGLYPUFF, 0),
                                            costume=0,
                                            autostart=True,
                                            swag=False)