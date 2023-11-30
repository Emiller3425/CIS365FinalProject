import argparse
import signal
import sys
import melee
import random
import os
import time
import psutil
from Stop import stop


# Assuming you're running this under Windows. Didn't want to have a CLI argument every time.
homeDirectory = os.path.expanduser('~'+os.environ.get("USERNAME"))
# Emulator config.
console = melee.Console(path=homeDirectory+"\\AppData\\Roaming\\Slippi Launcher\\netplay", slippi_address="127.0.0.1")
# Bot controller config.
controller = melee.Controller(console=console,
                              port=1)
# Selects an in-game bot to play against. If you want to play the bot yourself, remove anything using this opponentController
# and instead manually configure player 2's controls in Dolphin.
opponentController = melee.Controller(console=console, port=2)

# Theoretically, you should just be able to call console.stop() to kill the Dolphin instance.
# This doesn't seem to work. The implementation below definitely does but I'm leaving it commented
# out because it's Windows-specific code.
def close(sig, frame):
    stop(console)
    sys.exit(0)

signal.signal(signal.SIGINT, close)

# Start the emulator and connect to it. Put the game in the same directory as this file for this to work.
# Rename the iso to match. If it's not a .nkit I don't think it really matters, just rename it anyways.
console.run("./ssb.nkit.iso", environment_vars={"/b": "true"})
console.connect()

# Connect virtual controller.
controller.connect()
opponentController.connect()

while True:
    # Get next frame.
    gamestate = console.step()
    if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        # Game is active, logic goes here.

        if gamestate.distance < 4:
            controller.press_button(melee.Button.BUTTON_B)
            controller.tilt_analog(melee.Button.BUTTON_MAIN, 0.5, 0)
        else:
            onleft = gamestate.player[1].x < gamestate.player[2].x
            controller.tilt_analog(melee.Button.BUTTON_MAIN, int(onleft), 0.5)
            controller.release_button(melee.Button.BUTTON_B)

            if gamestate.player[1].y < gamestate.player[2].y:
                controller.press_button(melee.Button.BUTTON_X)
            else:
                controller.release_button(melee.Button.BUTTON_X)
    
    else:
        # Navigate menus.
        melee.MenuHelper.choose_character(melee.Character.FOX,
                                          gamestate,
                                          opponentController,
                                          cpu_level=3,
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