import argparse
import signal
import sys
import melee
import random
import os

def downB(controller):
    controller

# Assuming you're running this under Windows. Didn't want to have a CLI argument every time.
homeDirectory = os.path.expanduser('~'+os.environ.get("USERNAME"))
# Emulator config.
console = melee.Console(path=homeDirectory+"\\AppData\\Roaming\\Slippi Launcher\\netplay", slippi_address="127.0.0.1")
# Bot controller config, configure human controller within Dolphin under port 2.
controller = melee.Controller(console=console,
                              port=1)

# Start the emulator and connect to it.
console.run()
console.connect()

# Connect virtual controller.
controller.connect()

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
        melee.MenuHelper.menu_helper_simple(gamestate,
                                            controller,
                                            melee.Character.JIGGLYPUFF,
                                            melee.Stage.BATTLEFIELD,
                                            melee.gamestate.port_detector(gamestate, melee.Character.JIGGLYPUFF, 0),
                                            costume=0,
                                            autostart=True,
                                            swag=False)