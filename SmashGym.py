# Trying to think through a potential deep learning solution, using some references:
# https://towardsdatascience.com/how-to-train-an-ai-to-play-any-game-f1489f3bc5c
# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py
# https://web.stanford.edu/class/aa228/reports/2018/final112.pdf
#
# Very possible we don't have time for this, and I'm struggling to work my way through all this. Another alternative is an
# observation-based solution. I have 200+gb of Slippi replays downloaded if someone can find a good way to make that work. For
# some references, this video has stuff in the description I was trying to build off of but I got a bit lost there as well, mostly
# because I don't know the game well enough:
# https://www.youtube.com/watch?v=XmNQOSGcrUE
# Related but not from the same guy, where I got the replays from: https://bycn.github.io/2022/08/19/project-nabla-writeup.html
#
# If you get to a point where you're ready to train this thing, just let me know and I can do it. If you use PyTorch or some other
# library that can be GPU accelerated, I have a *lot* of compute I can throw at this. We'll be limited by the real-time speed limits
# of emulation though, of course.
#
# Worst case scenario we can hand-code a decision tree, but there has to be some sort of "training" per our proposal. I'd rather it not
# come to that though if we can help it.

import gymnasium
import melee
from gymnasium import spaces
import numpy
import os

class custom_game(gymnasium.Env):
    def __init__(self, console: melee.Console):
        self.observation_space = spaces.Dict(
            # Not sure what max values for coordinates are so I arbitrarily chose 500.
            "agent_coords": spaces.Box(low = -500, high = 500, shape=(2,), dtype=numpy.float32),
            "opponent_coords": spaces.Box(low = -500, high = 500, shape=(2,), dtype=numpy.float32),
            # This is the length of the action enum list. Possible we could work with a smaller space for any given character.
            "agent_action": spaces.Discrete(401),
            "opponent_action": spaces.Discrete(401),
            "agent_facing": spaces.Discrete(2),
            "opponent_facing": spaces.Discrete(2),
            "agent_percent": spaces.Box(low = 0, high = 999, shape=(1,), dtype=numpy.int32),
            "opponent_percent": spaces.Box(low = 0, high = 999, shape=(1,), dtype=numpy.int32),
            "agent_action_frame": spaces.Box(low=0, high=250, shape=(1,), dtype=numpy.int32),
            "opponent_action_frame": spaces.Box(low=0, high=250, shape=(1,), dtype=numpy.int32)
        )

        # If this proves inefficient, do 9 stick positions.
        self.action_space = spaces.Dict(
            # Center position, as well as any full tilt in the eight main directions or half tilt in the eight main directions.
            "stick_position": spaces.Discrete(17),
            # Only buttons that matter are A, B, Z, one of L/R, and no button pressed at all.
            "buttons": spaces.Discrete(5)
            # Every move can be represented by a combination of one of these stick positions and one of these button presses.
        )

        self.console = console

        self.init_run = 1

    def _get_obs(self, gamestate):
        players = gamestate.players

        return {
            "agent_coords": numpy.array([players[1].x, players[1].y], dtype=numpy.float32),
            "opponent_coords": numpy.array([players[2].x, players[2].y], dtype=numpy.float32),
            "agent_action": players[1].action,
            "opponent_action": players[2].action,
            "agent_facing": players[1].facing,
            "opponent_facing": players[2].facing,
            "agent_percent": numpy.array([players[1].percent], dtype=numpy.int32),
            "opponent_percent": numpy.array([players[2].percent], dtype=numpy.int32),
            "agent_action_frame": numpy.array([players[1].action_frame], dtype=numpy.int32),
            "opponent_action_frame": numpy.array([players[2].action_frame], dtype=numpy.int32)
        }
    
    def step(self, action):
        self.console.step()
        # Figure this out later.
        return self._get_obs(), reward, done, info
    
    def reset(self):
        # Windows-only code below, be warned. Since console.stop doesn't seem to work reliably.
        if self.init_run == 1:
            self.init_run = 0
        else:
            os.system("taskkill /im \"Slippi Dolphin.exe\"")
            os.system("taskkill /im \"Slippi Dolphin.exe\"")

        # More Windows-only code, swap out filepaths where needed.
        homeDirectory = os.path.expanduser('~'+os.environ.get("USERNAME"))
        console = melee.Console(path=homeDirectory+"\\AppData\\Roaming\\Slippi Launcher\\netplay", slippi_address="127.0.0.1")

        # Bot controller config, configure human controller within Dolphin under port 2.
        controller = melee.Controller(console=console, port=1)
        # Selects an in-game bot to play against.
        opponentController = melee.Controller(console=console, port=2)

        # Start the emulator and connect to it. Put the game in the same directory as this file for this to work.
        console.run("./ssb.nkit.iso")
        console.connect()
        self.console = console

        # Connect virtual controller.
        controller.connect()
        opponentController.connect()

        while True:
            gamestate = self.console.step()
            if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                return self._get_obs(gamestate)
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

        