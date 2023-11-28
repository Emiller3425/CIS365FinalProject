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
from gymnasium import spaces
import melee
import numpy
import os

class custom_game(gymnasium.Env):
    def __init__(self):
        self.observation_space = spaces.Dict({
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
            "opponent_action_frame": spaces.Box(low=0, high=250, shape=(1,), dtype=numpy.int32),
            "agent_off_stage": spaces.Discrete(2)
        })

        # If this proves inefficient, do 9 stick positions.
        self.action_space = spaces.Dict({
            # Can make fewer positions if desired.
            "stick_x": spaces.Discrete(5),
            "stick_y": spaces.Discrete(5), 
            # Only buttons that matter are A, B, Z, one of L/R, and no button pressed at all.
            "buttons": spaces.Discrete(5)
            # Every move can be represented by a combination of one of these stick positions and one of these button presses.
        })

        self.console: melee.Console = None
        self.controller: melee.Controller = None
        self.opponent_controller: melee.Controller = None
        self.current_state: melee.GameState = None
        self.init_run = 1
        self.buttons = [melee.Button.BUTTON_B, melee.Button.BUTTON_A, melee.Button.BUTTON_Z, melee.Button.BUTTON_L, None]
        self.stick_positions = [0, 0.25, 0.5, 0.75, 1]

    def _get_obs(self, gamestate: melee.GameState):
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
            "opponent_action_frame": numpy.array([players[2].action_frame], dtype=numpy.int32),
            "agent_off_stage": players[1].off_stage
        }
    
    def step(self, action):
        self.controller.simple_press(self.stick_positions[action["stick_x"]],
                                     self.stick_positions[action["stick_y"]],
                                     self.buttons[action["buttons"]])
        gamestate = self.console.step()

        if gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            done = True
        else:
            done = False

        reward = self._calculate_reward(gamestate)

        # Not sure what to do with this yet.
        info = {}
        
        # Update historical data.
        self.current_state = gamestate
        return self._get_obs(), reward, done, info
    
    def _calculate_reward(self, gamestate: melee.GameState):
        KNOCKOUT_REWARD = 1000
        OFF_STAGE_PENALTY = 10
        
        # Reward if action has caused KO to opponent
        reward = KNOCKOUT_REWARD * (self.current_state.players[2].stock - gamestate.players[2].stock)
        # Penalize if action has caused damage to us
        reward -= KNOCKOUT_REWARD * (self.current_state.players[1].stock - gamestate.players[1].stock)

        # If we just KO'd, then the previous state will have higher percent than the current state (0%). Don't mess with that.
        if reward == 0:
            # Reward if action has caused damage to opponent
            reward += gamestate.players[2].percent - self.current_state.players[2].percent
            # Penalize if action has caused damage to us
            reward -= gamestate.players[1].percent - self.current_state.players[1].percent

        # Penalize if off the stage
        if gamestate.players[1].off_stage:
            reward -= OFF_STAGE_PENALTY

    
    def reset(self):
        # Windows-only code below, be warned. Since console.stop doesn't seem to work reliably. Don't want to
        # try killing the process on first go because it won't be active, but we do want this here to ensure we
        # don't run out of resources over a long training session.
        if self.init_run == 1:
            self.init_run = 0
        else:
            os.system("taskkill /im \"Slippi Dolphin.exe\"")
            os.system("taskkill /im \"Slippi Dolphin.exe\"")

        # More Windows-only code, swap out filepaths where needed.
        homeDirectory = os.path.expanduser('~'+os.environ.get("USERNAME"))
        self.console = melee.Console(path=homeDirectory+"\\AppData\\Roaming\\Slippi Launcher\\netplay", slippi_address="127.0.0.1")

        # Bot controller config, configure human controller within Dolphin under port 2.
        self.controller = melee.Controller(console=self.console, port=1)
        # Selects an in-game bot to play against.
        self.opponent_controller = melee.Controller(console=self.console, port=2)

        # Start the emulator and connect to it. Put the game in the same directory as this file for this to work.
        self.console.run("./ssb.nkit.iso")
        self.console.connect()

        # Connect virtual controller.
        self.controller.connect()
        self.opponent_controller.connect()

        while True:
            self.current_state = self.console.step()
            if self.current_state.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                return self._get_obs(self.current_state)
            else:
                # Navigate menus.
                melee.MenuHelper.choose_character(melee.Character.FOX,
                    self.current_state,
                    self.opponent_controller,
                    cpu_level=9,
                    costume=0,
                    swag=False)
                melee.MenuHelper.menu_helper_simple(self.current_state,
                    self.controller,
                    melee.Character.JIGGLYPUFF,
                    melee.Stage.BATTLEFIELD,
                    melee.gamestate.port_detector(self.current_state, melee.Character.JIGGLYPUFF, 0),
                    costume=0,
                    autostart=True,
                    swag=False)