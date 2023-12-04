# Some of the references used to figure this part out:
# https://towardsdatascience.com/how-to-train-an-ai-to-play-any-game-f1489f3bc5c
# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py
# https://youtu.be/uKnjGn8fF70?si=PiV9WYX9FGrgrSDP
#
# This is our custom Gymnasium environment, which defines the environment for the Melee task and how the training algorithm should handle
# cleanup tasks.


import gymnasium
from gymnasium import spaces
import melee
import numpy
import os
import itertools
from Stop import stop
import time

class CustomGame(gymnasium.Env):
    def __init__(self):
        # Defines the Melee game state. This is the minimum amount of observations that I feel it would be possible for an agent
        # to learn to play the game from. Inferences based on these factors should be able to fill in any gaps we've left.
        self.observation_space = spaces.Dict({
            # Not sure what max values for coordinates are so I arbitrarily chose 500. I've not seen anything higher/lower than this
            # in a few example games I've run, so this should be safe.
            #
            # Box space gives a continuous range of possible values, and the shape means it's two dimensional for both X and Y coords.
            "agent_coords": spaces.Box(low = -500, high = 500, shape=(2,), dtype=numpy.float32),
            "opponent_coords": spaces.Box(low = -500, high = 500, shape=(2,), dtype=numpy.float32),
            # 401 is the length of the action enum list. It's possible we could work with a smaller action space given a fixed
            # number of characters, but it would be very hard to map which actions are possible with which characters, so I'm
            # leaving this at 401.
            #
            # Discrete means that there are 401 possibilities rather than a continuous space like with the Box.
            "agent_action": spaces.Discrete(401),
            "opponent_action": spaces.Discrete(401),
            # Players can be facing in two different directions, either left or right.
            "agent_facing": spaces.Discrete(2),
            "opponent_facing": spaces.Discrete(2),
            # Percent is only an integer on the screen, it's handled as a floating point value behind the scenes, between 0-999,
            # hence the use of a Box space.
            "agent_percent": spaces.Box(low = 0, high = 999, shape=(1,), dtype=numpy.int32),
            "opponent_percent": spaces.Box(low = 0, high = 999, shape=(1,), dtype=numpy.int32),
            # The most frames an action can take is 250. Most won't take that long at all, but we have to cover that space just in case.
            # I'm not totally sure why this errors out when it's discrete and not continuous considering that it should be returning int
            # values, but I suppose it doesn't hurt to leave it like this.
            "agent_action_frame": spaces.Box(low=0, high=250, shape=(1,), dtype=numpy.int32),
            "opponent_action_frame": spaces.Box(low=0, high=250, shape=(1,), dtype=numpy.int32),
            # A character is either off or on stage.
            "agent_off_stage": spaces.Discrete(2)
        })

        # Each move consists of one stick position and one button press. Among stick positions and button presses, we also include a "None"
        # button press for no attack and a "0.5,0.5" stick position for centered, no movement.
        self.action_space = spaces.Discrete(85)

        # Hold LibMelee required values.
        self.console: melee.Console = None
        self.controller: melee.Controller = None
        self.opponent_controller: melee.Controller = None
        self.current_state: melee.GameState = None

        # Generate list of possible moves from which to select.
        buttons = [melee.Button.BUTTON_B, melee.Button.BUTTON_A, melee.Button.BUTTON_Z, melee.Button.BUTTON_L, None]
        stick_positions = [(0.5,0.5), (0,0.5), (0,0), (0.5,0), (1,0), (1,0.5), (1,1), (0.5,1), (0,1),
                                (0.25,0.5), (0.25,0.25), (0.5,0.25), (0.75,0.25), (0.75,0.5), (0.75,0.75), (0.5,0.75), (0.25,0.75)]
        
        # Each number in the action space will translate to one index in this array, which contains all of the possible movement combinations.
        self.possible_moves = list(itertools.product(buttons, stick_positions))
        
    def _get_obs(self, gamestate: melee.GameState):
        players = gamestate.players

        # Returns all the game state data in the form of the observation space.
        return {
            "agent_coords": numpy.array([players[1].x, players[1].y], dtype=numpy.float32),
            "opponent_coords": numpy.array([players[2].x, players[2].y], dtype=numpy.float32),
            "agent_action": players[1].action.value,
            "opponent_action": players[2].action.value,
            "agent_facing": players[1].facing,
            "opponent_facing": players[2].facing,
            "agent_percent": numpy.array([players[1].percent], dtype=numpy.int32),
            "opponent_percent": numpy.array([players[2].percent], dtype=numpy.int32),
            "agent_action_frame": numpy.array([players[1].action_frame], dtype=numpy.int32),
            "opponent_action_frame": numpy.array([players[2].action_frame], dtype=numpy.int32),
            "agent_off_stage": players[1].off_stage
        }
    
    def step(self, action):
        # Agent makes one input in the space at each step.
        self._execute_action(action)

        # Game advances one step to evaluate that action.
        gamestate = self.console.step()

        # If the game is not active, then set the Done value to True.
        if gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            done = True
        else:
            done = False

        reward = self._calculate_reward(gamestate)

        # We never got far enough into this approach to figure out what to do with this variable.
        info = {}
        
        # Update historical data.
        self.current_state = gamestate

        return self._get_obs(gamestate), reward, done, False, info
    
    def _execute_action(self, action):
        # Selects the action from the array of possible actions based on the value chosen by the bot, then presses those buttons on the
        # controller.
        selected_move = self.possible_moves[action]
        self.controller.simple_press(selected_move[1][0], selected_move[1][1], selected_move[0])
    
    def _calculate_reward(self, gamestate: melee.GameState):
        KNOCKOUT_REWARD = 1000
        OFF_STAGE_PENALTY = 10
        
        # Reward if action has caused a KO to the opponent
        reward = KNOCKOUT_REWARD * (self.current_state.players[2].stock - gamestate.players[2].stock)
        # Penalize if action has caused a KO to the agent
        reward -= KNOCKOUT_REWARD * (self.current_state.players[1].stock - gamestate.players[1].stock)

        # If someone just KO'd, then their previous state will have higher percent than their current state (0%).
        # If that's the case, don't reward or penalize based on that.
        if reward == 0:
            # Reward if action has caused damage to opponent
            reward += gamestate.players[2].percent - self.current_state.players[2].percent
            # Penalize if action has caused damage to agent
            reward -= gamestate.players[1].percent - self.current_state.players[1].percent

        # Penalize if agent is off the stage
        if gamestate.players[1].off_stage:
            reward -= OFF_STAGE_PENALTY

        return reward

    
    def reset(self, seed=None, options=None):
        # There was some weird stuff going on with the emulator struggling to load, which caused the new instance to fail to connect.
        # This sleep call function made it work, so that's why it's here.
        if self.console:
            # A rewrite of the melee.Console.stop() function, which has a bug in it that keeps it from working.
            stop(self.console)
            time.sleep(1)

        # This code is only going to work on windows, but it'll get the default install directory for Slippi's Dolphin instance and set
        # it as the path in our melee.Console object.
        homeDirectory = os.path.expanduser('~'+os.environ.get("USERNAME"))
        self.console = melee.Console(path=homeDirectory+"\\AppData\\Roaming\\Slippi Launcher\\netplay", slippi_address="127.0.0.1")

        # Agent controller config.
        self.controller = melee.Controller(console=self.console, port=1)
        # This controller will be used to select an in-game bot to train against.
        self.opponent_controller = melee.Controller(console=self.console, port=2)

        # Start the emulator and connect to it. Put the game in the same directory as this file for this to work. The /b flag may or may not
        # be helping with the failing to connect bug, but I don't want to try removing it to see if it's necessary or not so it's staying there.
        self.console.run("./ssb.iso", environment_vars={"/b": "true"})
        # Again, helps with the failing to connect bug.
        time.sleep(3)
        # Connects agent to the emulator.
        self.console.connect()

        # Connect virtual controllers to the emulator.
        self.controller.connect()
        self.opponent_controller.connect()

        while True:
            # Advance one frame.
            self.current_state = self.console.step()
            # We're now in-game, so finish our reset.
            if self.current_state.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                return (self._get_obs(self.current_state), {})
            else:
                # Navigate the menus before we start our game.
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