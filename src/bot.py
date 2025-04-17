import os
import numpy as np
import torch
from rlbot.flat import BallAnchor, ControllerState, GamePacket, MatchPhase
from rlbot.managers import Bot
from rlgym_compat import GameState, common_values
from collections import OrderedDict
import colorama


# Your imports here
from act import MinimalistLookupTableAction
from obs import MinimalistRelativeDefaultObs
from custom_discrete import DiscreteFF

from rlgym_compat.sim_extra_info import SimExtraInfo


model_path = 'PPO_POLICY.pt'


def model_info_from_dict(loaded_dict):
    state_dict = OrderedDict(loaded_dict)

    bias_counts = []
    weight_counts = []
    for key, value in state_dict.items():
        if ".weight" in key:
            weight_counts.append(value.numel())
        if ".bias" in key:
            bias_counts.append(value.size(0))

    inputs = int(weight_counts[0] / bias_counts[0])
    outputs = bias_counts[-1]
    layer_sizes = bias_counts[:-1]

    return inputs, outputs, layer_sizes






class MyBot(Bot):

    def initialize(self):
        self.deterministic = False #NOTE: Set to True if you want to use deterministic actions, default is False
        self.ticks = self.tick_skip = 8 #NOTE: Set the amount of ticks to skip, default is 8
        self.device = torch.device("cpu") #NOTE: Set the device to use, default is cpu

        # Get the bot data from model, so no need to modfify anything here
        # Print a waring in Yellow that says to ignore the following warning
        print(colorama.Fore.YELLOW + "WARNING: The following warning is expected and can be ignored" + colorama.Fore.RESET)
        model_file = torch.load(model_path, map_location=self.device)
        input_amount, action_amount, layer_sizes = model_info_from_dict(model_file)

        # Make the policy
        self.policy = DiscreteFF(input_amount, action_amount, layer_sizes, self.device)
        self.policy.load_state_dict(model_file)
        torch.set_num_threads(1)

        #! update this with your action and obs
        action_parser = MinimalistLookupTableAction()
        Obs = MinimalistRelativeDefaultObs(
                             pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
                             ang_coef=1 / np.pi,
                             lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                             ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
                             boost_coef=1 / 100.0)
        

        # Some variables that are going to be used
        self.prev_time = 0.0
        self.prev_control = ControllerState()
        self.controls = ControllerState()
        self.obs = Obs
        self.action_parser = action_parser
        self.sent_more_than_one_ball_warning = False

        
        self.extra_info = SimExtraInfo(self.field_info, tick_skip=self.tick_skip)

    def get_output(self, packet: GamePacket) -> ControllerState:
        """
        This function will be called by the framework many times per second. This is where the bot makes any decisions.
        And outputs them.
        """


        # Calculate the time elapsed since the last frame
        cur_time = packet.match_info.frame_num
        ticks_elapsed = cur_time - self.prev_time
        self.prev_time = cur_time

        self.ticks += ticks_elapsed

        # Some checks
        if len(packet.balls) == 0 or packet.match_info.match_phase == MatchPhase.Ended:
            # If there are no balls current in the game (likely due to being in a replay) or game alredy ended, random movements.
            # Just use 5 random actions -1 to 1 and after that 3 random 0 or 1
            # Do a celebration :D
            return ControllerState(throttle=np.random.uniform(-1, 1), steer=np.random.uniform(-1, 1), pitch=np.random.uniform(-1, 1), yaw=np.random.uniform(-1, 1), roll=np.random.uniform(-1, 1), jump=np.random.choice([True, False]), boost=np.random.choice([True, False]), handbrake=np.random.choice([True, False]))
        if len(packet.balls) > 1 and self.sent_more_than_one_ball_warning == False:
            print(colorama.Fore.RED + "WARNING: More than one ball detected. This is unexpected and may cause issues.")
            self.sent_more_than_one_ball_warning = True

        # Get the model and use it to predict the next action using the obs and action parser
        
        if self.ticks >= self.tick_skip - 1:
            self.ticks = 0

            # Get the current game state
            self.game_state = self.game_state.create_compat_game_state(self.field_info, tick_skip=self.tick_skip)
            extra_info = self.extra_info.get_extra_info(packet)
            self.game_state.update(packet, extra_info=extra_info)
            # Get the car ids
            cars_ids = self.game_state.cars.keys()
			
            # Build the obs with the ids
            obs = self.obs.build_obs(cars_ids, self.game_state, {"sus": "sus"}) # IF A JUDGE SEES THIS, I AM SORRY. SUS

            # Get the obs of the current car
            obs = obs.get(self.spawn_id)
            obs = np.asarray(obs).flatten()
            obs_tensor = torch.tensor(np.array(obs, dtype=np.float32), dtype=torch.float32, device=self.device)


            # Get the prediction
            with torch.no_grad():
                action_idx, probs = self.policy.get_action(obs_tensor, deterministic=self.deterministic)
                #print(f"Action idx: {action_idx}")

                if self.deterministic == True:
                    # If it's true, we should place it inside a tensor
                    action_idx = torch.tensor([action_idx], device=self.device)

            # Based on the action, parse it into an array of 8 values
            parsed_actions = self.action_parser.parse_actions(actions={self.spawn_id: action_idx}, state=self.game_state, shared_info={"sus": "sus"}).get(self.spawn_id)
            

            # Check for errors
            if len(parsed_actions.shape) == 2:
                if parsed_actions.shape[0] == 1:
                    parsed_actions = parsed_actions[0]
		
            if len(parsed_actions.shape) != 1:
                raise Exception("Invalid action:", parsed_actions)

        else:
            # Still tick skip is not reached, return the previous control
            return self.prev_control

        # Update the controls, but if not able to, just return the previous control
        try:
            self.update_controls(parsed_actions)
        except:
            return self.prev_control
        
        # Save the previous control
        self.prev_control = self.controls

        return self.controls
    

    def update_controls(self, action):
        """
        Based on the action, update the controls
        """
        actions = []
        for actio in action:
            actions.append(float(actio))
        action = actions
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0


if __name__ == "__main__":
    # Connect to RLBot and run
    # Having the agent id here allows for easier development,
    # as otherwise the RLBOT_AGENT_ID environment variable must be set.
    MyBot("rlbot_community/python_example").run()
