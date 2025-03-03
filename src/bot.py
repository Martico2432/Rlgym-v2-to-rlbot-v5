from rlbot.flat import BallAnchor, ControllerState, GamePacket
from rlbot.managers import Bot

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import ControlStep, Sequence
from util.vec import Vec3
import torch

from custom_discrete import DiscreteFF
from collections import OrderedDict
from act import MinimalistLookupTableAction
from obs import MinimalistRelativeDefaultObs
from rlgym_compat import common_values
import numpy as np

from rlgym_compat import GameState

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

def load_model(model_path):
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    input_amount, action_amount, layer_sizes = model_info_from_dict(state_dict)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    policy = DiscreteFF(input_amount, action_amount, layer_sizes, device)

    new_state_dict = {}
    for key, value in state_dict.items():
        if key in policy.state_dict() and policy.state_dict()[key].shape == value.shape:
            new_state_dict[key] = value
        else:
            print(f"Skipping {key} due to shape mismatch: {value.shape} vs {policy.state_dict()[key].shape}")

    policy.load_state_dict(new_state_dict, strict=False)
    policy.eval()
    return policy, device, input_amount


model_path = 'PPO_POLICY.pt'







class MyBot(Bot):
    active_sequence: Sequence | None = None

    def initialize(self):
        # Set up information about the boost pads now that the game is active and the info is available
        best_model, device, input_amount = load_model(model_path)

        self.model = best_model
        self.input_amount = input_amount
        self.device = device


        action_parser = MinimalistLookupTableAction()
        Obs = MinimalistRelativeDefaultObs(#zero_padding=3, #!!!!! WARNING IMPorTANT change back to 3 after 24k bot and use my advanced
                             pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
                             ang_coef=1 / np.pi,
                             lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                             ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
                             boost_coef=1 / 100.0)
        

        self.game_state = GameState()

        self.prev_time = 0.0
        self.ticks = tick_skip = 8
        self.tick_skip = tick_skip
        self.update_action = True
        self.prev_control = ControllerState()
        self.controls = ControllerState()
        self.obs = Obs
        self.action_parser = action_parser

    def get_output(self, packet: GamePacket) -> ControllerState:
        """
        This function will be called by the framework many times per second. This is where you can
        see the motion of the ball, etc. and return controls to drive your car.
        """

        cur_time = packet.match_info.frame_num
        ticks_elapsed = cur_time - self.prev_time
        self.prev_time = cur_time

        self.ticks += ticks_elapsed


        # Keep our boost pad info updated with which pads are currently active
        if len(packet.balls) == 0:
            # If there are no balls current in the game (likely due to being in a replay), skip this tick.
            return ControllerState()
        # we can now assume there's at least one ball in the match

        # Get the model and use it to predict the next action using 
        
        if self.ticks >= self.tick_skip - 1:
            self.ticks = 0

            self.game_state = self.game_state.create_compat_game_state(self.field_info)
            self.game_state.update(packet)
            self.update_action = False

            player_car = self.game_state.cars.get(self.spawn_id)
            cars_ids = self.game_state.cars.keys()
            #print(f"Cars: {cars_ids}")
            #print(f"Spawn id:{self.spawn_id}")
			
            obs = self.obs.build_obs(cars_ids, self.game_state, {"sus": "sus"}) # IF A JUDGE SEES THIS, I AM SORRY. SUS
            #print(f"Obs: {obs}")
            #print(f"Obs keys: {obs.keys()}")
            obs = obs[self.spawn_id]
            #print(f"Obs user: {obs}")
            obs = np.asarray(obs).flatten()
            #print(f"Obs flattened: {obs}")
            obs_tensor = torch.tensor(np.array(obs, dtype=np.float32), dtype=torch.float32, device=self.device)
            #print(len(obs_tensor))
            model_actions = self.model.get_output(obs_tensor).detach().cpu().numpy()
            # Set model acrions to shape 1
            model_actions = np.argmax(model_actions).reshape(1)


            parsed_actions = self.action_parser.parse_actions(actions={self.spawn_id: model_actions}, state=self.game_state, shared_info={}).get(self.spawn_id)[0]

            # Pass the observation through the model to get the action
            #//print(f"Model action: {parsed_actions}")

        #print(self.ticks)

        try:
            self.update_controls(parsed_actions)
        except UnboundLocalError:
            return self.prev_control




        self.prev_control = self.controls

        return self.controls
    

    def update_controls(self, action):
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
