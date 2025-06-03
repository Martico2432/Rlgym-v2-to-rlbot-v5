import torch
import torch.nn as nn
import numpy as np
from obs import DefaultObs
from act import LookupTableAction

class DummyDiscreteFF(nn.Module):
    def __init__(self, input_shape, n_actions, layer_sizes, device):
        super().__init__()
        self.device = device

        assert len(layer_sizes) != 0, "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        layers = [nn.Linear(input_shape, layer_sizes[0]), nn.ReLU()]
        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(layer_sizes[-1], n_actions))
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers).to(self.device)

        self.n_actions = n_actions

    def get_output(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.model(obs)

    def save_state_dict(self, filename):
        torch.save(self.state_dict(), filename)

# Parameters for the dummy model
input_shape = DefaultObs().get_obs_space(None)[1]  # Example input shape
n_actions = LookupTableAction().get_action_space(None)[1]    # Example number of actions
layer_sizes = [64, 64]  # Example layer sizes
device = 'cpu'   # Use 'cuda' if you have a GPU

# Create an instance of the dummy model
dummy_model = DummyDiscreteFF(input_shape, n_actions, layer_sizes, device)

# Save the state dictionary
dummy_model.save_state_dict('PPO_POLICY.pt')

print("State dictionary saved as PPO_POLICY.pt")
