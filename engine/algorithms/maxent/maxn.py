import torch
import numpy as np

from engine.algorithms.DDPG_POLYRL.poly_rl import PolyRL
from engine.algorithms.SAC.sac import SAC


class Maxnet(SAC):

    def __init__(self, state_dim, action_dim, max_action, action_space, gamma, tau, alpha, device,
                 update_interval=1, env=None,nb_observations =None,
                 policy="Gaussian", automatic_entropy_tuning=False, hidden_size=256, lr=0.0003, start_steps=10000):
        super().__init__(5, action_dim, max_action, action_space, gamma, tau, alpha, device,
                         update_interval, policy, automatic_entropy_tuning, hidden_size, lr, start_steps)
        self.env = env
        self.nb_observations=nb_observations
        self.nb_environment_reset = 0
        self.previous_state = None
        self.buffer_Maxnet = []
        self.features = [2,7,8,9,10]
        self.normalization_factors=[]
        self.flag_finished = False
        self.min_bin = -1
        self.max_bin = 1
        self.num_bins = 15
        self.height_bins = 20
        self.state_bins=self.get_state_bins()


    def mod_select_action(self, state, tensor_board_writer, previous_action, step_number, nb_environment_reset):
        self.counter_actions += 1
        if (self.start_steps < self.counter_actions):
            if (self.flag_finished is False):
                self.flag_finished = True
            state = torch.Tensor(state).reshape(1, -1)
            action, _, _ = self.policy.sample(state.to(self.device))
            return action.detach().cpu().numpy()[0]
        else:
            action = self.env.action_space.sample()
            self.buffer_Maxnet.append(state)
            return action

    def get_normalization(self):
        for i in range(self.nb_observations):
            i_vals = [x[i] for x in self.buffer_Maxnet]
            max_i_val = max(i_vals)
            self.normalization_factors.append(max_i_val)
        print(self.normalization_factors)
        print(len(self.normalization_factors))

    def get_state_bins(self):
        state_bins = [
            # height
            self.discretize_range(0.2, 1.0, self.height_bins),
            # other fields
            self.discretize_range(self.min_bin, self.max_bin, self.num_bins),
            self.discretize_range(self.min_bin, self.max_bin, self.num_bins),
            self.discretize_range(self.min_bin, self.max_bin, self.num_bins),
            self.discretize_range(self.min_bin, self.max_bin, self.num_bins)
        ]
        return state_bins

    def discretize_range(self,lower_bound, upper_bound, num_bins):
        return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

    def discretize_state_normal(self,observation):
        state = []
        for i, idx in enumerate(self.features):
            state.append(self.discretize_value(observation[idx], self.state_bins[i]))
        return state

    def discretize_value(self,value, bins):
        return np.asscalar(np.digitize(x=value, bins=bins))
