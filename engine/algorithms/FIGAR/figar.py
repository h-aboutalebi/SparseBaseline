import numpy as np
import torch
import torch.nn as nn
from engine.algorithms.DDPG.ddpg import Critic

from torch.autograd import Variable
import torch.nn.functional as F

from engine.algorithms.DDPG.ddpg import DDPG
from engine.algorithms.abstract_agent import AbstractAgent


class Actor_repeat(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor_repeat, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.soft=nn.Softmax()

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        x=self.soft(x)
        return torch.argmax(x,dim=1)+1

class Figar(DDPG):

    def __init__(self, state_dim, action_dim, max_action, expl_noise, action_high, action_low, tau,device,lr_actor, W=10):
        super(Figar, self).__init__(state_dim=state_dim, action_dim=action_dim,
                                    max_action=max_action,device=device,action_high=action_high,
                                    action_low=action_low,expl_noise=expl_noise,lr_actor=lr_actor,
                                    tau=tau)
        action_dim=1+action_dim
        self.actor_repeat = Actor_repeat(state_dim, W).to(self.device)
        self.actor_repeat_target = Actor_repeat(state_dim, W).to(self.device)
        self.actor_repeat_target.load_state_dict(self.actor_repeat.state_dict())
        self.actor_repeat_optimizer = torch.optim.Adam(self.actor_repeat.parameters(), lr=lr_actor)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)


    def select_action(self, state, tensor_board_writer=None, previous_action=None, step_number=None):
        state = np.array(state)
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        action1 = self.actor(state).cpu().data.numpy().flatten()
        if self.expl_noise != 0:
            action1 = (action1 + np.random.normal(0, self.expl_noise, size=self.action_dim)).clip(
                self.action_low, self.action_high)
        action2 = self.actor_repeat(state).cpu().data.numpy().item()
        action=(action1,action2)
        return action

    def select_action_target(self, state,  tensor_board_writer=None, step_number=None):
        # print(state)
        state = np.array(state)
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        action1= self.actor_target(state).cpu().data.numpy().flatten()
        action2= self.actor_repeat(state).cpu().data.numpy().item()
        return (action1,action2)

    def train(self, replay_buffer, step_number, batch_size=64, gamma=0.99, writer=None, **kwargs):

        # Sample replay buffer
        x, y, u, r, d = replay_buffer.sample(batch_size)
        state = torch.Tensor(x).to(self.device)
        action1 = torch.Tensor(np.array([list(a[0]) for a in u])).to(self.device)
        action2 = torch.Tensor(np.array([a[1] for a in u])).to(self.device)
        next_state = torch.Tensor(y).to(self.device)
        done = torch.Tensor(1 - d).to(self.device)
        reward = torch.Tensor(r).to(self.device)

        next_act1=self.actor_target(next_state)
        nextact2=self.actor_repeat_target(next_state)
        concat_act=torch.cat((next_act1,torch.unsqueeze(nextact2, 1)),dim=1)
        # Compute the target Q value
        target_Q = self.critic_target(next_state, concat_act)
        target_Q = reward + (done * gamma * target_Q).detach()

        # Get current Q estimate
        concat_act2 = torch.cat((action1, torch.unsqueeze(action2, 1)), dim=1)
        current_Q = self.critic(state, concat_act2)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        act1 = self.actor_target(state)
        act2 = self.actor_repeat_target(state)
        concat_act3=torch.cat((act1,torch.unsqueeze(act2, 1)),dim=1)
        actor_loss = -self.critic(state, concat_act3).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        self.actor_repeat_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_repeat_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor_repeat.parameters(), self.actor_repeat_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))



