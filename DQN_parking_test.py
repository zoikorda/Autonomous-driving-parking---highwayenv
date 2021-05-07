import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import numpy as np
#import wandb
from collections import deque
from random import sample, random
from torch.autograd import Variable

from dataclasses import dataclass
from typing import Any
from random import sample

import gym
import time

class Model(nn.Module):

  def __init__(self, observation_shape, num_actions):
        super(Model, self).__init__()
        assert len(observation_shape) == 1
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.net = torch.nn.Sequential(
        torch.nn.Linear(observation_shape[0], 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, num_actions)
        )

        self.optimizer = optimizer.Adam(self.net.parameters(), lr=0.0001)

  def forward(self, x):
    return self.net(x)

@dataclass
class Sarsd:
  state: Any
  action: int
  reward: float
  next_state: Any
  done: bool

class ReplayBuffer:
  def __init__(self, buffer_size=100000):
    self.buffer_size = buffer_size
    #self.buffer = []
    self.buffer = deque(maxlen=buffer_size)

  def insert(self, sars):
    self.buffer.append(sars)
    #self.buffer = self.buffer[-self.buffer_size:]

  def sample(self, num_samples):
    assert num_samples <= len(self.buffer)
    return sample(self.buffer, num_samples)

gamma = 0.99

def update_target_model(model, target):
  target.load_state_dict(model.state_dict())

def train(model, state_transitions, target, num_actions):
    #print(state_transitions, num_actions)
    current_states = torch.stack(([torch.Tensor(s.state["observation"]) for s in state_transitions]))
    #current_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions]))
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions]))
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions]))
    next_states = torch.stack(([torch.Tensor(s.next_state["observation"]) for s in state_transitions]))
    #next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transitions]))
    actions = torch.stack(([torch.Tensor(s.action) for s in state_transitions]))
    #actions = [s.action for s in state_transitions]
    print(actions)
    print(next_states)

    state_action_values = model(current_states)
    print(state_action_values)
    next_state_action_values = target(next_states)
    print(next_state_action_values)
    expected_actions = next_state_action_values * gamma + rewards
    print(expected_actions)
    loss = F.smooth_l1_loss(next_state_action_values, expected_actions)
    print("LOSS",loss)
    ##with torch.no_grad():
    ##  target_max = torch.max(target.forward(next_states), dim=1)[0]
    ##  qvals_next = target(next_states).max(-1)[0]
    ##print(target_max)
    ##print(qvals_next)
    model.optimizer.zero_grad()
    ##print(torch.LongTensor(actions))
    ##one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions)
    ##print(one_hot_actions)
    ##qvals = model(current_states)
    ##print(qvals)
    #  td_target = torch.Tensor(rewards) + gamma * target_max * (1 - torch.Tensor(mask))
    #  print(td_target)
    #cs = model.forward(current_states)
    #print(cs.type())
    #print(actions.type())
    #old_val = model.forward(current_states)
    #print(old_val)
    ##print(torch.sum(qvals * one_hot_actions, -1))
    ##loss = (rewards + mask * qvals_next - torch.sum(qvals * one_hot_actions, -1)).mean()
    ##loss_function = nn.MSELoss()
#    print(current_states)
#    print(actions)
#    predictions = model(current_states, actions)
#    print(predictions)
#    print(next_states)
    #loss = loss_function(td_target, old_val)

    #qvals = model(current_states)
    #one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions)

    #loss = ((rewards + mask[:,0] * qvals_next - torch.sum(qvals * one_hot_actions,-1))**2).mean()
    loss.backward()
    model.optimizer.step()
    return loss

import highway_env
if __name__ == '__main__':
    test = False
#    chkpt = None
#    if not test:
#    wandb.init(project="dqn-tutorial", name="dqn-cartpole")
    min_replay_buffer_size = 10000
    sample_size = 2500

    #epsilon_max = 1.0
    epsilon_min = 0.01

    epsilon_decay = 0.999995

    env_steps_before_train = 100
    target_model_update = 150
    env = gym.make("parking-v0")
    #env = Monitor(env, './video', force=True, video_callable=lambda episode: True)
    last_observation = env.reset()
    print(last_observation) #1----

    print(env.observation_space["observation"].shape)
    print(env.action_space.shape[0])
    base_model = Model(env.observation_space["observation"].shape, env.action_space.shape[0])
#    if chkpt is not None:
#        base_model.load_state_dict(torch.load(chkpt))

    target_model = Model(env.observation_space["observation"].shape, env.action_space.shape[0])
    update_target_model(base_model, target_model)

    replay_buffer = ReplayBuffer()

    steps_since_train = 0
    epochs_since_target = 0
    step_num = -1 * min_replay_buffer_size

    episode_rewards = []
    rolling_reward = 0

    tq = tqdm()
#    print(done)

    try:
        while True:
#            if test:
#                env.render()
#            time.sleep(0.05)

            tq.update(1)
            eps = epsilon_decay**(step_num)
            env.render()

#            if test:
#                eps = 0

            if random() < eps:
                action = env.action_space.sample()
                print(action) #2----
            else:
                print(last_observation)
                #action = torch.Tensor(last_observation.action)
                print("Here is the problem maybe.....")
                with torch.no_grad():
                  action = base_model(torch.Tensor(last_observation["observation"]))
                  print(action) #2----

            observation, reward, done, info = env.step(action)
            rolling_reward += reward
            reward = reward/100.0

            replay_buffer.insert(Sarsd(last_observation, action, reward, observation, done))
            last_observation = observation

            if done:
                print("IT IS DONE")
                episode_rewards.append(rolling_reward)
#                print(rolling_reward)
#                if test:
#                    print(rolling_reward)
                rolling_reward = 0
                observation = env.reset()

            steps_since_train += 1
            step_num += 1

            if (not test) and len(replay_buffer.buffer) > min_replay_buffer_size and steps_since_train > env_steps_before_train:

              #print(env.action_space.shape[0])
              print(replay_buffer.sample(sample_size))
              loss = train(base_model, replay_buffer.sample(sample_size), target_model, env.action_space.shape[0])
#              print(replay_buffer, env.action_space)
#              print(replay_buffer.sample(sample_size), env.action_space.n)
#              wandb.log({'loss':loss.detach().item(), 'eps': eps, 'avg_reward': np.mean(episode_rewards)}, step=step_num)
              #print(step_num, loss.detach().item())
              print("LOSS", loss)
              episode_rewards = []
              epochs_since_target += 1
              if epochs_since_target > target_model_update:
                  print("Updating target model")
                  update_target_model(base_model, target_model)
                  epochs_since_target = 0
#                  torch.save(target_model.state_dict(), f"models/{step_num}.pth")
              steps_since_train = 0
              #print(replay_buffer.buffer[0])

    except KeyboardInterrupt:
        pass
    #env.render(close=True)
    env.close()
