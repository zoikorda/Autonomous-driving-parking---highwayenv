import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import numpy as np
import wandb
from collections import deque
from random import sample, random

from dataclasses import dataclass
from typing import Any
from random import sample

import gym
import highway_env
import time

class DQNAgent:
    def __init__(self, model):
        self.model = model

    def get_actions(self, observations):
        # observations; shape is(N, 4)
        q_vals = self.model(observations)
        # q_vals shape (N,2)
        return q_vals.max(-1)[-1]

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

def update_target_model(model, target):
  target.load_state_dict(model.state_dict())

def train(model, state_transitions, target, num_actions):
    print(state_transitions, num_actions)
    current_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions]))
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions]))
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions]))
    next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transitions]))
    actions = [s.action for s in state_transitions]

    with torch.no_grad():
      qvals_next = target(next_states).max(-1)[0]

    model.optimizer.zero_grad()
    qvals = model(current_states)
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions)

    loss = ((rewards + mask[:,0] * qvals_next - torch.sum(qvals * one_hot_actions,-1))**2).mean()
    loss.backward()
    model.optimizer.step()
    return loss

#def main(test=False, chkpt=None):
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

    env = gym.make("CartPole-v0")
    #env = Monitor(env, './video', force=True, video_callable=lambda episode: True)
    last_observation = env.reset()

    base_model = Model(env.observation_space.shape, env.action_space.n)
#    if chkpt is not None:
#        base_model.load_state_dict(torch.load(chkpt))

    target_model = Model(env.observation_space.shape, env.action_space.n)
    update_target_model(base_model, target_model)

    replay_buffer = ReplayBuffer()

    steps_since_train = 0
    epochs_since_target = 0
    step_num = -1 * min_replay_buffer_size

    episode_rewards = []
    rolling_reward = 0

    tq = tqdm()

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
            else:
                action = base_model(torch.Tensor(last_observation)).max(-1)[-1].item()

            observation, reward, done, info = env.step(action)
            rolling_reward += reward
            reward = reward/100.0

            replay_buffer.insert(Sarsd(last_observation, action, reward, observation, done))
            last_observation = observation

            if done:
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
              #print(replay_buffer.sample(sample_size))
              loss = train(base_model, replay_buffer.sample(sample_size), target_model, env.action_space.n)
              print(replay_buffer, env.action_space)
              print(replay_buffer.sample(sample_size), env.action_space.n)
#              wandb.log({'loss':loss.detach().item(), 'eps': eps, 'avg_reward': np.mean(episode_rewards)}, step=step_num)
              #print(step_num, loss.detach().item())
              episode_rewards = []
              epochs_since_target += 1
              if epochs_since_target > target_model_update:
                  print("Updating target model")
                  update_target_model(base_model, target_model)
                  epochs_since_target = 0
                  torch.save(target_model.state_dict(), f"models/{step_num}.pth")
              steps_since_train = 0
              #print(replay_buffer.buffer[0])

    except KeyboardInterrupt:
        pass
    #env.render(close=True)
    env.close()

#if __name__ == '__main__':
#    main(False, None)
#    import ipdb; ipdb.set_trace()
#    for _ in range(1000):
#      env.render()
#      time.sleep(0.1)
#      import ipdb; ipdb.set_trace()
#      action = env.action_space.sample()
#      observation, reward, done, info = env.step(action)
#      if done:
#        observation = env.reset()
#    env.close
    #show_video()
