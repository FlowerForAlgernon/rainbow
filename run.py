import time
import math
import numpy as np
from itertools import count
import gym
import torch
import matplotlib.pyplot as plt

from environment import make_atari, wrap_deepmind
from agent import Agent
from config import Config



def train():
    steps_done = 0
    mean_100ep_reward = []
    episode_reward = []

    for episode in range(config.n_episodes):
        obs = env.reset()
        state = agent.get_state(obs, config)
        total_reward = 0.0

        fraction = min(episode / config.n_episodes, 1.0)
        config.beta = config.beta + fraction * (1.0 - config.beta)

        for t in count():
            epsilon = config.epsilon_min + (config.epsilon_max - config.epsilon_min) * math.exp(-1. * steps_done / config.eps_decay)
            action = agent.select_action(state, epsilon, config)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            next_state = agent.get_state(obs, config) if not done else None
            reward = torch.tensor([reward])

            state = state.to(torch.device("cpu"))
            action = action.to(torch.device("cpu"))
            next_state = None if done else next_state.to(torch.device("cpu"))
            reward = reward.to(torch.device("cpu"))

            #agent.memory.push(state, action, next_state, reward)
            # N Step
            transition = agent.memory_n.push(state, action, next_state, reward)
            if transition:
                agent.memory.push(*transition)

            state = None if done else next_state.to(config.device)
            steps_done += 1

            if len(agent.memory) >= config.learning_start:
                agent.optimize_model(config)

            if steps_done % config.target_update == 0 and steps_done != 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

            if done:
                break

        print('Total steps: {} \t Episode: {}/{} \t Total reward: {} \t Memory Allocated: {}'.format(steps_done, episode, t, total_reward, torch.cuda.memory_allocated(0)))
        episode_reward.append(total_reward)
        if len(episode_reward) < 100:
            mean_100ep_reward.append(round(np.mean(episode_reward),1))
        else:
            mean_100ep_reward.append(round(np.mean(episode_reward[-100:]),1))
        if episode % 10 == 0 and episode != 0:
            plt.figure()
            plt.plot(range(len(mean_100ep_reward)), mean_100ep_reward)
            plt.savefig('mean_100ep_reward.png')
            plt.close()
            plt.figure()
            plt.plot(range(len(episode_reward)), episode_reward)
            plt.savefig('episode_reward.png')
            plt.close()

    env.close()


def test(env):
    env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video')

    obs = env.reset()
    state = agent.get_state(obs, config)
    total_reward = 0.0

    for t in count():
        env.render()
        time.sleep(0.02)

        action = agent.policy_net(state).max(1)[1].view(1,1)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        next_state = agent.get_state(obs, config) if not done else None
        state = next_state

        if done:
            break

    print("Finished Episode {} with reward {}".format(episode, total_reward))
    env.close()


if __name__ == '__main__':
    env = make_atari('PongNoFrameskip-v4')
    env = wrap_deepmind(env, scale=False, frame_stack=True)

    config = Config()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.n_actions = env.action_space.n
    _, config.c, config.h, config.w = Agent.get_state(env.reset(), config).shape

    agent = Agent(config)

    train()
    #torch.save(agent.policy_net, "dqn_pong_model")
    #agent.policy_net = torch.load("dqn_pong_model")
    #test(env)
