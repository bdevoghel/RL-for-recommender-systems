import gym
import virtualTB
import numpy as np
from tqdm import tqdm

from utils import Agent, Memory

env = gym.make('VirtualTB-v0')
agent = Agent(env.action_space)
memory = Memory()

nb_episodes = 1000
rewards = []
steps = []

verbose = False

for i_episode in tqdm(range(nb_episodes), "Episodes", ncols=64):
    state = env.reset()

    episode_reward = 0
    episode_steps = 0
    while True:
        render = env.render(mode='return', short=True)
        if verbose:
            print(render)

        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        done = bool(done)

        episode_reward += reward
        episode_steps += 1

        memory.push(state, action, rewards, next_state, done)
        agent.learn(memory)

        if verbose:
            print(f"{' '*32} r:{reward} d:{done} {info}\n" +
                  f"{' '*32} e:{env}"
            )
        
        if done:
            break
    if verbose:
        env.render(short=True)
        print("\n\n")

    rewards.append(episode_reward)
    steps.append(episode_steps)

rolling_n = 10
print(f"Rewards mean : {np.mean(rewards)}\n        rolling (last {nb_episodes//rolling_n}) average : {[np.mean(rewards[i:i+nb_episodes//rolling_n]) for i in range(0, nb_episodes, nb_episodes//rolling_n)]}")
print(f"Steps   mean : {np.mean(steps)}\n        rolling (last {nb_episodes//rolling_n}) average : {[np.mean(steps[i:i+nb_episodes//rolling_n]) for i in range(0, nb_episodes, nb_episodes//rolling_n)]}")
