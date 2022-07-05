import gym
import virtualTB
import numpy as np
from tqdm import tqdm

from utils import DDPGAgent

nb_episodes = 200
verbose = False

env = gym.make("VirtualTB-v0")
agent = DDPGAgent(
    observations_space=env.observation_space,
    action_space=env.action_space,
    ounoise_decay_period=10 * nb_episodes,
    memory_sample_size=128,
    gamma=0.95,
    tau=0.001,
    hidden_size=128,
)

total_rewards = []  # clicks on products
total_steps = []  # pages viewed
total_ounoise_sigma = []
for i_episode in tqdm(range(nb_episodes), "Episodes", ncols=64):
    state = env.reset()

    episode_reward = 0
    episode_steps = 0
    while True:
        render = env.render(mode="return", short=True)
        if verbose:
            print(render)

        action = agent.select_action(state)

        next_state, reward, has_left, info = env.step(action)
        has_left = bool(has_left)

        episode_reward += reward
        episode_steps += 1

        agent.memory.push(state, action, reward, next_state, int(not has_left))
        agent.learn()

        if verbose:
            print(f"{' '*32} r:{reward} d:{has_left} {info}\n" + f"{' '*32} e:{env}")

        if has_left:
            break
    if verbose:
        env.render(short=True)
        print("\n\n")

    total_rewards.append(episode_reward)
    total_steps.append(episode_steps)
    total_ounoise_sigma.append(agent.ounoise.sigma)

    agent.ounoise.reset(t=i_episode)

print(f"\n{agent.updates} learning steps performed on {nb_episodes} episodes.")
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

total_rewards_rolling_window_size = 0.05
rolling_window_width = int(nb_episodes * total_rewards_rolling_window_size)
total_rewards_average = [
    np.mean(total_rewards[i : i + rolling_window_width])
    for i in range(0, nb_episodes, rolling_window_width)
]

ax1.plot(
    np.arange(-rolling_window_width, nb_episodes, rolling_window_width)
    + rolling_window_width,
    [0] + total_rewards_average,
    "C0",
    label=f"rewards (rolling average)",
)
ax1.plot(total_rewards, "C0", alpha=0.7, label="rewards")
ax1.plot(total_steps, "C1", alpha=0.5, label="steps")
ax2.plot(total_ounoise_sigma, "C2", label="sigma")

ax1.set_xlabel("Episodes")
ax1.set_ylabel("Reward & Num steps")
ax2.set_ylabel("OUNoise sigma", color="C2")

plt.title(f"{agent}")
ax1.legend(loc="center left")
ax2.legend(loc="center right")

plt.savefig(
    f"training_results/training_{agent}_{nb_episodes}eps_{agent.updates}upd.png"
)
