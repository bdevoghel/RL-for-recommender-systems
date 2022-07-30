import gym
import virtualTB
import numpy as np
from tqdm import tqdm

from utils.utils import DDPGAgent, TD3Agent

nb_episodes = 2000
verbose = False
plot_q = True
# algo = "DDPG"
algo = "TD3"

env = gym.make("VirtualTB-v0")

if algo == "DDPG":
    print("Creating DDPG agent")
    agent = DDPGAgent(
        observations_space=env.observation_space,
        action_space=env.action_space,
        ounoise_sigma=0.5,
        ounoise_sigma_min=0.02,
        ounoise_decay_period=40 * nb_episodes,
        memory_sample_size=100,
        alpha=0.0001,  # actor lr
        beta=0.01,  # critic lr
        gamma=0.95,  # discount factor for future rewards
        tau=0.001,  # target network update weight
        hidden_size=128,
    )
elif algo == "TD3":
    print("Creating TD3 agent")
    agent = TD3Agent(
        alpha=0.0001,  # actor lr
        beta=0.01,  # critic lr
        gamma=0.99,  # discount factor for future rewards
        tau=0.001,  # target network update weight
        env=env,
        input_dims=env.observation_space.shape,
        n_actions=env.action_space.shape[0],
        layer1_size=128,
        layer2_size=64,
        batch_size=400,
        warmup=nb_episodes//10,
    )
    # agent.load_models()

history_rewards = []  # clicks on products
history_steps = []  # pages viewed
history_ctr = []  # click-through-rate at end of episode
history_q = []  # cummulative q value of predicted actions
history_ounoise_sigma = []
for i_episode in tqdm(range(nb_episodes), "Episodes", ncols=64):
    state = env.reset()

    episode_reward = 0
    episode_steps = 0
    episode_q = 0
    has_left = False
    while not has_left:
        render = env.render(mode="return", short=True)
        if verbose:
            print(render)

        action = agent.select_action(state, i_episode)
        if plot_q:
            q = agent.get_q(state, action)

        next_state, reward, has_left, info = env.step(action)
        has_left = bool(has_left)

        episode_reward += reward
        episode_steps += 1
        if plot_q:
            episode_q += q

        agent.remember(state, action, reward, next_state, int(not has_left))
        agent.learn()

        if verbose:
            print(f"{' '*32} r:{reward} d:{has_left} {info}\n" + f"{' '*32} e:{env}")

        state = next_state

    if verbose:
        print(f"{env.render(mode='return', short=True)} -- r:{episode_reward} s:{episode_steps}")

    history_rewards.append(episode_reward)
    history_steps.append(episode_steps)
    history_q.append(episode_q / episode_steps)
    history_ctr.append(info['CTR'])
    if algo == "DDPG":
        history_ounoise_sigma.append(agent.ounoise.sigma)

    if algo == "DDPG":
        agent.ounoise.reset(t=i_episode)

print(f"\n{agent.updates} learning steps performed on {nb_episodes} episodes.")


# Render info of training in plot and save it
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

rolling_window_size = 0.05
rolling_window_width = int(nb_episodes * rolling_window_size)
history_rewards_average = [
    np.mean(history_rewards[i : i + rolling_window_width])
    for i in range(0, nb_episodes, rolling_window_width)
]
history_ctr_average = [
    np.mean(history_ctr[i : i + rolling_window_width])
    for i in range(0, nb_episodes, rolling_window_width)
]
history_q_average = [
    np.mean(history_q[i : i + rolling_window_width])
    for i in range(0, nb_episodes, rolling_window_width)
]

ax1.plot(
    np.arange(-rolling_window_width, nb_episodes, rolling_window_width)
    + rolling_window_width,
    [0] + history_rewards_average,
    "C0",
    label=f"rewards (rolling average)",
)
ax1.scatter(np.arange(len(history_rewards)), history_rewards, s=.5, c="C0", alpha=.7, label="rewards")
if plot_q:
    ax1.scatter(np.arange(len(history_q)), history_q, s=.5, c="C3", alpha=.5, label="average Q")
    ax1.plot(
        np.arange(-rolling_window_width, nb_episodes, rolling_window_width)
        + rolling_window_width,
        [0] + history_q_average,
        c="C3",
        label=f"average Q (rolling average)",
    )
    ax1.set_ylim(top=max(max(history_q_average), max(history_rewards_average))+100, bottom=0)
if algo == "DDPG":
    ax2.plot(history_ounoise_sigma, c="C2", label="sigma")
ax2.scatter(np.arange(len(history_ctr)), history_ctr, s=.5, c="C1", alpha=.5, label="CTR")
ax2.plot(
    np.arange(-rolling_window_width, nb_episodes, rolling_window_width)
    + rolling_window_width,
    [0] + history_ctr_average,
    c="C1",
    label=f"CTR (rolling average)",
)

ax1.set_xlabel("Episodes")
ax1.set_ylabel("Reward")
if algo == "DDPG":
    ax2.set_ylabel("OUNoise sigma & CTR")
else:
    ax2.set_ylabel("CTR")


ax2.set_ylim(top=max(history_ctr)+0.3)

plt.title(f"{agent}")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.savefig(
    f"training_results/___training_{agent}_{nb_episodes}eps_{agent.updates}upd.png"
)
