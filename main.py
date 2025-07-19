# ---------------------------------------------------------------
# TRAIN / TEST DDPG (really TD3-lite) on MountainCarContinuous-v0
# ---------------------------------------------------------------
import gym
import numpy as np
import torch           # makes sure cuda kernels are initialised early
from collections import deque
import time, os, random
from ddpg_agent import DDPGAgent

# ----------------------------------------------------------------
# 1.  Environment
# ----------------------------------------------------------------
ENV_NAME = "MountainCarContinuous-v0"
env = gym.make(ENV_NAME)

state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print(f"State dim: {state_dim},  Action dim: {action_dim}")

# ----------------------------------------------------------------
# 2.  Agent
# ----------------------------------------------------------------
agent = DDPGAgent(state_dim, action_dim,
                  buffer_size=200_000,
                  gamma=0.99,
                  tau=0.005,
                  policy_delay=2,            # TD3 style
                  actor_lr=1e-4,
                  critic_lr=1e-3,
                  warmup_steps=5_000,
                  batch_size=1024,
                  chkpt_dir="chkpt/")

print("Actor lives on", next(agent.actor.parameters()).device)

# ----------------------------------------------------------------
# 3.  Training loop
# ----------------------------------------------------------------
n_episodes        = 500
max_env_steps     = env._max_episode_steps  # 999 for MountainCarContinuous
target_mean_score = 90                      # “solved” threshold
print_every       = 10
update_every = 4
gradient_steps = 4

scores      = []
scores_ma   = deque(maxlen=100)            # moving average window

for ep in range(1, n_episodes + 1):
    # Gym ≥ 0.26 returns (obs, info)
    out = env.reset()
    state = out[0] if isinstance(out, tuple) else out
    ep_reward = 0.0
    step_count = 0

    agent.noise.reset()                    # reset exploration noise each episode

    for t in range(max_env_steps):
        action = agent.select_action(state, add_noise=True)
        step_out = env.step(action)
        if len(step_out) == 5:             # new API: obs, reward, term, trunc, info
            next_state, reward, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:                              # old API: obs, reward, done, info
            next_state, reward, done, _ = step_out

        agent.store_transition(state, action, reward, next_state, terminated, truncated)
        step_count += 1
        if step_count % update_every == 0:
            for _ in range(gradient_steps):
                agent.train()

        ep_reward += reward
        state = next_state
        if done:
            break

    # bookkeeping
    scores.append(ep_reward)
    scores_ma.append(ep_reward)
    mean_last_100 = np.mean(scores_ma)

    if ep % print_every == 0 or mean_last_100 >= target_mean_score:
        print(f"Episode {ep:4d} | Episode reward: {ep_reward:8.2f} | "
              f"100-episode mean: {mean_last_100:7.2f}")

    if mean_last_100 >= target_mean_score:
        print(f"Solved after {ep} episodes! 🎉")
        break

# ----------------------------------------------------------------
# 4.  Save model & reward history
# ----------------------------------------------------------------
agent.save("mountaincar_ddpg")
np.save("training_scores.npy", np.array(scores))

# ----------------------------------------------------------------
# 5.  Evaluation (greedy, no noise, render)
# ----------------------------------------------------------------
eval_episodes = 5
for ep in range(eval_episodes):
    state, _ = env.reset()
    ep_reward = 0.0
    for t in range(max_env_steps):
        env.render()

        action = agent.select_action(state, add_noise=False)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        if done:
            break
    print(f"[EVAL] Episode {ep+1}: reward = {ep_reward:.2f}")

env.close()
