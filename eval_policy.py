import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np

"""
    This file is used to evaluate a trained policy (actor model) after
    training it in main.py with ppo.py. The idea is that our policy
    exists independently of the PPO training code, so we can load the
    model weights and run it in any environment that is consistent.

    Usage example in main.py: 
        if mode == 'test':
            test(env=env, actor_model='ppo_actor.pth')
"""

def _log_summary(ep_len, ep_ret, ep_num):
    """
        Print to stdout what we've logged for the most recent episode.

        Parameters:
            ep_len (float): The total length (timesteps) of this episode.
            ep_ret (float): The total return for this episode.
            ep_num (int): Which episode number is this?

        Returns:
            None
    """
    # Round decimal places for more aesthetic logging messages
    ep_len = str(round(ep_len, 2))
    ep_ret = str(round(ep_ret, 2))

    # Print logging statements
    print(flush=True)
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Episodic Length: {ep_len}", flush=True)
    print(f"Episodic Return: {ep_ret}", flush=True)
    print(f"------------------------------------------------------", flush=True)
    print(flush=True)


def rollout(policy, env, render=False):
    """
        Returns a generator to roll out each episode given a trained policy 
        and environment to test on.

        Parameters:
            policy (torch.nn.Module): The trained policy (actor) to evaluate.
            env (gym.Env): The environment to evaluate the policy on.
            render (bool): If True, will call `env.render()` each step.

        Returns:
            A generator object that yields (episodic_length, episodic_return, 
            portfolio_values, actions_list) for each episode.

        Note:
            This generator runs forever (until you manually stop). 
            Each iteration = one episode.
    """
    while True:
        obs, info = env.reset()  # Reset the environment
        done = False

        step_counter = 0
        ep_ret = 0.0
        ep_len = 0

        # Track portfolio values and actions at each step
        portfolio_values = []
        actions_list = []

        while not done:
            step_counter += 1

            if render:
                env.render()

            # Convert observation to torch if it's not already
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.tensor(obs, dtype=torch.float)
            else:
                obs_tensor = obs  # Already a torch tensor

            # Forward pass through policy to get raw (mean) action
            # If the policy is just an MLP, it should output a 1D tensor of shape (action_dim,).
            with torch.no_grad():
                action_raw = policy(obs_tensor).detach()

            # If we want to treat these raw outputs as logits or means,
            # we could do something like a softmax or clip. E.g.:
            action_clamped = F.softmax(action_raw, dim=-1).numpy()
            actions_list.append(action_clamped)

            # Step the environment
            obs, rew, terminated, truncated, info = env.step(action_clamped)
            done = terminated or truncated

            # Keep track of portfolio values & accumulative reward
            portfolio_values.append(info.get("portfolio_value", None))
            ep_ret += rew

        ep_len = step_counter

        # Once the episode is done, yield the info
        yield ep_len, ep_ret, portfolio_values, actions_list


def eval_policy(policy, env, render=False, csv_prefix="portfolio_values_episode_"):
    """
        Main function to evaluate our policy with. It iterates the generator
        object `rollout()`, which simulates each episode and returns (ep_len, 
        ep_ret, portfolio_vals, actions).

        Parameters:
            policy (torch.nn.Module): The trained policy (actor).
            env (gym.Env): The environment to test on.
            render (bool): If True, calls `env.render()` each step.
            csv_prefix (str): Base filename for CSV logs.

        Returns:
            None

        Note:
            The function runs indefinitely, one episode at a time, 
            until you kill the process.
    """
    for ep_num, (ep_len, ep_ret, portfolio_values, actions_list) in enumerate(rollout(policy, env, render)):
        # Log to stdout
        _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)

        # Optionally save each episode’s data (portfolio value, actions, etc.) to CSV
        data_dict = {
            "step": range(len(portfolio_values)),
            "portfolio_value": portfolio_values
        }

        # We can store the action dimensions in separate columns
        if len(actions_list) > 0:
            action_dim = len(actions_list[0])
            for i in range(action_dim):
                data_dict[f"weight_{i}"] = [a[i] for a in actions_list]

        df = pd.DataFrame(data_dict)
        csv_filename = f"{csv_prefix}{ep_num+17}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Saved portfolio values for episode {ep_num+17} to {csv_filename}")

        # Stop after 1 episode if you only want a single run:
        # break
