import os
import sys
import torch
from ppo import PPO
from network import FeedForwardNN
from eval_policy import eval_policy
from trading_env_v3 import PortfolioEnv
from deepar import DeepARModel
from arguments import get_args

def train(env, hyperparameters, actor_model, critic_model):
    print("Starting PPO training...")
    model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)
    if actor_model and critic_model:
        print(f"Loading actor model from {actor_model} and critic model from {critic_model}")
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
    elif actor_model or critic_model:
        print("Error: Specify both actor and critic model paths or neither")
        sys.exit(1)
    model.learn(total_timesteps=1_000_000)

def test(env, actor_model):
    print(f"Testing using actor model: {actor_model}")
    if not actor_model:
        print("No actor model provided. Exiting.")
        sys.exit(1)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = FeedForwardNN(obs_dim, act_dim)
    policy.load_state_dict(torch.load(actor_model))
    eval_policy(policy=policy, env=env, render=True)

def main():
    args = get_args()
    hyperparameters = {
        'timesteps_per_batch': 252,
        'max_timesteps_per_episode': 252,
        'gamma': 0.99,
        'n_updates_per_iteration': 10,
        'lr': 1e-4,
        'clip': 0.2,
        'render': True,
        'render_every_i': 10
    }

    # Load the pretrained DeepAR model if available.
    deepar_model = None
    deepar_model_path = "deepar_model.pth"  # Default path; adjust or add an argument in arguments.py if needed.
    if os.path.exists(deepar_model_path):
        print(f"Loading DeepAR model from {deepar_model_path}")
        deepar_model = DeepARModel()
        deepar_model.load_state_dict(torch.load(deepar_model_path))
        deepar_model.eval()
    else:
        print("Pretrained DeepAR model not found. Continuing without DeepAR predictions.")

    # Create the Portfolio Trading Environment with DeepAR integration.
    env = PortfolioEnv(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        start_date='2024-01-01',
        end_date='2025-01-01',
        initial_balance=100000,
        window_len=20,
        deepar_model=deepar_model
    )
    print(f"Observation Space shape: {env.observation_space.shape}")

    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters,
              actor_model=args.actor_model, critic_model=args.critic_model)
    else:
        test(env=env, actor_model=args.actor_model)

if __name__ == '__main__':
    main()



'''
Train to Test Check list:
1. confirm hyper parameters -> dates and batch/episode sizes
2. ensure file paths are correct (ppo actor path for training, portfolio values path for testing) -> don't overwrite existing data!
3. switch environment from selecting random start date (for training) to starting at first date (for testing)
4. python main2.py --mode test --actor_model="ppo_actor_v10.pth"

'''