import os
from stable_baselines3 import PPO


def initialize_ppo(env, model_path=None):

    if model_path and os.path.exists(model_path):
        print("Loading existing model from:", model_path)
        model = PPO.load(model_path, env=env)
        print("model policy:", model.policy)
    else:
        print("Creating new model")
        model = PPO("MlpPolicy", env, verbose=1,
                    tensorboard_log="./ppo_chess_tensorboard/",
                    learning_rate=0.0003, n_steps=4096, batch_size=64,
                    n_epochs=10, gamma=0.99, ent_coef=0.01)
    return model
