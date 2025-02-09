import os
# Substituímos a importação de PPO pela do sb3-contrib (MaskablePPO):
from sb3_contrib import MaskablePPO


def initialize_ppo(env, model_path=None):
    """
    Initializes (or loads) a MaskablePPO model for the given environment.
    This version uses sb3_contrib's MaskablePPO instead of the standard PPO
    to handle action masking (legal vs. illegal moves).
    """
    if model_path and os.path.exists(model_path):
        print("Loading existing model from:", model_path)
        model = MaskablePPO.load(model_path, env=env)
        print("model policy:", model.policy)
    else:
        print("Creating new MaskablePPO model")
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./ppo_chess_tensorboard/",
            learning_rate=0.0003,
            n_steps=4096,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01
        )
    return model
