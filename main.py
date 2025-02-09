import argparse
from app.envs.chess_env import ChessEnv
from app.agents.ppo_agent import initialize_ppo
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from dotenv import load_dotenv
import os
from app.callbacks.training_log_callback import ChessRatingLogger

load_dotenv()


def make_env(env_id, stockfish_path, opponent, elo_range):
    """
    Creates a function that instantiates a ChessEnv with fixed arguments,
    needed for SubprocVecEnv (each subprocess will call this function).
    """
    def _init():
        env = ChessEnv(
            stockfish_path=stockfish_path,
            opponent=opponent,
            elo_range=elo_range,
            environment_id=env_id  # pass the ID to identify in logs
        )
        return env
    return _init


def main(args):
    stockfish_path = os.getenv(
        'STOCKFISH_PATH', 'stockfish/stockfish-ubuntu-x86-64-avx2')

    # If num_envs == 1, create only one normal env; otherwise, create a SubprocVecEnv
    if args.num_envs == 1:
        env = ChessEnv(stockfish_path=stockfish_path,
                       opponent=args.opponent,
                       elo_range=(args.elo_start, args.elo_end),
                       environment_id=0)
    else:
        env_fns = []
        for i in range(args.num_envs):
            env_fns.append(
                make_env(i, stockfish_path, args.opponent,
                         (args.elo_start, args.elo_end))
            )
        # SubprocVecEnv or DummyVecEnv, here we choose SubprocVecEnv
        env = SubprocVecEnv(env_fns)

    model_path = "models/chess_model.zip"
    model = initialize_ppo(env, model_path=model_path)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq, save_path='./models/', name_prefix='chess_model'
    )
    rating_logger = ChessRatingLogger(
        log_path="training_rating_log.csv", verbose=1
    )

    try:
        callback_list = CallbackList([checkpoint_callback, rating_logger])

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback_list
        )
        model.save(model_path)
        print(f"Model trained and saved at: {model_path}")
    except Exception as e:
        print(f"Error during model loading or training: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a chess model against specified opponents.'
    )
    parser.add_argument('--opponent', type=str, choices=['self', 'stockfish', 'both'], default="self",
                        help='Who the model plays against: self, stockfish, or both.')
    parser.add_argument('--elo_start', type=int, default=800,
                        help='Starting ELO rating of Stockfish.')
    parser.add_argument('--elo_end', type=int, default=2000,
                        help='Ending ELO rating of Stockfish.')
    parser.add_argument('--total_timesteps', type=int, default=102400,
                        help='Total number of timesteps for training the model.')
    parser.add_argument('--save_freq', type=int, default=4096,
                        help='Frequency of saving the model.')
    parser.add_argument('--num_envs', type=int, default=1,
                        help='Number of parallel environments to run.')

    args = parser.parse_args()
    main(args)
