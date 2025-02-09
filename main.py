import argparse
from app.envs.chess_env import ChessEnv
from app.agents.ppo_agent import initialize_ppo
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from dotenv import load_dotenv
import os
from app.callbacks.training_log_callback import ChessRatingLogger

load_dotenv()


def main(args):
    stockfish_path = os.getenv(
        'STOCKFISH_PATH', 'stockfish/stockfish-ubuntu-x86-64-avx2')
    env = ChessEnv(stockfish_path=stockfish_path,
                   opponent=args.opponent,
                   elo_range=(args.elo_start, args.elo_end))
    model_path = "models/chess_model.zip"
    model = initialize_ppo(env, model_path=model_path)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq, save_path='./models/', name_prefix='chess_model')
    rating_logger = ChessRatingLogger(
        log_path="training_rating_log.csv", verbose=1)

    try:
        # Adicionamos ambos os callbacks em uma CallbackList
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
        description='Train a chess model against specified opponents.')
    parser.add_argument('--opponent', type=str, choices=['self', 'stockfish', 'both'], default="self",
                        help='Who the model plays against: self, stockfish, or both.')
    parser.add_argument('--elo_start', type=int, default=800,
                        help='Starting ELO rating of Stockfish.')
    parser.add_argument('--elo_end', type=int, default=2000,
                        help='Ending ELO rating of Stockfish.')
    parser.add_argument('--total_timesteps', type=int, default=100000,
                        help='Total number of timesteps for training the model.')
    parser.add_argument('--save_freq', type=int, default=10000,
                        help='Frequency of saving the model.')

    args = parser.parse_args()
    main(args)
