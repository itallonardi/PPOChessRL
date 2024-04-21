from stockfish import Stockfish
from dotenv import load_dotenv
import os

load_dotenv()

stockfish_path = os.getenv(
    'STOCKFISH_PATH', 'stockfish/stockfish-ubuntu-x86-64-avx2')

stockfish = Stockfish(stockfish_path)

stockfish.set_fen_position(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
stockfish.make_moves_from_current_position(["e2e4"])

print(stockfish.get_board_visual())

evaluation = stockfish.get_evaluation()
print("Evaluation after 1.e4:", evaluation)
