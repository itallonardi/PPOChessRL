import chess
import numpy as np
from gym import Env, spaces
from stockfish import Stockfish
import random


class ChessEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, stockfish_path=None, opponent='self', elo_range=(800, 2500), history_length=2):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.opponent = opponent
        self.elo_range = elo_range
        self.stockfish = Stockfish(stockfish_path) if stockfish_path else None
        if self.stockfish:
            self.stockfish.set_elo_rating(random.randint(*self.elo_range))
        self.action_space = spaces.Discrete(218)
        self.observation_space = spaces.Box(low=0, high=1, shape=(
            8, 8, 13 + 12 * history_length), dtype=np.uint8)
        self.history_length = history_length
        self.move_history = []
        self.illegal_moves = 0
        self.legal_moves = 0
        self.checks = 0
        self.checkmates = 0
        self.model_plays_white = True
        self.stockfish_is_opponent = False

    def step(self, action):
        move = self.decode_action(action)
        if move in self.board.legal_moves:
            self.board.push(move)  # The model performs the movement

            if (self.opponent == 'stockfish' or (self.opponent == 'both' and self.stockfish_is_opponent)) and not self.board.is_game_over():
                self.stockfish.set_fen_position(self.board.fen())
                best_move = self.stockfish.get_best_move()
                if best_move:
                    self.board.push(chess.Move.from_uci(best_move))

            self.move_history.append(self.board_to_observation()[:, :, :12])
            if len(self.move_history) > self.history_length:
                self.move_history.pop(0)
            self.legal_moves += 1
            done = self.board.is_game_over()
            previous_evaluation = self.stockfish.get_evaluation() if self.stockfish else None
            reward = self.calculate_reward(previous_evaluation)
            return self.board_to_observation(), reward, done, {}
        else:
            self.illegal_moves += 1
            return self.board_to_observation(), -1, False, {}

    def board_to_observation(self):
        observation = np.zeros(
            (8, 8, 13 + 12 * self.history_length), dtype=np.uint8)
        # Current board state
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                layer_index = (piece.piece_type - 1) * 2 + \
                    (0 if piece.color == chess.WHITE else 1)
                rank, file = divmod(square, 8)
                observation[rank, file, layer_index] = 1
        if self.board.turn == chess.WHITE:
            observation[:, :, 12] = 1  # Turn layer

        # Add historical layers
        for i, hist in enumerate(self.move_history):
            observation[:, :, 13 + 12 * i: 13 + 12 * (i + 1)] = hist

        return observation

    def reset(self):
        self.board.reset()
        self.move_history = []
        self.illegal_moves = 0
        self.legal_moves = 0
        self.checks = 0
        self.checkmates = 0
        if self.opponent in ['stockfish', 'both']:
            self.stockfish.set_elo_rating(random.randint(*self.elo_range))
            self.model_plays_white = random.choice([True, False])
            if self.opponent == 'both':
                self.stockfish_is_opponent = random.choice([True, False])
            else:
                self.stockfish_is_opponent = True

            if not self.model_plays_white and self.stockfish_is_opponent:
                # Stockfish makes the first move if playing as white
                self.stockfish.set_fen_position(self.board.fen())
                first_move = self.stockfish.get_best_move()
                if first_move:
                    self.board.push(chess.Move.from_uci(first_move))
        else:
            self.model_plays_white = True
            self.stockfish_is_opponent = False
        return self.board_to_observation()

    def decode_action(self, action):
        legal_moves = list(self.board.legal_moves)
        return legal_moves[action % len(legal_moves)]

    def render(self, mode='human'):
        if mode == 'human':
            print("Current board state:")
            print(self.board)

    def calculate_reward(self, previous_evaluation=None):
        if self.stockfish:
            self.stockfish._set_option("MultiPV", 3)
            self.stockfish._set_option("UCI_ShowWDL", True)
            self.stockfish.set_fen_position(self.board.fen())
            current_evaluation = self.stockfish.get_evaluation()
            best_moves = self.stockfish.get_top_moves(3)

            if current_evaluation['type'] == 'mate':
                return 100000 if current_evaluation['value'] > 0 else -100000

            reward = 0
            if previous_evaluation and previous_evaluation['type'] == 'cp' and current_evaluation['type'] == 'cp':
                reward += (current_evaluation['value'] -
                           previous_evaluation['value'])

            for move in best_moves:
                if move.get('Mate') is not None:
                    mate_in = move['Mate']
                    move_value = 100000 / mate_in  # adds more value to a mate in fewer moves
                else:
                    move_value = move.get('Centipawn', 0)

                reward += move_value * 0.1

                if 'wdl' in move:
                    win, draw, loss = move['wdl']
                    reward += (win - loss) * 0.05

            return reward
        return 0
