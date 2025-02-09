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
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(8, 8, 13 + 12 * history_length),
            dtype=np.uint8
        )
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
            # The model performs the move
            self.board.push(move)

            # If the opponent is Stockfish (or "both" with Stockfish as current opponent),
            # and the game isn't over, let Stockfish make a move.
            if (
                (self.opponent == 'stockfish' or (
                    self.opponent == 'both' and self.stockfish_is_opponent))
                and not self.board.is_game_over()
            ):
                self.stockfish.set_fen_position(self.board.fen())
                best_move = self.stockfish.get_best_move()
                if best_move:
                    self.board.push(chess.Move.from_uci(best_move))

            # Store the current board state in the move history (without the turn layer).
            self.move_history.append(self.board_to_observation()[:, :, :12])
            if len(self.move_history) > self.history_length:
                self.move_history.pop(0)

            self.legal_moves += 1
            done = self.board.is_game_over()
            previous_evaluation = self.stockfish.get_evaluation() if self.stockfish else None
            reward = self.calculate_reward(previous_evaluation)
            return self.board_to_observation(), reward, done, {}
        else:
            # Illegal move
            self.illegal_moves += 1
            return self.board_to_observation(), -1, False, {}

    def board_to_observation(self):
        """
        Returns an array representation of the board, with:
        - 12 channels for the pieces (6 types x 2 colors),
        - 1 channel for the side to move (White or Black),
        - 12 * history_length channels for historical states.
        """
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

        # Turn layer (if it's White's turn, set to 1)
        if self.board.turn == chess.WHITE:
            observation[:, :, 12] = 1

        # Historical layers
        for i, hist in enumerate(self.move_history):
            start = 13 + 12 * i
            end = 13 + 12 * (i + 1)
            observation[:, :, start:end] = hist

        return observation

    def reset(self):
        self.board.reset()
        self.move_history = []
        self.illegal_moves = 0
        self.legal_moves = 0
        self.checks = 0
        self.checkmates = 0

        # If Stockfish is set as the opponent (or "both"), initialize and possibly let Stockfish move first.
        if self.opponent in ['stockfish', 'both']:
            self.stockfish.set_elo_rating(random.randint(*self.elo_range))
            self.model_plays_white = random.choice([True, False])
            if self.opponent == 'both':
                self.stockfish_is_opponent = random.choice([True, False])
            else:
                self.stockfish_is_opponent = True

            # If the model is not playing White and Stockfish is the opponent, Stockfish moves first.
            if not self.model_plays_white and self.stockfish_is_opponent:
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
        """
        Calculates a reward based on Stockfish evaluation.
        - If it's a mate evaluation, reward is +/-100000 (depending on perspective).
        - Otherwise, uses centipawn difference and top moves from Stockfish to shape the reward.
        """
        if self.stockfish:
            self.stockfish._set_option("MultiPV", 3)
            self.stockfish._set_option("UCI_ShowWDL", True)
            self.stockfish.set_fen_position(self.board.fen())
            current_evaluation = self.stockfish.get_evaluation()
            best_moves = self.stockfish.get_top_moves(3)

            # If the evaluation type is mate
            if current_evaluation['type'] == 'mate':
                # Convert to White's perspective
                raw_reward = 100000 if current_evaluation['value'] > 0 else -100000
                # If the agent is playing Black, invert the sign
                return raw_reward if self.model_plays_white else -raw_reward

            # Normal evaluation in centipawns
            raw_reward = 0
            if (
                previous_evaluation
                and previous_evaluation['type'] == 'cp'
                and current_evaluation['type'] == 'cp'
            ):
                # Difference in centipawns from White's perspective
                raw_reward += current_evaluation['value'] - \
                    previous_evaluation['value']

            # Reward based on Stockfish's top moves
            for move in best_moves:
                if move.get('Mate') is not None:
                    mate_in = move['Mate']
                    # More value for mate in fewer moves
                    move_value = 100000 / mate_in
                else:
                    move_value = move.get('Centipawn', 0)

                raw_reward += move_value * 0.1

                if 'wdl' in move:
                    win, draw, loss = move['wdl']
                    # WDL stands for Win-Draw-Loss probabilities
                    raw_reward += (win - loss) * 0.05

            # If the agent is playing Black, invert the final sign
            if not self.model_plays_white:
                raw_reward = -raw_reward

            return raw_reward
        return 0
