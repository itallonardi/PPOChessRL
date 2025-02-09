import chess
import numpy as np
from gym import Env, spaces
from stockfish import Stockfish
import random
from app.callbacks.training_log_callback import approximate_rating_from_acpl


class ChessEnv(Env):
    """
    Custom Chess environment for reinforcement learning using python-chess and Stockfish.
    This environment includes:
      - A discrete action space representing possible chess moves (up to 218).
      - An observation space encoding the board state plus move history.
      - Integration with Stockfish for reward calculation.
      - Support for different opponents: self, stockfish, or both.
      - Maskable PPO support via action_masks().
      - Turn channel included in the historical states.
      - Reward now purely difference-based (plus a large bonus/penalty for checkmates).
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, stockfish_path=None, opponent='self', elo_range=(800, 2500), history_length=2, environment_id=0):
        """
        Initializes the Chess environment.

        :param stockfish_path: Path to the Stockfish engine executable.
        :param opponent: Type of opponent ('self', 'stockfish', or 'both').
        :param elo_range: Tuple for random ELO rating range of Stockfish.
        :param history_length: Number of previous board states (including turn channel) in the observation.
        :param environment_id: Identifier for this environment instance.
        """
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.opponent = opponent
        self.elo_range = elo_range

        # Guardamos apenas o path. Stockfish será criado no reset.
        self.stockfish_path = stockfish_path
        self.stockfish = None

        self.action_space = spaces.Discrete(218)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(8, 8, 13 + 13 * history_length),
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

        self.acpl_accumulator = 0.0
        self.acpl_steps = 0

        self.environment_id = environment_id

    # >>> MÉTODOS PARA EVITAR PICKLE DO STOCKFISH
    def __getstate__(self):
        """
        Before pickling (e.g. SubprocVecEnv fork), remove the Stockfish object.
        """
        state = self.__dict__.copy()
        if 'stockfish' in state:
            del state['stockfish']
        return state

    def __setstate__(self, state):
        """
        After unpickling, restore the dictionary. We set stockfish=None
        so it will be recreated on reset if needed.
        """
        self.__dict__.update(state)
        self.stockfish = None
    # <<<

    def step(self, action):
        """
        Executes a step for the agent ONLY. Does not automatically perform the opponent's move.
        ...
        """
        if self.stockfish:
            self.stockfish.set_fen_position(self.board.fen())
            previous_evaluation = self.stockfish.get_evaluation()
        else:
            previous_evaluation = None

        move = self.decode_action(action)
        if move and move in self.board.legal_moves:
            self.board.push(move)
            self.legal_moves += 1

            self.move_history.append(self.board_to_observation()[:, :, :13])
            if len(self.move_history) > self.history_length:
                self.move_history.pop(0)

            done = self.board.is_game_over()
            reward = self.calculate_reward(previous_evaluation)

            current_eval = None
            if self.stockfish:
                self.stockfish.set_fen_position(self.board.fen())
                current_eval = self.stockfish.get_evaluation()

            if previous_evaluation and current_eval:
                if previous_evaluation['type'] == 'cp' and current_eval['type'] == 'cp':
                    step_diff = abs(
                        current_eval['value'] - previous_evaluation['value'])
                    self.acpl_accumulator += step_diff
                    self.acpl_steps += 1

            info = {}
            if done:
                if self.acpl_steps > 0:
                    episode_acpl = self.acpl_accumulator / self.acpl_steps
                else:
                    episode_acpl = 0.0
                info["episode_acpl"] = episode_acpl

                approx_rating = approximate_rating_from_acpl(episode_acpl)
                print(
                    f"[Env {self.environment_id}] Game ended! ACPL={episode_acpl:.2f}, approx. rating={approx_rating}")

                self.acpl_accumulator = 0.0
                self.acpl_steps = 0

            return self.board_to_observation(), reward, done, info

        else:
            self.illegal_moves += 1
            return self.board_to_observation(), -1, False, {}

    def step_opponent(self):
        """
        Performs the opponent's move (Stockfish or otherwise), if applicable, in a separate step.
        ...
        """
        if self.board.is_game_over() or self.opponent == 'self':
            return self.board_to_observation(), 0, self.board.is_game_over(), {}

        if self.opponent == 'stockfish' or (self.opponent == 'both' and self.stockfish_is_opponent):
            if self.stockfish:
                self.stockfish.set_fen_position(self.board.fen())
                previous_evaluation = self.stockfish.get_evaluation()
            else:
                previous_evaluation = None

            best_move = None
            if self.stockfish and not self.board.is_game_over():
                best_move = self.stockfish.get_best_move()

            if best_move:
                self.board.push(chess.Move.from_uci(best_move))
                self.legal_moves += 1

                self.move_history.append(
                    self.board_to_observation()[:, :, :13])
                if len(self.move_history) > self.history_length:
                    self.move_history.pop(0)

                done = self.board.is_game_over()
                reward = self.calculate_reward(previous_evaluation)

                if self.stockfish and not done:
                    self.stockfish.set_fen_position(self.board.fen())
                    current_eval = self.stockfish.get_evaluation()

                    if previous_evaluation and current_eval:
                        if previous_evaluation['type'] == 'cp' and current_eval['type'] == 'cp':
                            step_diff = abs(
                                current_eval['value'] - previous_evaluation['value'])
                            self.acpl_accumulator += step_diff
                            self.acpl_steps += 1

                info = {}
                if done:
                    if self.acpl_steps > 0:
                        episode_acpl = self.acpl_accumulator / self.acpl_steps
                    else:
                        episode_acpl = 0.0
                    info["episode_acpl"] = episode_acpl

                    approx_rating = approximate_rating_from_acpl(episode_acpl)
                    print(
                        f"[Env {self.environment_id}] Game ended! ACPL={episode_acpl:.2f}, approx. rating={approx_rating}")

                    self.acpl_accumulator = 0.0
                    self.acpl_steps = 0

                return self.board_to_observation(), reward, done, info

        return self.board_to_observation(), 0, self.board.is_game_over(), {}

    def board_to_observation(self):
        """
        Builds the observation tensor with shape (8, 8, 13 + 13 * self.history_length):
        ...
        """
        observation = np.zeros(
            (8, 8, 13 + 13 * self.history_length), dtype=np.uint8)

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                layer_index = (piece.piece_type - 1) * 2 + \
                    (0 if piece.color == chess.WHITE else 1)
                rank, file = divmod(square, 8)
                observation[rank, file, layer_index] = 1

        if self.board.turn == chess.WHITE:
            observation[:, :, 12] = 1

        for i, hist in enumerate(self.move_history):
            start = 13 + 13 * i
            end = 13 + 13 * (i + 1)
            observation[:, :, start:end] = hist

        return observation

    def reset(self):
        """
        Resets the environment to the initial state:
          - Clears move history.
          - Resets counters for illegal and legal moves, checks, and checkmates.
          - Randomizes whether the model is white or black if opponent is Stockfish or both.
          - Makes Stockfish's first move if Stockfish is white (optional).
        """
        self.board.reset()
        self.move_history = []
        self.illegal_moves = 0
        self.legal_moves = 0
        self.checks = 0
        self.checkmates = 0

        self.acpl_accumulator = 0.0
        self.acpl_steps = 0

        # Se self.stockfish for None, criamos agora
        if self.stockfish is None and self.stockfish_path:
            self.stockfish = Stockfish(self.stockfish_path)

        if self.opponent in ['stockfish', 'both'] and self.stockfish:
            self.stockfish.set_elo_rating(random.randint(*self.elo_range))
            self.model_plays_white = random.choice([True, False])
            if self.opponent == 'both':
                self.stockfish_is_opponent = random.choice([True, False])
            else:
                self.stockfish_is_opponent = True

            if not self.model_plays_white and self.stockfish_is_opponent:
                self.stockfish.set_fen_position(self.board.fen())
                first_move = self.stockfish.get_best_move()
                if first_move:
                    self.board.push(chess.Move.from_uci(first_move))
                    self.legal_moves += 1
        else:
            self.model_plays_white = True
            self.stockfish_is_opponent = False

        print(f"[Env {self.environment_id}] Starting new game...")
        return self.board_to_observation()

    def decode_action(self, action):
        """
        Decodes the action index into an actual chess Move:
         - If action >= number of legal moves, returns None (illegal).
         - Otherwise, returns the corresponding move from the current list of legal_moves.
        """
        legal_moves = list(self.board.legal_moves)
        if 0 <= action < len(legal_moves):
            return legal_moves[action]
        return None

    def render(self, mode='human'):
        """
        Renders the current board state in a human-readable format (ASCII).
        """
        if mode == 'human':
            print("Current board state:")
            print(self.board)

    def calculate_reward(self, previous_evaluation=None):
        """
        Calculates the reward based on Stockfish's evaluations:
          - If there is a checkmate (type 'mate'), returns +/-100000 depending on
            which side is winning, then applies perspective for the agent's color.
          - Otherwise, returns the difference in centipawns between current and previous
            evaluation (also adjusted for agent's color if it is black).
          - No extra reward from top Stockfish moves is added (purely difference-based).
        """
        if self.stockfish:
            self.stockfish._set_option("MultiPV", 3)
            self.stockfish._set_option("UCI_ShowWDL", True)
            self.stockfish.set_fen_position(self.board.fen())
            current_evaluation = self.stockfish.get_evaluation()

            if current_evaluation['type'] == 'mate':
                raw_reward = 100000 if current_evaluation['value'] > 0 else -100000
                return raw_reward if self.model_plays_white else -raw_reward

            raw_reward = 0
            if (previous_evaluation and previous_evaluation['type'] == 'cp'
               and current_evaluation['type'] == 'cp'):
                raw_reward = (
                    current_evaluation['value'] - previous_evaluation['value'])

            if not self.model_plays_white:
                raw_reward = -raw_reward

            return raw_reward
        return 0

    def action_masks(self):
        """
        Returns a boolean mask of shape (218,) indicating which discrete actions
        are legal in the current position. True = legal, False = illegal.
        This method is used by MaskablePPO to avoid selecting illegal moves.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        legal_moves = list(self.board.legal_moves)
        for i in range(len(legal_moves)):
            mask[i] = True
        return mask
