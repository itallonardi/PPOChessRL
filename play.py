import tkinter as tk
from PIL import Image, ImageTk
import chess
import chess.svg
import numpy as np
from stable_baselines3 import PPO
import cairosvg
import io
from app.envs.chess_env import ChessEnv


class InteractiveChessEnv(ChessEnv):
    def __init__(self, player_color='white'):
        super().__init__()
        self.player_color = chess.WHITE if player_color == 'white' else chess.BLACK
        self.ai_color = not self.player_color
        self.selected_piece_square = None
        self.pending_promotion_move = None

    def click_square(self, event):
        col_size = row_size = 50
        col = event.x // col_size
        row = 7 - (event.y // row_size)
        square = chess.square(col, row)
        piece = self.board.piece_at(square)
        if piece and piece.color == self.player_color:
            self.selected_piece_square = square
        elif self.selected_piece_square:
            from_square = self.selected_piece_square
            to_square = square
            promoting_piece = self.board.piece_at(from_square)
            if promoting_piece.piece_type == chess.PAWN and (to_square // 8 in [0, 7]):
                self.pending_promotion_move = chess.Move(
                    from_square, to_square)
                self.promotion_dialog()
            else:
                move = chess.Move(from_square, to_square)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.render('human')
                    self.after_move()
            self.selected_piece_square = None

    def promotion_dialog(self):
        promotion_types = {
            'Queen': chess.QUEEN,
            'Rook': chess.ROOK,
            'Bishop': chess.BISHOP,
            'Knight': chess.KNIGHT
        }
        popup = tk.Toplevel(self.root)
        popup.title("Choose piece for promotion")
        for piece, type in promotion_types.items():
            button = tk.Button(
                popup, text=piece,
                command=lambda t=type, p=popup: self.choose_promotion(t, p)
            )
            button.pack()

    def choose_promotion(self, piece_type, popup):
        if self.pending_promotion_move:
            promo_move = chess.Move(
                self.pending_promotion_move.from_square,
                self.pending_promotion_move.to_square,
                promotion=piece_type
            )
            self.board.push(promo_move)
            self.render('human')
            self.after_move()
            self.pending_promotion_move = None
        popup.destroy()

    def after_move(self):
        if not self.board.is_game_over():
            action, _states = self.model.predict(self.board_to_observation())
            self.step(action)
            self.render('human')

    def render(self, mode='human'):
        board_svg = chess.svg.board(self.board, size=400)
        png_data = cairosvg.svg2png(bytestring=board_svg.encode('utf-8'))
        image = Image.open(io.BytesIO(png_data))

        if not hasattr(self, 'root'):
            self.root = tk.Tk()
            self.canvas = tk.Canvas(self.root, width=400, height=400)
            self.canvas.pack()
            self.canvas.bind("<Button-1>", self.click_square)
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)
        self.root.update()


def main():
    env = InteractiveChessEnv(player_color='white')
    env.model = PPO.load("models/chess_model.zip", env=env)
    env.render('human')

    env.root.mainloop()


if __name__ == "__main__":
    main()
