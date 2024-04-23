# PPOChessRL

PPOChessRL is a project that combines reinforcement learning techniques with the powerful Stockfish chess engine to train artificial intelligence models capable of playing high-level chess.

## Overview

The project utilizes the `stable-baselines3` library and Stockfish engine to create an environment where a reinforcement learning model can learn to play chess. The model is trained to make decisions based on the state of the chessboard, aiming to maximize its effectiveness in games against Stockfish or in simulations against itself.

## Features

- **Integration with Stockfish**: Uses Stockfish to evaluate and guide moves during training, providing a high-quality benchmark for the model's learning.
- **Optimized Action Space**: Limits the action space to legal moves in each game state, improving the efficiency of training.
- **Custom Environment**: Implements a custom environment using the OpenAI `gym` library, suitable for training reinforcement learning models in board game problems.

## How to Use

### Prerequisites

- Python 3.8+
- Libraries: `numpy`, `gym`, `stable-baselines3`, `python-chess`, `stockfish`

### Installation

Clone the repository and navigate into its folder:
```bash
git clone https://github.com/itallonardi/PPOChessRL
cd ChessRL
```

Install the dependencies in your Python environment::
```bash
pip install -r requirements.txt
```

### Configuration
To start training the chess model, you can customize the process through various command line arguments that control the training behavior. Below are details on how to use each argument:

#### Configurando stockfish:
- The Stockfish chess engine is an essential requirement as the reward system uses it to determine the eventual rewards and penalties. Download the appropriate version of Stockfish from https://stockfishchess.org/download/, and after giving all permissions and running it, you can configure a .env file at the root of your project pointing to the path, for example:
```env
# Stockfish Path
STOCKFISH_PATH=stockfish/stockfish-ubuntu-x86-64-avx2
```
An .env.example file is available at the root of the project as an example. If you choose not to set up a .env file, the system will assume **stockfish/stockfish-ubuntu-x86-64-avx2** (inside the project folder) as the default.

- You can test the Stockfish setup by running at the project root:
```bash
python3 stockfish_validation.py
```

This should produce output similar to a representation of the pawn moving from e2 to e4, accompanied by its evaluation:
```bash
+---+---+---+---+---+---+---+---+
| r | n | b | q | k | b | n | r | 8
+---+---+---+---+---+---+---+---+
| p | p | p | p | p | p | p | p | 7
+---+---+---+---+---+---+---+---+
|   |   |   |   |   |   |   |   | 6
+---+---+---+---+---+---+---+---+
|   |   |   |   |   |   |   |   | 5
+---+---+---+---+---+---+---+---+
|   |   |   |   | P |   |   |   | 4
+---+---+---+---+---+---+---+---+
|   |   |   |   |   |   |   |   | 3
+---+---+---+---+---+---+---+---+
| P | P | P | P |   | P | P | P | 2
+---+---+---+---+---+---+---+---+
| R | N | B | Q | K | B | N | R | 1
+---+---+---+---+---+---+---+---+
  a   b   c   d   e   f   g   h

Evaluation after 1.e4: {'type': 'cp', 'value': 39}
```


### Training

Several options are available to customize the training.
#### Available Arguments:

- **--opponent**: Specifies against whom the model will play during training. Can be **self** to play against itself, **stockfish** to play against the Stockfish chess engine, or **both** to alternate between playing against itself and Stockfish. The default is self.

    **Example**: ```bash --opponent both```

- **--elo_start**: Sets the starting ELO rating of Stockfish. This adjusts the difficulty of the Stockfish opponent at the start of training. The default is **800**.

    **Example**: ```bash --elo_start 1200```

- **--elo_end**: Sets the ending ELO rating of Stockfish. As the model learns, the difficulty of Stockfish can be increased up to this limit. The default is **2500**.

    **Example**: ```bash --elo_end 2000```

- --total_timesteps: Specifies the total number of timesteps for the training. This number defines the duration of the learning process. The default is 524288.
    
    **Example**: ```bash --total_timesteps 1000000```

- **--save_freq**: Defines the frequency, in terms of number of steps, at which the model is automatically saved during training. This is useful for maintaining checkpoints and assessing progress. The default is **22118**.

    **Example**: ```bash --save_freq 50000```

#### Starting Training
To start training with the default settings, run the following command in the terminal:
```bash
python main.py
```
**(note: the default settings use only "self" as the opponent)**

To customize the training, you can add arguments as needed. For example, to train the model against Stockfish varying the ELO from 1200 to 2000, saving every 50000 timesteps, and running for a total of 1 million timesteps, you would use:

```
python main.py --opponent stockfish --elo_start 1200 --elo_end 2000 --total_timesteps 1000000 --save_freq 50000
```

### Testing
There is a file called **play.py** in the root of the project. To test playing against the trained model, just execute the following command in the terminal to play:

```bash
python3 play.py
```
**(will not work without prior training)**

## Notes
### Performance
The performance of the model can vary significantly depending on the training conducted. By default, the project will attempt to load the model to continue training whenever you run training after a previous training session.

Note that this project is for educational purposes and a high computational and time cost would be required to make the model highly competitive at a high level.

## Contributions
Contributions are always welcome! Feel free to clone, fork, or send pull requests to the project.

## License
Distributed under the MIT License.

## Authors
Itallo Nardi – @itallonardi

## Acknowledgements
- chess
- gym
- gymnasium
- matplotlib
- numpy
- pandas
- python-chess
- stable_baselines3
- stockfish
- tensorboard
- torch
