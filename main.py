from Kalahboard import Kalah
from alphabeta import Player
from mcts import Player as mPlayer
from reinforce import Player as rlPlayer
from PPO2 import BasicEnv as custom_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
import random
import pandas as pd
import time
import multiprocessing
import warnings
import os

warnings.filterwarnings("ignore")


# Test games
def test_games():
    all_data = []
    for h1 in range(1, 7):
        for h2 in range(1, 7):
            p1_wins = 0
            p2_wins = 0
            print("Heuristic ", h1, "Heuristic", h2)
            for i in range(1):
                depth = 20
                print("depth is:", depth)
                print("Game no.", i + 1)

                player_1 = Player(6, 6, "Player1", True, h1, depth)
                player_2 = Player(6, 6, "Player2", False, h2, depth)
                kalah = Kalah(player_1, player_2)
                winner, score1, score2 = kalah.game_loop()
                if winner == 1:
                    p1_wins += 1
                else:
                    p2_wins += 1
            all_data.append(
                {
                    "Player1 Heuristic": "H" + str(h1),
                    "Player2 Heuristic": "H" + str(h2),
                    "Player1 wins": p1_wins,
                    "Player2 wins": p2_wins,
                }
            )
    results = pd.DataFrame(all_data)
    results.to_csv("./test2.csv", index=False)
    return


class Results:
    def __init__(self):
        self.gamedata = []
        self.RESULTS_CSV_NAME = None

    def main_loop(self, players, num_games, seeds, pits, algo_c, choice, iters=None, heuristic=None, depth=None):
        for i in range(num_games):
            if algo_c == 1:
                # Store AlphaBeta pruning results
                self.RESULTS_CSV_NAME = "GameResultsAlphaBeta.csv"

                if choice == 3:
                    test_games()
                    exit()
                if choice == 2:

                    heuristic1 = heuristic
                    depth1 = depth
                    player_1 = Player(seeds, pits, "Player2", True, heuristic1, depth1)

                else:
                    player_random = random.randint(1, 7)
                    player_1 = Player(seeds, pits, player_random, False)

                if choice == 1:

                    heuristic2 = heuristic
                    depth2 = depth
                    player_2 = Player(seeds, pits, "Player2", True, heuristic2, depth2)
                else:
                    player_random = random.randint(1, 7)
                    player_2 = Player(seeds, pits, player_random, False)

                kalah = Kalah(player_1, player_2)
                winner, score1, score2, player1_count, player2_count, total_time = kalah.game_loop()
                self.gamedata.append({
                    'Game': i + 1,
                    'Winner': players[winner - 1],
                    'Player1 Score': score1,
                    'Player2 Score': score2,
                    'Player1 count': player1_count,
                    'Player2 count': player2_count,
                    'Total_time': total_time

                })


            elif algo_c == 2:
                # Store MCTS results
                self.RESULTS_CSV_NAME = "GameResultsMCTS.csv"
                wins = 0

                player_1 = mPlayer(seeds, pits, "Player1", False, iters)
                player_2 = mPlayer(seeds, pits, "Player2", True, iters)
                kalah = Kalah(player_1, player_2)
                winner, score1, score2, player1_count, player2_count, total_time = kalah.game_loop()
                self.gamedata.append({
                    'Game': i + 1,
                    'Winner': players[winner - 1],
                    'Player1 Score': score1,
                    'Player2 Score': score2,
                    'Player1 count': player1_count,
                    'Player2 count': player2_count,
                    'Total_time': total_time
                })


            elif algo_c == 3:
                # Store PPO2 results
                self.RESULTS_CSV_NAME = "GameResultsPPO2.csv"
                wins = 0
                env = custom_env(pits)
                model = PPO2(MlpPolicy, env, verbose=1)
                model.learn(total_timesteps=1000000)

                obs = env.reset()
                # for i in range(num_games):
                player_1 = rlPlayer(seeds, pits, "Player1", True, iters)
                player_2 = rlPlayer(seeds, pits, "Player2", False, iters)
                kalah = Kalah(player_1, player_2)
                winner, score1, score2, player1_count, player2_count, total_time = kalah.game_loop2()
                self.gamedata.append({
                    'Game': i + 1,
                    'Winner': players[winner - 1],
                    'Player1 Score': score1,
                    'Player2 Score': score2,
                    'Player1 count': player1_count,
                    'Player2 count': player2_count,
                    'Total_time': total_time
                })
        results = pd.DataFrame(self.gamedata)
        results.to_csv(self.RESULTS_CSV_NAME, index=False)
        os.kill(os.getppid(), 15)
        # return False


if __name__ == '__main__':
    players = ['Player1', 'Player2']
    num_games = int(input("Enter number of games: "))

    seeds = int(input("Enter number of seeds: "))
    pits = int(input("Enter number of pits: "))

    print('Choose Algorithm: \n1. Alpha Beta pruning \n 2. Monte Carlo Tree Search \n 3. PPO2')
    algo_c = int(input('Enter Choice: '))

    print("Choose game:\n1. player 2 vs player 1\n2. player 1 vs player 2\n3. test games\n")
    choice = int(input("Enter choice: "))
    if algo_c == 1:
        iters = None
        if choice == 2:
            heuristic = int(input("Choose heuristic number for player 1: "))
            depth = int(input("Choose depth for player 1: "))

        elif choice == 1:
            heuristic = int(input("Choose heuristic number for player 2: "))
            depth = int(input("Choose depth for player 2: "))
    elif algo_c == 2:
        iters = int(input("Enter Iterations for MCTS: "))
        heuristic = None
        depth = None
    elif algo_c == 3:
        iters = int(input('Enter iterations for training of PPO2 Agent: '))
        heuristic = None
        depth = None
    else:
        raise Exception('Wrong input')
    start = time.time()

    r = Results()
    r.p = multiprocessing.Process(target=r.main_loop, name="Main Loop",
                                  args=(players, num_games, seeds, pits, algo_c, choice, iters, heuristic, depth))
    r.p.start()

    results = pd.DataFrame(r.gamedata)
    results.to_csv(r.RESULTS_CSV_NAME, index=False)
