import time
import logging
from PPO2 import BasicEnv as custom_env
from stable_baselines import PPO2
from stable_baselines.common.cmd_util import make_vec_env

class Kalah:

    def __init__(self, player_1, player_2):
        self.player_1 = player_1
        self.player_2 = player_2
        self.player_1.set_opponent(self.player_2)
        self.player_2.set_opponent(self.player_1)
        self.round = 0

    def game_loop(self):
        start_time = time.time()
        self.print_game()
        while True:

            if self.round % 2 == 0:
                game_state = self.make_move(self.player_1, self.player_2)
            else:
                game_state = self.make_move(self.player_2, self.player_1)

            if game_state is None:  # invalid move
                print("Invalid Move.")
                # raise Exception('Invalid Move')
                continue

            self.print_game()
            if (game_state == 1 and self.round % 2 == 0) or (game_state == -1 and self.round % 2 == 1):  # win for player1
                print(self.player_1.player_name, "Wins!")
                res = 1
                break

            elif (game_state == -1 and self.round % 2 == 0) or (game_state == 1 and self.round % 2 == 1):  # win for player2
                print(self.player_2.player_name, "Wins!")
                res = 2
                break
            elif game_state == 0:  # normal game
                self.round += 1
            elif game_state == 2:  # free turn for current player
                print("Free turn.")

        end_time = time.time()

        print("Total time taken =", end_time - start_time)
        return res, self.player_1.board[0], self.player_2.board[0]

    def make_move(self, current_player, opponent_player):
        print(str(current_player.player_name) + "'s move.")

        chosen_pit = current_player.get_pit_choice([current_player.board, opponent_player.board])

        current_player_board, opponent_player_board, special_state, _ = \
            current_player.make_move([current_player.board, opponent_player.board], chosen_pit)

        if current_player_board is None:
            return None

        current_player.update_board(current_player_board)
        opponent_player.update_board(opponent_player_board)

        return special_state

    def print_game(self):

        # pit numbers
        print('Player 2:', end='')
        for i in range(1, self.player_1.number_of_pits + 1):
            print('\t', "(" + str(i) + ")", end=' \t')
        print("\n")

        # player 2 pits
        print('\t', " ", end=' \t|')
        for x in self.player_2.board[1:]:
            print('\t', x, end=' \t\t|')
        print(' \t\t', " ", end=' \t')
        print("\n")

        # stores for the players
        print(" ", self.player_2.board[0], end=' \t')
        for i in range(self.player_2.number_of_pits):
            print(' \t', " ", end=' \t\t')
        print('\t', self.player_1.board[0], end=' \t')
        print("\n")

        # player1 pits
        print('\t', " ", end=' \t|')
        for x in self.player_1.board[::-1][:-1]:
            print('\t', x, end=' \t\t|')
        print(' \t\t', " ", end=' \t')
        print("\n")

        # player1 pits numbers
        print("Player 1:", end='')
        for i in range(self.player_1.number_of_pits, 0, -1):
            print('\t', "(" + str(i) + ")", end=' \t')
        print("\n")

        #return self.print_game()

    def game_loop2(self, iters = 1000):
        start_time = time.time()
        self.print_game()
        from stable_baselines import PPO2
        from stable_baselines.common.policies import MlpPolicy
        env = custom_env(self.player_1.number_of_pits)

        model = PPO2(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=iters)
        while True:

            if self.round % 2 == 0:
                game_state = self.make_move(self.player_1, self.player_2)
            else:
                game_state = self.make_move(self.player_2, self.player_1)

            if game_state is None:  # invalid move
                print("Invalid Move.")
                continue

            action, states = model.predict(game_state)
            self.print_game()
            if (game_state == 1 and self.round % 2 == 0) or (
                    game_state == -1 and self.round % 2 == 1):  # win for player1
                obs, reward, done, info = env.step(action=True)
                print(self.player_1.player_name, "Wins!")
                res = 1
                break

            elif (game_state == -1 and self.round % 2 == 0) or (
                    game_state == 1 and self.round % 2 == 1):  # win for player2
                print(self.player_2.player_name, "Wins!")
                obs, reward, done, info = env.step(action=False)
                res = 2
                break
            elif game_state == 0:  # normal game
                self.round += 1
            elif game_state == 2:  # free turn for current player
                print("Free turn.")

        end_time = time.time()

        print("Total time taken =", end_time - start_time)
        return res, self.player_1.board[0], self.player_2.board[0]
