from node import Node
import copy
import random


class Player:

    def __init__(self, number_of_counters, number_of_pits, player_name, is_ai=False, iters=100):
        self.number_of_pits = number_of_pits
        self.number_of_counters = number_of_counters
        self.board = [0] + [self.number_of_counters] * self.number_of_pits
        self.player_name = player_name
        self.is_ai = is_ai
        self.iters = iters
        self.opponent = None
        self.root = Node()

    def set_opponent(self, opponent):
        self.opponent = opponent

    def update_board(self, new_board):
        self.board = new_board

    def get_pit_choice(self, current_board):
        chosen_pits = random.randint(1, self.number_of_pits)
        print("choose move btn (1-6):", chosen_pits)
        return chosen_pits

    def make_move(self, current_board, chosen_pits=None):

        counters = current_board[0][chosen_pits]
        if chosen_pits == 0 or counters == 0:
            return None, None, None, None

        row = 0
        current_pit = chosen_pits
        boards = copy.deepcopy(current_board)
        boards[0][chosen_pits] = 0

        while counters > 0:
            current_pit -= 1
            if row == 0 and current_pit < 0:
                row = 1
                current_pit = self.number_of_pits
            elif row == 1 and current_pit == 0:
                row = 0
                current_pit = self.number_of_pits

            boards[row][current_pit] += 1
            counters -= 1

        special_state = 0
        captured = 0
        opposite_position = self.number_of_pits + 1 - current_pit

        # check capturing move
        if row == 0 and current_pit != 0 and boards[0][current_pit] == 1 and boards[1][opposite_position] != 0:
            captured = boards[1][opposite_position]
            boards[0][0] += (boards[0][current_pit] + boards[1][opposite_position])
            boards[0][current_pit] = 0
            boards[1][opposite_position] = 0

        # checking if game is over
        for i in range(2):
            if sum(boards[i][1:]) == 0:
                boards[1 - i] = [sum(boards[1 - i])] + [0] * self.number_of_pits
                if boards[0][0] > boards[1][0]:
                    special_state = 1
                elif boards[0][0] < boards[1][0]:
                    special_state = -1
                elif i == 0:
                    special_state = -1
                else:
                    special_state = 1
                break

        # checking for free turn
        if special_state == 0 and current_pit == 0:
            special_state = 2

        own_new_board, opponent_new_board = boards[0], boards[1]
        return own_new_board, opponent_new_board, special_state, captured