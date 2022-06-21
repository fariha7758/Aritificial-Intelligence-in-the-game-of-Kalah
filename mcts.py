from node import Node
import copy
import random
import math


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
        if self.is_ai:
            chosen_pits = self.choose_best_move()
        else:
            cur_pit_counter = 0
            while cur_pit_counter == 0:
                # print(cur_pit_counter, 'choosing')
                chosen_pits = random.randint(1, self.number_of_pits)
                cur_pit_counter = current_board[0][chosen_pits]
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

    def expansion(self, current_node, boards):
        if sum(boards[1:]) != 0:
            for i in range(1, self.number_of_pits + 1):
                if boards[i] != 0:
                    new_node = Node(i)
                    current_node.children.append(new_node)
                    current_node.child_pointers.append(i)
        return current_node.children

    def UCB(self, current_node):
        max = {}
        node = current_node.children[0]
        if node.visits != 0:
            maxval = float(node.score + 2 * (math.sqrt(math.log1p(self.root.visits) / node.visits)))
        else:
            maxval = float('inf')

        max[maxval] = node

        for node in current_node.children:
            if node.visits != 0:
                temp = float(node.score + 2 * (math.sqrt(math.log1p(self.root.visits) / node.visits)))
            else:
                temp = float('inf')

            if temp > maxval:
                max[temp] = node
                maxval = temp

        return max[maxval]

    def move_opp(self, current_node, boards):
        if sum(boards[1:]) >0:
            score = 0
            options = list(range(1, self.number_of_pits + 1))
            random.shuffle(options)
            for i in options:
                if boards[1][i] != 0:
                    b0, b1, special_state, captured = self.make_move(boards, i)
                    break

            new_node = Node(i)
            if special_state == 0:
                self.expansion(new_node, copy.deepcopy([b0, b1]))
                score += self.selection(new_node, copy.deepcopy([b0, b1]))

            elif special_state == 2:
                score += self.move_opp(new_node, copy.deepcopy([b1, b0])) * -1

            elif special_state == 1:
                score = 1

            elif special_state == -1:
                score = -1

        return score

    def selection(self, current_node, boards):
        score = 0
        new_node = self.UCB(current_node)
        i = current_node.children.index(new_node)
        b0, b1, special_state, captured = self.make_move(boards, i)

        if special_state == 2:
            self.expansion(new_node, copy.deepcopy([b0, b1]))
            score += self.selection(new_node, copy.deepcopy([b0, b1]))

        elif special_state == 0:
            score += self.move_opp(new_node, copy.deepcopy([b1, b0])) * -1

        elif special_state == 1:
            score = 1

        elif special_state == -1:
            score = -1
        return score

    def choose_best_move(self):
        self.root = Node()
        root_node = self.root
        self.expansion(root_node, self.board)
        finalsc = float('-inf')
        if len(root_node.children) == 1:
            root_node.best_child = root_node.child_pointers[0]
        else:
            for i in range(len(root_node.children)):
                child = root_node.children[i]
                self.expansion(child, self.board)
                score = self.selection(child, [self.board, self.opponent.board])
                if (score > finalsc):
                    finalsc = score
                    root_node.best_child = root_node.child_pointers[i]
        return root_node.best_child