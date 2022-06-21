class Node:

    def __init__(self, chosen_pits=None, visits=0, score=0):
        self.best_child = None
        self.chosen_pits = chosen_pits
        self.children = []
        self.child_pointers = []
        self.visited = {}
        self.visits = 0
        self.score = score
