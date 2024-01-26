import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import trange

PAWNS = 9
SIZE = 24


class JDM:
    def __init__(self, policy_1, policy_2, max_iter=500, train = True):
        self.train = train
        self.max_iter = max_iter
        self.pawn_1 = 0
        self.pawn_2 = 0
        self.policy_1 = lambda x: policy_1(x, -1)
        self.policy_2 = lambda x: policy_2(x, 1)
        self.history = []
        self.starter = 1
        self.playing = self.starter
        self.before_moving = True
        self.pawn_placed = 0

        # The 24 elements on board and the last three represent the action (just to feed it directly to neural network)
        self.board = np.zeros(shape=SIZE+3, dtype=int)
        self.neighbors = [
            [1,9],
            [0,2,4],
            [1,14],
            [4,10],
            [1,3,5,7],
            [4,13],
            [7,11],
            [4,6,8],
            [7,12],
            [0,10,21],
            [3,9,11,18],
            [6,10,15],
            [8,13,17],
            [5,12,14,20],
            [2,13,23],
            [11,16],
            [15,17,19],
            [12,16],
            [10,19],
            [16,18,20,22],
            [13,19],
            [9,22],
            [19,21,23],
            [14,22]
        ]
        self.alignments = [
            (
                (1,2),
                (9,21),
            ),
            (
                (0,2),
                (4,7),
            ),
            (
                (1,0),
                (23,14),
            ),
            (
                (4,5),
                (10,18),
            ),
            (
                (3,5),
                (1,7),
            ),
            (
                (3,4),
                (13,20),
            ),
            (
                (11,15),
                (7,8),
            ),
            (
                (6,8),
                (1,4),
            ),
            (
                (7,6),
                (12,17),
            ),
            (
                (0,21),
                (10,11),
            ),
            (
                (9,11),
                (3,18),
            ),
            (
                (9,10),
                (6,15),
            ),
            (
                (8,17),
                (13,14),
            ),
            (
                (12,14),
                (5,20),
            ),
            (
                (12,13),
                (2,23),
            ),
            (
                (11,6),
                (16,17),
            ),
            (
                (15,17),
                (19,22),
            ),
            (
                (8,12),
                (15,16),
            ),
            (
                (10,3),
                (19,20),
            ),
            (
                (16,22),
                (18,20),
            ),
            (
                (18,19),
                (5,13),
            ),
            (
                (0,9),
                (23,22),
            ),
            (
                (21,23),
                (19,16),
            ),
            (
                (22,21),
                (14,2),
            ),
        ]

    def is_placing(self):
        return self.board[-3] == 1

    def is_removing(self):
        return self.board[-2] == 1
    
    def is_moving(self):
        return self.board[-1] == 1
    
    def getBoardSize(self):
        return len(self.board)

    def getActionSize(self):
        return len(self.board)-3 # board[-1 -> -3] indicate the state of the board

    def new_aligned(self, i):
        player = self.board[i]
        ali1, ali2 = self.alignments[i]
        return (self.board[ali1[0]] == self.board[ali1[1]] and self.board[ali1[1]] == player) or (self.board[ali2[0]] == self.board[ali2[1]] and self.board[ali2[1]] == player)

    def is_open(self, point):
        if self.board[point] == 0: return False
        for neigh in self.neighbors[point]:
            if self.board[neigh] == 0:
                return True
        return False

    def is_finished(self):
        if self.before_moving: return 0
        blocked_1 = True
        blocked_2 = True
        for i in range(SIZE):
            if self.is_open(i):
                if self.board[i] == -1:
                    blocked_1 = False
                else:
                    blocked_2 = False

        if blocked_1 or self.pawn_1 <= 2: return 1
        if blocked_2 or self.pawn_2 <= 2: return -1
        return 0

    def assign(self, player, pos):
        if self.board[pos] == 0:
            self.board[pos] = player
            if player == -1:
                self.pawn_1 += 1
            else: self.pawn_2 += 1
            return True
        else:
            return False

    def remove(self, player, pos):
        if self.board[pos] == player:
            self.board[pos] = 0
            if player == -1:
                self.pawn_1 -= 1
            else: self.pawn_2 -= 1
            return True
        return False

    def move(self, from_pos, to_pos, player):
        if from_pos == 25: return False # can't play
        if to_pos not in self.neighbors[from_pos]: return False
        if self.board[from_pos] != player: return False
        if self.board[to_pos] != 0: return False
        self.board[from_pos] = 0
        self.board[to_pos] = player
        return True

    def restart(self):
        self.board.fill(0)
        self.pawn_1 = 0
        self.pawn_2 = 0
        self.history = []
        self.state_place()

    def state_place(self):
        self.board[-3] = 1
        self.board[-2] = 0
        self.board[-1] = 0
    def state_remove(self):
        self.board[-3] = 0
        self.board[-2] = 1
        self.board[-1] = 0
    def state_move(self):
        self.board[-3] = 0
        self.board[-2] = 0
        self.board[-1] = 1

    def legal_actions(self, player):
        state = 2*self.board[-1] + self.board[-2]
        board = self.board[:-3]
        if state == 0: # choose a place to place a pawn
            return [(x, 25) for x in (board==0).nonzero()[0]] # 25=flag fail
        elif state == 1: # remove a pawn from a place
            return np.random.choice((board==-player).nonzero()[0]),25
        elif state == 2: # move a pawn from a place to another
            possible = (board==player).nonzero()[0]         
            return [(x, y) for x in possible for y in self.neighbors[x] if board[y] == 0]
        return []

    def play(self, action1, action2=25):
        reward = 0
        if self.board[-3]:
            assert action2==25, "Placing not moving"
            if self.assign(self.playing, action1):
                if self.playing != self.starter:
                    self.pawn_placed += 1
                    if self.pawn_placed == 9:
                        self.before_moving = False
                if self.new_aligned(action1):
                    self.state_remove()
                    reward = +1
                else:
                    if not self.before_moving:
                        self.state_move()
                    self.playing *= -1
            else:
                reward = -1

        elif self.board[-2]:
            assert action2==25, "Removing not moving"
            self.remove(-self.playing, action1)
            if self.before_moving:
                self.state_place()
            else:
                self.state_move()
            self.playing *= -1

        elif self.board[-1]:
            if self.move(action1, action2, self.playing):
                if self.new_aligned(action2):
                    self.state_remove()
                    reward = +1
                else:
                    self.playing *= -1
            else:
                reward = -1
        else:
            assert False, "Not supposed to happen"
        return reward

    def run(self, save_history = False, only_start=False):
        iter = 0
        # place the pawns
        for _ in range(PAWNS):
            self.state_place()
            choice, useless = self.policy_1(self)
            if save_history:
                self.history.append((self.board.copy(), (choice, useless)))
            if self.assign(-1, choice):
                if self.new_aligned(choice):
                    self.state_remove()
                    remove, useless = self.policy_1(self)
                    if save_history:
                        self.history.append((self.board.copy(), (remove, useless)))
                    self.remove(1, remove)

            self.state_place()
            choice, useless = self.policy_2(self)
            if save_history:
                self.history.append((self.board.copy(), (choice, useless)))
            if self.assign(1, choice):
                if self.new_aligned(choice):
                    self.state_remove()
                    remove, useless = self.policy_2(self)
                    if save_history:
                        self.history.append((self.board.copy(), (remove, useless)))
                    self.remove(-1, remove)

        # move the pawns
        if only_start:
            save_history = False
        self.before_moving = False
        while not self.is_finished() and iter < self.max_iter:
            self.state_move()
            from_pos, to_pos = self.policy_1(self)
            if save_history:
                self.history.append((self.board.copy(), (from_pos, to_pos)))
            if self.move(from_pos, to_pos, -1):
                if self.new_aligned(to_pos):
                    self.state_remove()
                    remove, useless = self.policy_1(self)
                    if save_history:
                        self.history.append((self.board.copy(), (remove, useless)))
                    self.remove(1, remove)

            self.state_move()
            from_pos, to_pos = self.policy_2(self)
            if save_history:
                self.history.append((self.board.copy(), (from_pos, to_pos)))
            if self.move(from_pos, to_pos, 1):
                if self.new_aligned(to_pos):
                    self.state_remove()
                    remove, useless = self.policy_2(self)
                    if save_history:
                        self.history.append((self.board.copy(), (remove, useless)))
                    self.remove(-1, remove)
            iter+=1

        return self.is_finished()


    def batch_example(self, batch_size=64):
        games_history = []
        for i in trange(batch_size):
            winner = self.run(True)
            games_history.extend(list(map(lambda x: (x[0], x[1], winner), self.history)))
            self.restart()
        return games_history

    def batch_reward(self, batch_size=64):
        games_history = []
        games_rewards = []
        for _ in range(batch_size):
            self.restart()
            iter = 0
            while not self.is_finished() and iter<self.max_iter:
                if self.playing < 0:
                    action = self.policy_1(self)
                else:
                    action = self.policy_2(self)
                history = self.board.copy()
                temp = [self.playing] + [action[0]] + (action[1]!=25)*[action[1]]
                history = np.append(history, temp)
                games_history.append(history)
                reward = self.play(*action)
                games_rewards.append(reward)
        winner = self.is_finished()
        games_rewards = list(map(lambda x: x[1]+3 if x[0][SIZE+3]==winner else x[0]-3, zip(games_history, games_rewards)))
        print(games_history)
        return games_history, games_rewards

    def generate_start(self, batch_size=64):
        games_history = []
        for i in trange(batch_size):
            winner = self.run(True, only_start=True)
            games_history.extend(list(map(lambda x: (x[0], x[1], winner), self.history)))
            self.restart()
        return games_history

    def idx_to_point(idx):
        if idx == 0: return (0,0)
        if idx == 1: return (0,3)
        if idx == 2:  return (0,6)
        if idx == 3:  return (1,1)
        if idx == 4: return (1,3)
        if idx == 5: return (1,5)
        if idx == 6: return (2,2)
        if idx == 7: return (2,3)
        if idx == 8: return (2,4)
        if idx == 9: return (3,0)
        if idx == 10: return (3,1)
        if idx == 11: return (3,2)
        if idx == 12: return (3,4)
        if idx == 13: return (3,5)
        if idx == 14: return (3,6)
        if idx == 15: return (4,2)
        if idx == 16: return (4,3)
        if idx == 17: return (4,4)
        if idx == 18: return (5,1)
        if idx == 19: return (5,3)
        if idx == 20: return (5,5)
        if idx == 21: return (6,0)
        if idx == 22: return (6,3)
        if idx == 23: return (6,6)
        raise Exception("No equivalence")

    def display_state(self, state):
        colors_list = ['#325D75', '#BFC29F', '#FFFFFF', '#000000']
        cmap = colors.ListedColormap(colors_list)
        matrix = np.zeros(shape=(7,7), dtype = int)
        for i in range(7):
            for j in range(7):
                matrix[i,j] = -2
        for i in range(SIZE):
            matrix[JDM.idx_to_point(i)] = state[i]
        plt.clf()
        plt.imshow(matrix, cmap=cmap, vmin=-2, vmax=1)
        plt.title("État du jeu")
        plt.colorbar(ticks=[-2, -1, 0, 1], label="Accessible / Joueur")
        plt.show()

    def display_last_game(self):
        colors_list = ['#325D75', '#FFFFFF', '#BFC29F', '#000000']
        cmap = colors.ListedColormap(colors_list)
        for state in self.history:
            matrix = np.zeros(shape=(7,7), dtype = int)
            for i in range(7):
                for j in range(7):
                    matrix[i,j] = -2
            for i in range(SIZE):
                matrix[JDM.idx_to_point(i)] = state[0][i]
            plt.clf()
            plt.imshow(matrix, cmap=cmap, vmin=-2, vmax=1)
            plt.title("État du jeu")
            plt.colorbar(ticks=[-1, 0, 1, 2], label="Accessible / Joueur")
            plt.pause(0.1)
        print("Winner:", self.is_finished(), "in", len(self.history), "steps")
        plt.show()
