import numpy as np
import pygame
from threading import Thread
from threading import Event
pygame.font.init()
from model import random_policy, NetJDM
from model3 import NetJDM as MM3
from jdm import JDM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--load-model", help="model to load", type=str, default="model.pt")
parser.add_argument("--load-path", help="path to the model", type=str, default="/home/fred/ENS/jdm/models/")
parser.add_argument("--random", help="random model", action="store_true")
parser.add_argument("--is3", help="three heads model", action="store_true")
args = parser.parse_args()


if args.random:
    print("Using random policy")
    enemy = random_policy
else:
    if args.is3:
        model = MM3()
    else:
        model = NetJDM()
    model.load_checkpoint(args.load_path, args.load_model)
    enemy = model.predict
    print("Using selected model")

# SETTINGS
BACKGROUND = (110, 110, 110) # gray
PLAYER1 = (220, 220, 220) # almost white
PLAYER2 = (30, 30, 30) # almost black
LINES = (183, 157, 107) # white
SURLIGN = (200, 150, 150) # redish
POSSIBLE = (150, 150, 200) # blueish

FONT = pygame.font.SysFont("comicsans", 40)

PIECE_SIZE = 30
SCALE = 40

# globals - don't change
clickCondition = Event()
clickX = 0
clickY = 0

# Lookup table for coordinates
coord_arr = np.array([(1,1), (7,1), (13,1),    (3,3), (7,3), (11,3),    (5,5), (7,5), (9,5),    (1,7), (3,7), (5,7),
                    (9,7), (11,7), (13,7),    (5,9), (7,9), (9,9),    (3,11), (7,11), (11,11),    (1,13), (7,13), (13,13)], dtype=[('x', 'i4'),('y', 'i4')])



def getCoords(i):
    return [coord_arr['x'][i], coord_arr['y'][i]]
def getIndex(x,y):
    for i, pos in enumerate(coord_arr):
        if x == pos[0] and y==pos[1]:
            return i
    return -1

class GameState:
    def __init__(self, enemy):
        self.enemy = enemy
        self.game = JDM(None, enemy)
        self.selected = None
        self.game.restart()
        self.init_gui()

    def init_gui(self):
        thread = Thread(target = self.update_gui)
        thread.start()

    def update_gui(self):
        done = False
        if pygame.get_init(): return
        print("Initializing game")
        self.screen = pygame.display.set_mode((550, 580))
        pygame.display.set_caption("Jeu du moulin")
        self.screen.fill(BACKGROUND)
        self.clock = pygame.time.Clock()
        print("Game launched")

        while not done and not self.game.is_finished():
            global clickX, clickY, clickCondition
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        clickCondition.set()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        clickX, clickY = pygame.mouse.get_pos()
                        clickX = round(clickX / SCALE)
                        clickY = round(clickY / SCALE)
                        clickCondition.set()

                self.screen.fill(BACKGROUND)
                self.draw_info()
                self.update_game(clickX, clickY)
                clickX, clickY = 0,0
                self.draw_lines()
                self.draw_pieces()

                pygame.display.flip()
                self.clock.tick(60)
            except pygame.error:
                import sys
                sys.exit(0)
        pygame.quit()

    def update_game(self, clickX, clickY):
        index = getIndex(clickX, clickY)
        if (clickX>0 or clickY>0) and self.game.playing == 1 and self.selected is not None and (index < 0 or self.game.board[index]!=0):
            self.selected = None
        if self.game.playing == 1 and index>=0:
            if self.game.board[-3]:
                if self.game.board[index] == 0:
                    self.game.play(index, 25)
            elif self.game.board[-2]:
                if self.game.board[index] == -1:
                    self.game.play(index, 25)
            elif self.game.board[-1]:
                if self.selected is None:
                    if self.game.board[index] == 1:
                        self.selected = index
                else:
                    if self.game.board[index] == 0:
                        self.game.play(self.selected, index)
                        self.selected = None
        elif self.game.playing == -1: # opponent turn
            from_pos, to_pos = random_policy(self.game, -1)
            self.game.play(from_pos, to_pos)

    def draw_info(self):
        action = None
        if self.game.board[-3]:
            action = "placing"
        elif self.game.board[-2]:
            action = "removing"
        elif self.game.board[-1]:
            action = "moving"

        action_text = FONT.render("Action: "+ str(action),1,(255,255,255))
        self.screen.blit(action_text, (10, 550))

        if self.game.before_moving:
            placing_text = FONT.render("x"+ str(9-self.game.pawn_placed),1,(255,255,255))
            self.screen.blit(placing_text, (500, 550))

    def draw_lines(self):
        # Upper horizontal lines
        self.draw_line([0.9, 1], [13.1, 1])
        self.draw_line([2.9, 3], [11.1, 3])
        self.draw_line([4.9, 5], [9.1, 5])
        # Lower horizontal lines
        self.draw_line([4.9, 9], [9.1, 9])
        self.draw_line([2.9, 11], [11.1, 11])
        self.draw_line([0.9, 13], [13.1, 13])
        # Middle horizontal lines
        self.draw_line([0.9, 7], [5.1, 7])
        self.draw_line([8.9, 7], [13.1, 7])

    def draw_line(self, start, end):
        pygame.draw.line(self.screen, LINES, [x*SCALE for x in start], [x*SCALE for x in end], 10)
        pygame.draw.line(self.screen, LINES, [x*SCALE for x in start[::-1]], [x*SCALE for x in end[::-1]], 10)

    def draw_pieces(self):
        # Draw board
        for i in range(self.game.getActionSize()):
            self.draw_piece(getCoords(i), self.game.board[i])
        if self.selected is not None:
            self.draw_piece(getCoords(self.selected), 2)
            for neigh in self.game.neighbors[self.selected]:
                if self.game.board[neigh] == 0:
                    self.draw_piece(getCoords(neigh), 3)

    def draw_piece(self, pos, value):
        if value != 0:
            color = None
            if value == 1:
                color = PLAYER1
            elif value == -1:
                color = PLAYER2
            elif value == 2:
                color = SURLIGN
            elif value == 3:
                color = POSSIBLE
            pygame.draw.circle(self.screen, color, [x*SCALE for x in pos], PIECE_SIZE)
            pygame.draw.circle(self.screen, [x-value*20 for x in color], [x*SCALE for x in pos], PIECE_SIZE, int(PIECE_SIZE/6))


if __name__ == "__main__":
    GameState(enemy)