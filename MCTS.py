import numpy as np
from collections import defaultdict
from jdm import JDM
from model import random_policy


class MonteCarloTreeSearchNode():
    def __init__(self, game: JDM, parent=None, parent_action=None):
        self.game = game
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = self.untried_actions()
    
    def untried_actions(self):
        self._untried_actions = self.get_legal_actions()
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        
        action = self._untried_actions.pop()
        next_state = self.game.move(action)
        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node 

    def is_terminal_node(self):
        return self.game.is_finished()
    
    def rollout(self):
        reward = 0
        current_rollout_state = self.game.board
        
        while not current_rollout_state.is_finished():
            
            possible_moves = current_rollout_state.legal_actions()
            action = self.rollout_policy(possible_moves)
            reward += current_rollout_state.play(*action)*0.1
        return reward+current_rollout_state.is_finished()*2

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):

        current_node = self
        while not current_node.is_terminal_node():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        simulation_no = 100
        for i in range(simulation_no):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        
        return self.best_child(c_param=0.)

    def get_legal_actions(self):
        return self.game.legal_actions()








def main():
    jdm = JDM(random_policy, random_policy)
    root = MonteCarloTreeSearchNode(jdm)
    selected_node = root.best_action()
    print(selected_node)





