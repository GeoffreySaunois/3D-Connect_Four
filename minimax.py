import numpy as np
from Connect4 import *
from copy import deepcopy
import time
import numpy.random as npr


##
class Minimax:
    def __init__(self, _game):
        self.game = _game

    def max_ab(self, alpha, beta, depth):
        if self.game.check_connect4(show=False):
            return -1000 - depth, None                # L'adversaire vient de gagner
        if self.game.end_game():
            return 0, None
        if depth == 0:
            return self.game.get_position_value(), None

        # If not
        best_value = -np.inf
        best_pos = None
        possible_moves = deepcopy(self.game.free_positions)
        npr.shuffle(possible_moves)        # Pour supprimer le biais
        for pos in possible_moves:
            self.game.add_tokens(pos)
            next_value, _ = self.min_ab(alpha, beta, depth-1)
            if next_value >= best_value:
                best_value = next_value
                best_pos = pos
            alpha = max(alpha, next_value)
            self.game.remove_last()
            if alpha > beta:
                return best_value, best_pos

        return best_value, best_pos

    def min_ab(self, alpha, beta, depth):
        if self.game.check_connect4(show=False):
            return 1000 + depth, None                 # On vient de gagner
        if self.game.end_game():
            return 0, None
        if depth == 0:
            return self.game.get_position_value(), None

        # If not
        best_value = +np.inf        # best_value est ici la valeur minimum trouvée
        best_pos = None
        possible_moves = deepcopy(self.game.free_positions)
        npr.shuffle(possible_moves)  # Pour supprimer le biais
        for pos in possible_moves:
            self.game.add_tokens(pos)
            next_value, _ = self.max_ab(alpha, beta, depth-1)
            if next_value <= best_value:
                best_value = next_value
                best_pos = pos
            beta = min(beta, next_value)
            self.game.remove_last()
            if alpha > beta:
                return best_value, best_pos

        return best_value, best_pos

    def play_game_ab(self, depth_max):
        self.game.clear_board()
        while True:
            if self.game.check_connect4(show=False):
                # print("it's a 4")
                self.game.display_board()
                if self.game.ntok % 2 == 0:
                    print('player 2 win!')
                else:
                    print('player 1 win!')
                break
            if self.game.end_game():
                self.game.display_board()
                print("It's a tie!")
                break

            if self.game.ntok % 2 == 0:
                node_value, best_move = self.max_ab(-np.inf, np.inf, depth_max)
                print("player 1", node_value, best_move)
                self.game.add_tokens(best_move)
                self.game.display_board()
            else:
                node_value, best_move = self.min_ab(-np.inf, np.inf, depth_max)
                print("player 2", node_value, best_move)
                self.game.add_tokens(best_move)
                self.game.display_board()

    def play_against(self, human_first=None, depth_max=4):
        if human_first is None:
            if npr.rand() < 0.5:
                human_first = True
            else:
                human_first = False
        print("Ceci est une fonction permettant de jouer contre l'IA du minimax.")
        print("L'ensemble des coups possibles est une grille 4x4.")
        print("Pour jouer un coup en position (x, y), il suffit de renter le nombre 'xy'")
        if human_first:
            print("Vous commencez !")

        else:
            print("l'IA commence !")

        self.game.clear_board()

        while True:
            if self.game.check_connect4(show=False):
                self.game.display_board()
                if (self.game.ntok % 2 == 0) == (human_first):
                    print("L'IA a gagné !")
                else:
                    print('Vous avez gagné !')
                break
            if self.game.end_game():
                self.game.display_board()
                print("Match Nul !")
                break

            if (self.game.ntok % 2 == 0) == (human_first):
                player_move = int(input('À vous de jouer ! Rentrez la position de votre coup : '))
                while player_move not in self.game.free_positions:
                    print("Coup invalide, les coups disponibles sont:")
                    print(self.game.free_positions)
                self.game.add_tokens(player_move)
                self.game.display_board()
            else:
                node_value, best_move = self.min_ab(-np.inf, np.inf, depth_max)
                print("L'IA joue en :", best_move, "la valeur du jeu est:", node_value)
                self.game.add_tokens(best_move)
                self.game.display_board()


g = Game()
m = Minimax(g)

# start = time.time()
# m.play_game_ab(depth_max=4)
# end = time.time()
# print("Running time for game simulation: ", round(end-start, 7))

##

m.play_against()

##

