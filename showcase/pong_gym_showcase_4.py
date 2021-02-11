from games.pong_gym_v2 import PongGame
from lib.neat import Neat
from lib.utilities import load_obj


neat = load_obj(100, 'models/pong_gym_v2/neat_pong_gym_det')
game = PongGame()
individual = neat.bestgens[-1][2]
game.play(individual, True, wait=None, move_fun=PongGame.make_move_det)
