from games.balance_game import BalanceGame
from lib.utilities import load_obj

model = load_obj(300, 'models/neat_balance_v3') 
game = BalanceGame()
print(game.play(model.bestgens[-1][2] ,move_fun=BalanceGame.make_move_det, render=True, games_amount=3))
