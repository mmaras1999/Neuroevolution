
import numpy as np
import pickle
import matplotlib.pyplot as plt
from lib.utilities import load_obj

#check best[644] (645 population), had -14 score v10
#check best[1163] (1164 population), had -15.8 score v10
cma_es = load_obj(140, 'models/neat_pong_v5') 

scores = []
for gen, scr, fen in cma_es.bestgens:
    scores.append(scr)

plt.plot(scores, label='best')
aver = 30
plt.plot(range(aver-1, len(scores)), np.convolve(scores, np.ones(aver) / aver, mode='valid'), label=f'moving avg {aver}')
plt.legend()
plt.show()

