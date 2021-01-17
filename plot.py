
import numpy as np
import pickle
import matplotlib.pyplot as plt
from lib.utilities import load_obj

#check best[644] (645 population), had -14 score
#check best[1163] (1164 population), had -15.8 score
cma_es = load_obj(1360, 'models/cmaes_pong_v10')

scores = []
for gen, scr in cma_es.bestgens:
    scores.append(scr)

plt.plot(scores, label='best')
aver = 30
plt.plot(range(aver-1, len(scores)), np.convolve(scores, np.ones(aver) / aver, mode='valid'), label=f'moving avg {aver}')
plt.legend()
plt.show()

