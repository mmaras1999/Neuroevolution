#not much of a game, just o benchmark for xor problem

class XorGame():
    def play(self, NN, render=False):
        score = 0
        for x in [0,1]:
            for y in [0,1]:
                correct = x ^ y
                result = NN.eval([x,y])[0]
                score += abs(correct - result) ** 2

                if render:
                    print(x, y, f'-> correct: {correct}, nn: {result}')

        return 4 - score
