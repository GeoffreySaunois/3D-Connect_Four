from Connect4 import *
from MCTS import *
from CNN import *

# hyper-parameters
num_iters = 1
numEps = 1
threshold = 0.6
numMCTSSims = 1


def policyIterSP():
    # nnet = initNNet()  # initialise random neural network
    nnet = 1
    examples = []
    for i in range(num_iters):
        for e in range(numEps):
            examples += executeEpisode(nnet)  # collect examples from this game
        new_nnet = trainNNet(examples)
        frac_win = compare_nets(new_nnet, nnet)  # compare new net with previous net
        if frac_win > threshold:
            nnet = new_nnet  # replace with new net
    return nnet


def executeEpisode():  # executeEpisode(game, nnet):
    examples = []
    game = Game()
    mcts = MCTS()  # initialise search tree

    while True:
        for _ in range(numMCTSSims):
            mcts.search(game)  # mcts.search(s, game, nnet)
            print(mcts.visited)
        examples.append([s, mcts.pi(s), None])  # rewards can not be determined yet
        a = random.choice(len(mcts.pi(s)), p=mcts.pi(s))  # sample action from improved policy
        s = game.nextState(s, a)
        if game.gameEnded(s):
            examples = assignRewards(examples, game.gameReward(s))
            return examples


executeEpisode()


def compare_nets():
    return 0
