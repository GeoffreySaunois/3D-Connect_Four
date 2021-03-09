from DRL import *
from Connect4 import *

##
import torch
import torch.nn as nn
import numpy.random as npr
from random import sample
import torch.nn.functional as F

##
n_epochs = 10000

replay_memory_size = 1000
batch_size = 64
gamma = 0.9
lr = 1e-3
##
agent = DQNAgent('Chamby', lr=lr)
# todd = DQNAgent('Todd')         # Joueur 1
# alice = DQNAgent('Alice')       # Joueur 2
# agents = [todd, alice]
# past_state = None
# past_action = None
##


def get_action(state, _tau=1.):
    # Plus tard : jouer le coup gagnat si disponible
    probas = F.softmax(agent(state) * _tau, dim=0).detach().numpy()
    return npr.choice(len(probas), p=probas)


def get_action_eps(state, game, _eps=0):
    # Plus tard : jouer le coup gagnat si disponible
    if npr.rand() < _eps:
        ind = npr.randint(len(game.free_positions))
        return pos_to_seize(game.free_positions[ind])
    else:
        return agent(state).detach().numpy().argmax()


def add_replay_move(game, _tau=None, _eps=None):
    # Ajout d'un nouveau coup à la base de données
    state = game.get_state()
    if _tau is not None:
        action = get_action(state, _tau=_tau)
    else:
        action = get_action_eps(state, game, _eps)
    pos = seize_to_pos(action)

    # try_count = 0
    if pos not in game.free_positions:
        # L'agent a joué un coup illégal
        replay_memory.append((state, action, state, -100., True))
        if start_display:
            print("Tentative de coup illégal")
            # game.display_board()
        # On joue à la place un coup aléatoire parmi les coups possibles
        ind = npr.randint(len(game.free_positions))
        pos = game.free_positions[ind]
        action = pos_to_seize(pos)

    game.add_tokens(pos)

    if game.check_connect4(show=False):  # On vient de gagner
        replay_memory.append((temp[0], temp[1], temp[0], -1., True))
        replay_memory.append((state, action, state, 1., True))
        if start_display:
            print("Fini par p4")
            game.display_board(title='Ui')
        game.clear_board()  # Reset board
        temp[0] = None  # past_state = None
        temp[1] = None  # past_action = None

    if game.end_game():
        replay_memory.append((temp[0], temp[1], temp[0], 0., True))
        replay_memory.append((state, action, state, 0., True))
        if start_display:
            print("Fini car rempli")
            game.display_board()
        game.clear_board()  # Reset board
        temp[0] = None  # past_state = None
        temp[1] = None  # past_action = None


    new_state = game.get_state()
    if temp[0] is not None:
        replay_memory.append((temp[0], temp[1], new_state, 0., False))
    temp[0] = state
    temp[1] = action


def compute_q(agent_output, _actions):
    return agent_output[torch.arange(agent_output.shape[0]), _actions]


def compute_targets_q(_next_states, _rewards, _finals, _gamma):
    return _rewards + _gamma * (1 - _finals * 1) * torch.max(agent(_next_states), dim=1)[0]

current_game = Game()
temp = [None, None]  # (past_state, past_action)
replay_memory = []
start_display = False
count_display = 0

n_epochs = 30000
# tau_ini = 100
# tau_fin = 100
eps_ini = 1
eps_fin = 0.05

loss_mean = 0
inv_play_mean = 0
loss_history = []
inv_play_history = []

for _ in range(replay_memory_size):
    add_replay_move(current_game, eps_ini)
while len(replay_memory) > replay_memory_size:
    replay_memory.pop(0)

for epoch in range(n_epochs):
    agent.optimizer.zero_grad()

    eps = eps_ini + epoch * (eps_fin - eps_ini) / n_epochs

    # Ajout d'un nouveau coup
    # print("replay_mem size %d" %len(replay_memory))
    add_replay_move(current_game, _eps=eps)
    new_entries = len(replay_memory) - replay_memory_size
    if new_entries > 3:
        print('Noooooon')
        4/0
    while len(replay_memory) > replay_memory_size:
        replay_memory.pop(0)

    # # On regarde des trucs:
    # if epoch > 512:
    #     if current_game.ntok == 0:
    #         if not start_display:
    #             start_display = True
    #         else:
    #             4 / 0
    #     if start_display:
    #         count_display += 1
    #         print('n_tok: %d, n# new_entries: %d, count_display: %d'
    #               % (current_game.ntok,
    #                  new_entries,
    #                  count_display))
    #         print(np.abs(current_game.board).sum(axis=2))
    #         if count_display > 70:
    #             4 / 0
            # print(current_game.display_board())

    # Entrainement
    batch = sample(replay_memory, batch_size)
    states = torch.stack(tuple(d[0] for d in batch))
    actions = torch.tensor([d[1] for d in batch])
    next_states = torch.stack(tuple(d[2] for d in batch))
    rewards = torch.tensor([d[3] for d in batch])
    finals = torch.tensor([d[4] for d in batch])

    q_values = compute_q(agent(states), actions)
    target_q_values = compute_targets_q(next_states, rewards, finals, gamma)
    target_q_values.detach()

    loss = F.mse_loss(q_values, target_q_values)
    loss.backward()
    agent.optimizer.step()

    loss_mean += loss.item() / batch_size
    inv_play_mean += finals.sum().float().item() / batch_size
    loss_history.append(loss_mean / batch_size / (epoch + 1))
    inv_play_history.append(inv_play_mean / (epoch + 1))

    print("\rEpoch %d/%d, loss = %.4f, invalid rate = %.4f"
          % (epoch,
             n_epochs,
             loss_mean/(epoch+1),
             inv_play_mean / (epoch+1)), end='')

print()
##
plt.plot(loss_history)
plt.title('loss_history')
plt.show()

plt.plot(inv_play_history)
plt.title('invalid rate history')
plt.show()
##

# À toi de jouer

g = Game()

##
agent(g.get_state()).view(4, 4)

##
def play():
    print(agent(g.get_state()).view(4, 4))
    n = torch.argmax(agent(g.get_state())).item()
    pos = 10 * (n // 4) + (n % 4)
    print(pos)
    g.add_tokens(pos)
    g.display_board(g.ntok)
    if g.check_connect4(show=False):
        print('Connect 4 !!')
        g.clear_board()

##
