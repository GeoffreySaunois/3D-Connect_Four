import torch
import torch.nn as nn


##

class DQNAgent(nn.Module):

    def __init__(self, name, in_dim=128, h_dim=128, out_dim=16, lr=1e-5, gamma=0.99):
        super(DQNAgent, self).__init__()
        self.name = name
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.lr = lr
        # self.gamma = gamma

        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, 2*self.h_dim),
            nn.ReLU(),
            nn.Linear(2*self.h_dim, self.out_dim)
        )

        self.optimizer = torch.optim.Adam(self.fc.parameters(), lr=self.lr)

    def forward(self, state):
        if state is None:
            return 0.
        return self.fc(state)
