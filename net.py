import torch.nn as nn
import torch.nn.functional as F

class NNMNN(nn.Module):
    def __init__(self, game, dropout):

        super(NNMNN, self).__init__()

        self.size = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.dropout = dropout

        self.fc1 = nn.Linear(self.size + 1 + 1, 256) # [board, (action_in, action_out)] -> prob to win

        self.fc2 = nn.Linear(256, 64)

        self.fc3 = nn.Linear(64, 1)


    def forward(self, s):
        s = F.dropout(F.relu(self.fc1(s)), p=self.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc2(s)), p=self.dropout, training=self.training)

        return self.fc3(s)



class NNMNNN(nn.Module):
    def __init__(self, game, dropout, n):

        super(NNMNNN, self).__init__()

        self.size = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.dropout = dropout

        self.fc1 = nn.Linear(self.size + n, 256) # [board, (action_in, action_out)] -> prob to win

        self.fc2 = nn.Linear(256, 64)

        self.fc3 = nn.Linear(64, 1)


    def forward(self, s):
        s = F.dropout(F.relu(self.fc1(s)), p=self.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc2(s)), p=self.dropout, training=self.training)

        return self.fc3(s)
