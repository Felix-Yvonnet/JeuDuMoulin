from jdm import *
from net import *
import torch
import torch.optim as optim
import pickle
import os

args = {
    'lr': 0.001,
    'dropout': 0.1,
    'epochs': 12,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'decay' : 0.85,
}


def random_policy(game, player):
    state = 2*game.board[-1] + game.board[-2]
    board = game.board[:-3]
    if state == 0: # choose a place to place a pawn
        return np.random.choice((board==0).nonzero()[0]),25 # 25=flag fail
    elif state == 1: # remove a pawn from a place
        return np.random.choice((board==-player).nonzero()[0]),25
    elif state == 2: # move a pawn from a place to another
        possible = (board==player).nonzero()[0]
        possible = [p for p in possible if game.is_open(p)]
        if len(possible) == 0: return 25,25
        choice = np.random.choice(possible)
        return choice, np.random.choice([p for p in game.neighbors[choice] if board[p] == 0])

class NetJDM:
    def __init__(self):
        game = JDM(random_policy,random_policy,200)
        self.nnet = NNMNN(game, args["dropout"])
        self.game = game
        self.size = self.game.getBoardSize()
        self.action_size = self.game.getActionSize()
        if args["cuda"]:
            self.nnet.cuda()

    def update_function(self):
        self.game.policy_2 = self.game.policy_1
        self.game.policy_1 = lambda game: self.predict(game,-1)        

    def train(self, set_size = 1000, iter_fun=1, epochs=10):
        optimizer = optim.Adam(self.nnet.parameters(), lr = args["lr"])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

        mean_loss = []
        self.nnet.train()
        for fun_idx in range(iter_fun):
            examples = self.game.batch_example(set_size)
            t = tqdm(range(epochs), desc='Epoch')
            for epoch in t:
                loss_history = []

                batch_count = int(len(examples) / args["batch_size"])

                for _ in range(batch_count):
                    sample_ids = np.random.randint(len(examples), size=args["batch_size"])
                    boards, actions, vs = list(zip(*[examples[i] for i in sample_ids]))
                    boards = list(map(lambda x: np.append(x[0],x[1]), zip(boards, actions))) # extend boards with chosen actions
                    boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                    target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                    # predict
                    if args["cuda"]:
                        boards, target_vs = boards.contiguous().cuda(), target_vs.contiguous().cuda()

                    # compute output
                    out_vs = self.nnet(boards).flatten()
                    loss = self.loss_fun(target_vs, out_vs)

                    # record loss
                    loss_history.append(loss.item())

                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                mean_loss.append(sum(loss_history) / len(loss_history))

            self.update_function()
            scheduler.step()


        plt.plot(mean_loss)
        plt.title("Loss at each epoch")
        plt.show()

    def predict(self, game, player):
        """
        board: np array with board
        """
        # timing
        self.nnet.eval()
        board = game.board
        state = 2*game.board[-1] + game.board[-2]
        board_play = board[:-3]
        board = torch.FloatTensor(np.append(board.astype(np.float64), (0,0))).unsqueeze(dim=0)
        if state == 0: # choose a place to place a pawn
            possible = (board_play==0).nonzero()[0]
            actions = np.zeros(len(possible))
            for i,action in enumerate(possible):
                board[0][-2] = action
                board[0][-1] = 25
                actions[i] = self.nnet(board)
            out = possible[actions.argmax() if player>0 else actions.argmin()], 25
        elif state == 1: # remove a pawn from a place
            possible = (board_play==-player).nonzero()[0]
            actions = np.zeros(len(possible))
            for i,action in enumerate(possible):
                board[0][-2] = action
                board[0][-1] = 25
                actions[i] = self.nnet(board)
            if len(actions) == 0:
                return 25,25
            out = possible[actions.argmax() if player>0 else actions.argmin()], 25
        elif state == 2: # move a pawn from a place to another
            possible = (board_play==player).nonzero()[0]
            possible = [p for p in possible if game.is_open(p)]
            if len(possible) == 0: return 25,25

            actions = []
            for action1 in possible:
                for action2 in game.neighbors[action1]:
                    if game.board[action2] == 0:
                        board[0][-2] = action1
                        board[0][-1] = action2
                        actions.append((self.nnet(board).item(), action1, action2))
            if len(actions) == 0:
                return 25,25
            if player > 0:
                max = np.argmax([action[0] for action in actions])
            else:
                max = np.argmin([action[0] for action in actions])

            out = actions[max][1], actions[max][2]

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return out


    def test(self, num_tests=100):

        examples = self.game.batch_example(num_tests)
        self.nnet.eval()

        boards, actions, vs = list(zip(*examples))
        boards = list(map(lambda x: np.append(x[0],x[1]), zip(boards, actions))) # extend boards with chosen actions
        boards = torch.FloatTensor(np.array(boards).astype(np.float64))
        target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

        # compute output
        out_vs = self.nnet(boards).flatten()
        success = (out_vs.round()==target_vs).sum()
        print(f"On dataset, performs {success/len(examples):0.03f}")


    def loss_fun(self, targets, outputs):
        return torch.sum((targets - outputs)**2) / targets.size()[0]


    def save_checkpoint(self, folder='/home/fred/ENS/jdm/models/', filename='model.pt'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        torch.save(self.nnet.state_dict(), filepath)
        print("Weights saved!")

    def load_checkpoint(self, folder='/home/fred/ENS/jdm/models/', filename='model.pt'):
        device = torch.cuda.is_available()
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        print("Loarding...")
        self.nnet.load_state_dict(torch.load(filepath))
        self.nnet.eval()


    def dump_history(self, examples, folder=r'D:\info\jeu_du_moulin_deep_learning', filename='examples.pickle', append=True):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        with open(filepath, "ab" if append else "wb") as fd:
            pickle.dump(examples, fd)
        print("Data successfully saved at", folder)

    def load_history(slef, folder=r'D:\info\jeu_du_moulin_deep_learning', filename='examples.pickle'):

        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        with open(filepath, "rb") as fd:
            data = pickle.load(fd)
        return data


    def test_player(self, enemy, n_iter):
        tot_win = 0
        tot_loss = 0
        game = JDM(enemy, self.predict, 100)
        for _ in tqdm(range(n_iter)):
            game.restart()
            winner = game.run()
            if winner == 1:
                tot_win+=1
            elif winner == -1:
                tot_loss+=1
        print(f"Losses: {tot_loss}")
        print(f"Victories: {tot_win}")
        print(f"Ties: {n_iter-tot_loss-tot_win}")


