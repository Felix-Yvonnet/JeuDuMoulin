from jdm import *
from net import *
import torch
import torch.optim as optim
import pickle
import os
from tqdm import tqdm

args = {
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'eta' : 0.3,
    'decay' : 0.85
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
        self.place = NNMNNN(game, args["dropout"],2)
        self.remove = NNMNNN(game, args["dropout"],2)
        self.move = NNMNNN(game, args["dropout"],3)
        self.game = game
        self.size = self.game.getBoardSize()
        self.action_size = self.game.getActionSize()
        if args["cuda"]:
            self.place.cuda()
            self.remove.cuda()
            self.move.cuda()

    def update_function(self, eta):
        # self.game.policy_2 = self.game.policy_1
        self.game.policy_1 = lambda game: self.predict(game,-1, train=True)
        # self.game.policy_1 = lambda game: self.predict(game,-1) if np.random.random()>eta else random_policy(game,-1)
        # self.game.policy_2 = lambda game: self.predict(game,1) if np.random.random()>eta else random_policy(game,1)


    def train(self, set_size = 1000, iter_fun=1, epochs=10,  init_only=False):
        optimizer = optim.Adam(self.place.parameters(), lr = args["lr"])
        optimizer2 = optim.Adam(self.remove.parameters(), lr = args["lr"])
        optimizer3 = optim.Adam(self.move.parameters(), lr = args["lr"])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.9)
        scheduler3 = optim.lr_scheduler.ExponentialLR(optimizer3, gamma=0.9)
        mean_loss = []
        self.place.train()
        self.remove.train()
        self.move.train()
        for fun_idx in range(iter_fun):
            examples, rewards = self.game.batch_reward(set_size)
            places = list(filter(lambda x: x[SIZE], examples))
            places = torch.FloatTensor(np.array(places).astype(np.float64))
            removes = list(filter(lambda x: x[SIZE+1], examples))
            removes = torch.FloatTensor(np.array(removes).astype(np.float64))
            moves = list(filter(lambda x: x[SIZE+2], examples))
            moves = torch.FloatTensor(np.array(moves).astype(np.float64))
            places_reward = list(filter(lambda x: examples[x[0]][SIZE], enumerate(rewards)))
            places_reward = [p[0] for p in places_reward]
            places_reward = torch.FloatTensor(np.array(places_reward).astype(np.float64))
            removes_reward = list(filter(lambda x: examples[x[0]][SIZE+1], enumerate(rewards)))
            removes_reward = [p[0] for p in removes_reward]
            removes_reward = torch.FloatTensor(np.array(removes_reward).astype(np.float64))
            moves_reward = list(filter(lambda x: examples[x[0]][SIZE+2], enumerate(rewards)))
            moves_reward = [p[0] for p in moves_reward]
            moves_reward = torch.FloatTensor(np.array(moves_reward).astype(np.float64))
            
            t = tqdm(range(epochs), desc='Epoch')
            for epoch in t:
                np.random.shuffle(places)
                if not init_only:
                    np.random.shuffle(removes)
                    np.random.shuffle(moves)
                
                loss_history = []
                mean_loss_temp = []

                batch_size = args["batch_size"]
                
                place_count = int(batch_size/len(examples)*len(places))                
                for i in range(int(np.ceil(len(places) / place_count))):
                    boards = places[place_count * i: min(place_count*(i+1), len(places))]
                    target_vs = places_reward[place_count * i: min(place_count*(i+1), len(places))]
                    
                    # predict
                    if args["cuda"]:
                        boards, target_vs = boards.contiguous().cuda(), target_vs.contiguous().cuda()

                    # compute output
                    out_vs = self.place(boards).flatten()
                    loss = self.loss_fun(target_vs, out_vs)

                    # record loss
                    loss_history.append(loss.item())

                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                mean_loss_temp.append(np.mean(loss_history))
                loss_history = []
                
                removes_count = int(batch_size/len(examples)*len(removes))
                for i in range(int(np.ceil(len(removes) / removes_count))):
                    boards = removes[removes_count * i: min(removes_count*(i+1), len(removes))]
                    target_vs = removes_reward[removes_count * i: min(removes_count*(i+1), len(removes))]
                    
                    # predict
                    if args["cuda"]:
                        boards, target_vs = boards.contiguous().cuda(), target_vs.contiguous().cuda()

                    # compute output
                    out_vs = self.remove(boards).flatten()
                    loss = self.loss_fun(target_vs, out_vs)

                    # record loss
                    loss_history.append(loss.item())

                    # compute gradient and do SGD step
                    optimizer2.zero_grad()
                    loss.backward()
                    optimizer2.step()

                mean_loss_temp.append(np.mean(loss_history))
                loss_history = []

                moves_count = int(batch_size/len(examples)*len(moves))
                for i in range(int(np.ceil(len(moves) / moves_count))):
                    boards = moves[moves_count * i: min(moves_count*(i+1), len(moves))]
                    target_vs = moves_reward[moves_count * i: min(moves_count*(i+1), len(moves))]
                    
                    # predict
                    if args["cuda"]:
                        boards, target_vs = boards.contiguous().cuda(), target_vs.contiguous().cuda()

                    # compute output
                    out_vs = self.move(boards).flatten()
                    loss = self.loss_fun(target_vs, out_vs)

                    # record loss
                    loss_history.append(loss.item())

                    # compute gradient and do SGD step
                    optimizer3.zero_grad()
                    loss.backward()
                    optimizer3.step()
                
                mean_loss_temp.append(np.mean(loss_history))

                mean_loss.append(mean_loss_temp)

            self.update_function(args["eta"]*(args["decay"] ** fun_idx))
            scheduler.step()
            
            if not init_only:
                scheduler2.step()
                scheduler3.step()
            # set_size //= 2


        plt.plot(mean_loss)
        plt.title("Loss at each epoch")
        plt.show()

    def predict(self, game, player, train=False):
        """
        board: np array with board
        """
        # timing
        if train:
            self.place.train()
            self.remove.train()
            self.move.train()
        else:
            self.place.eval()
            self.remove.eval()
            self.move.eval()
        board = game.board
        state = 2*game.board[-1] + game.board[-2]
        board_play = board[:-3]
        board_tensor = board + [player]
        board_tensor = torch.FloatTensor(np.append(board_tensor.astype(np.float64), (0,0) if game.board[-1] else (0,))).unsqueeze(dim=0)
        if state == 0: # choose a place to place a pawn
            possible = (board_play==0).nonzero()[0]
            actions = np.zeros(len(possible))
            for i,action in enumerate(possible):
                board_tensor[0][-1] = action
                actions[i] = self.place(board_tensor)
            out = possible[actions.argmax() if player>0 else actions.argmin()], 25
        elif state == 1: # remove a pawn from a place
            possible = (board_play==-player).nonzero()[0]
            actions = np.zeros(len(possible))
            for i,action in enumerate(possible):
                board_tensor[0][-1] = action
                actions[i] = self.remove(board_tensor)
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
                        actions.append((self.move(board).item(), action1, action2))
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
        places = list(filter(lambda x: x[0][-3], examples))
        removes = list(filter(lambda x: x[0][-2], examples))
        moves = list(filter(lambda x: x[0][-1], examples))
        self.place.eval()
        self.remove.eval()
        self.move.eval()

        success = 0
        for i,example in enumerate([places, removes, moves]):
            boards, actions, vs = list(zip(*example))
            boards = list(map(lambda x: np.append(x[0],x[1]), zip(boards, actions))) # extend boards with chosen actions
            boards = torch.FloatTensor(np.array(boards).astype(np.float64))
            target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

            # compute output
            if i == 0:
                out_vs = self.place(boards).flatten()
            elif i == 1:
                out_vs = self.remove(boards).flatten()
            else:
                out_vs = self.move(boards).flatten()
            success += (out_vs.round()==target_vs).sum()
        print(f"On dataset, performs {success/len(examples):0.03f}")


    def loss_fun(self, targets, outputs):
        return torch.sum((targets - outputs)**2) / targets.size()[0]


    def save_checkpoint(self, folder='/home/fred/ENS/jdm/models/', filename='model.pt'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        torch.save(self.place.state_dict(), filepath + ".p")
        torch.save(self.remove.state_dict(), filepath + ".r")
        torch.save(self.move.state_dict(), filepath + ".m")
        print("Weights saved!")

    def load_checkpoint(self, folder='/home/fred/ENS/jdm/models/', filename='model.pt'):
        device = torch.cuda.is_available()
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath+".p"):
            raise ("No model in path {}".format(filepath))
        print("Loarding...")
        self.place.load_state_dict(torch.load(filepath + ".p"))
        self.remove.load_state_dict(torch.load(filepath + ".r"))
        self.move.load_state_dict(torch.load(filepath + ".m"))


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


