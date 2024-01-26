from model import *
from model3 import NetJDM as Model
from model3reward import NetJDM as RewardModel
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="number of epoch", type=int, default=10)
parser.add_argument("--trains", help="number of tests", type=int, default=100)
parser.add_argument("--iter-fun", help="number of function iterated", type=int, default=1)
parser.add_argument("--name", help="number of function iterated", type=str, default="model.pt")
parser.add_argument("--tests", help="number of function iterated", type=int, default=100)
parser.add_argument("--init-only", help="number of function iterated", action="store_true")
parser.add_argument("--load", help="number of function iterated", type=str, default="")
args = parser.parse_args()

model = RewardModel()
if args.load != "":
    model.load_checkpoint(filename=args.load)

model.train(args.trains, args.iter_fun, args.epoch, args.init_only)
model.test(args.tests)
model.save_checkpoint("/home/fred/ENS/jdm/models/", args.name)