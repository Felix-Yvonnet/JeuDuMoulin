from model import random_policy, NetJDM

model = NetJDM()

model.load_checkpoint("/home/fred/ENS/jdm/models/", "model.pt")
model.test(1000)

model.test_player(random_policy, 1000)
