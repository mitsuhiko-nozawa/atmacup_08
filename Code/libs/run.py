from prepro import Prepro
from train import Train
from inference import Infer
from logger import Logger
from utils import make_data

class Run():
    def __init__(self, param):
        self.param = param
        self.param["prepro_param"].update(self.param["common_param"])
        self.param["train_param"].update(self.param["common_param"])

        self.Prepro = Prepro(self.param["prepro_param"])
        self.Train = Train(self.param["train_param"])
        self.Infer = Infer(self.param["train_param"])
        self.Logger = Logger(self.param["train_param"])

    def __call__(self):
        module = make_data(self.param["common_param"])
        module = self.Prepro(module)
        module = self.Train(module)
        module = self.Infer(module)
        self.Logger(module)
        print("success")
