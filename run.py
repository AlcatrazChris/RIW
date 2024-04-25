import torch
import argparse
import logging

import utils
import config
from model import Model
from train import train
from test import test

"""Initialization"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(16)

