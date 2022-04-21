#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:50:14 2022

@author: Owen
"""
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.utils.data as data_utils
import torchvision.transforms as transforms

conn = sqlite3.connect('master.sqlite')
c = conn.cursor()

df_week = pd.read_sql_query("SELECT * FROM combined_weekly_encoded_scaled GROUP BY year_week",conn)
df_y = pd.read_sql_query("SELECT avg_deaths FROM combined_weekly_encoded_scaled",conn)


target = df_week['avg_deaths']
del df_week['avg_deaths']
del df_week['year_week']

train = df_week.to_numpy(dtype = "float32")
target = target.to_numpy(dtype = 'float32')

train = torch.from_numpy(train)
target = torch.from_numpy(target)

small_train = train[0:60000]
small_target = target[0:60000]

class LNet(nn.Module):
    
    def __init__(self, in_size, out_size, hid_size,layers):
        super().__init__()
        self.in_to_hidden = nn.Linear(in_size, hid_size, bias = False)
        self.hidden_to_out = nn.Linear(hid_size, out_size, bias = False)
    
    def forward(self, x):
        hidden = self.in_to_hidden(x)
        output = self.hidden_to_out(hidden)
        return hidden, output
    
def train(model, inputs, targets, n_epochs, lr, illusory_i=0):
  """
  Training function

  Args:
    model: torch nn.Module
      The neural network
    inputs: torch.Tensor
      Features (input) with shape `[batch_size, input_dim]`
    targets: torch.Tensor
      Targets (labels) with shape `[batch_size, output_dim]`
    n_epochs: int
      Number of training epochs (iterations)
    lr: float
      Learning rate
    illusory_i: int
      Index of illusory feature

  Returns:
    losses: np.ndarray
      Record (evolution) of training loss
    modes: np.ndarray
      Record (evolution) of singular values (dynamic modes)
    rs_mats: np.ndarray
      Record (evolution) of representational similarity matrices
    illusions: np.ndarray
      Record of network prediction for the last feature
  """
  losses = np.zeros(n_epochs)  # Loss records

  optimizer = optim.SGD(model.parameters(), lr=lr)
  criterion = nn.MSELoss()
  
  
  for i in range(n_epochs):
    optimizer.zero_grad()
    predictions, hiddens = model(inputs)
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()

    # Logging (recordings)
    losses[i] = loss.item()

  return losses

input_size = 60
out_size = 2
hidden_size = 119


model = LNet(input_size, out_size, hidden_size, 1)
losses = train(model, small_train, small_target, n_epochs = 500, lr = 1e-5)
print(losses)