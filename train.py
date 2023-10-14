import torch
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import StockPredictor, settings

def load(dataset, context_length):
    """Returns src/tgt train/valid data given a context length"""
    values = dataset["value"].to_list()
    data = []

    # Get all sequences of length context_length
    for i in range(len(values) - context_length):
        data.append(values[i:i + context_length])

    # Convert to tensor format
    data = torch.tensor(data)

    # Generate train/valid split
    valid_set_length = round(data.shape[0] * 0.2)
    train_set_length = data.shape[0] - valid_set_length

    src_train = data[:train_set_length,:-1]
    tgt_train = data[:train_set_length,-1]
    src_valid = data[-valid_set_length:,:-1]
    tgt_valid = data[-valid_set_length:,-1]

    return src_train, tgt_train, src_valid, tgt_valid


# Load and prepare dataset
with open("vt.csv") as file:
    csv_reader = csv.reader(file)
    data = {
        "date": [],
        "value": [],
    }
    next(csv_reader)
    for row in csv_reader:
        data["date"].append(row[0])
        data["value"].append(float(row[4]))

dataset = pd.DataFrame(data)
src_train, tgt_train, src_valid, tgt_valid = load(dataset, settings["context_length"])

# Initialize the model
model = StockPredictor()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])

# Train the model
for t in range(settings["epochs"]):
    batch_idx = random.randint(0, src_train.size(0) - settings["batch_size"])
    src_batch = src_train[batch_idx:batch_idx + settings["batch_size"],]
    tgt_batch = tgt_train[batch_idx:batch_idx + settings["batch_size"],]

    tgt_pred = model(src_batch)

    # Reshape to calculate loss
    batch, _ = tgt_pred.shape
    tgt_pred = tgt_pred.view(batch)
    loss = loss_fn(tgt_pred, tgt_batch)
    if t % 10 == 0 and t != 0:
        print("Epoch ", t, "MSE: ", loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Test the model
tgt_pred = model(src_valid)
batch, _= tgt_pred.shape
tgt_pred = tgt_pred.view(batch)
print("Predicted price", tgt_pred)
print("Actual price: ", tgt_valid)
