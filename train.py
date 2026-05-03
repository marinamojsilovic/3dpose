import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from model import *
from posedataset import PoseDataset

batch_size = 32
learning_rate = 0.001
epochs = 100

def train_one_epoch(model, optimizer, dataloader, device):
    total_steps = len(dataloader) 
    running_loss = 0.0

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        # forward
        preds = model(x)
        loss = compute_mpjpe(preds, y)

        # backward
        optimizer.zero_grad()
        #compute gradients
        loss.backward()
        #update weights
        optimizer.step()

        running_loss += loss.item()

        if (i+1) % 1000 == 0:
            print(f"Step {i+1}/{total_steps}, loss: {loss.item()}")
            print(f"Loss: {running_loss/i}")

    return running_loss / total_steps


def evaluate(model, dataloader, device):
    loss_acc = 0
    steps = len(dataloader)
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            loss = compute_mpjpe(preds, y)

            loss_acc += loss.item()

    return loss_acc / steps


if __name__ == "__main__":
    #device
    device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
    )
    
    print(f"Using device: {device}")

    print("Loading data...")
    #dataset
    train_dataset = PoseDataset(TRAIN_SUBJECTS)
    val_dataset = PoseDataset(VAL_SUBJECTS)
    test_dataset = PoseDataset(TEST_SUBJECTS)

    print("Creating dataloaders...")
    #dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    print("Creating model...")
    #model
    model = PoseTransformer(N_LAYERS, VECTOR_SIZE, N_HEADS, DROPOUT).to(device)
    opt = optim.Adagrad(model.parameters(), lr = learning_rate)

    print("Training...")
    for epoch in range(epochs):
        #train
        train_loss = train_one_epoch(model, opt, train_dataloader, device)
        
        #eval
        val_loss = evaluate(model, val_dataloader, device)

        if (epoch+1) % 10 == 0:
             print(f"Epoch {epoch} | train loss: {train_loss} | val loss: {val_loss}")

    #test
    print("Testing...")
    test_loss = evaluate(model, test_dataloader, device)
    print(f"Test loss: {test_loss}")