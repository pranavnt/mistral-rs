from model import Transformer, Tokenizer, generate
from dataset import CodeTextDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import argparse
import os
import time

from model import Transformer, Tokenizer, generate
from dataset import CodeTextDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

import numpy as np
import argparse
import os
import time
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, required=True, help='Path to the JSONL data file.')
parser.add_argument('--model-path', type=str, required=True, help='Path to the model directory')
parser.add_argument('--model-save-path', type=str, default='./models', help='Path to save the trained model.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training.')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer.')
parser.add_argument('--max-seq-len', type=int, default=512, help='Maximum sequence length for the model.')
parser.add_argument('--log-interval', type=int, default=10, help='Interval for logging to TensorBoard.')
args = parser.parse_args()

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = Path(args.model_path)

model = Transformer.from_folder(path, 8192)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.to(device)

tokenizer = Tokenizer(folder / 'tokenizer.model')

# Prepare the dataset and dataloader
dataset = CodeTextDataset(args.data_path, args.tokenizer_path)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=CodeTextDataset.collate_fn)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# TensorBoard writer
writer = SummaryWriter(f'runs/{datetime.now().strftime("%b%d_%H-%M-%S")}')

# Training loop
for epoch in range(args.epochs):
    model.train()
    start_time = time.time()
    total_loss = 0
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        inputs, labels = batch[:,:-1], batch[:,1:].contiguous().view(-1)
        inputs, labels = inputs.to(device), labels.to(device)
        
        logits = model(inputs, torch.arange(0, args.max_seq_len).to(device))
        logits = logits.view(-1, logits.size(-1))
        
        loss = criterion(logits, labels)
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()
        if i % args.log_interval == 0 and i > 0:
            avg_loss = total_loss / args.log_interval
            writer.add_scalar('training_loss', avg_loss, epoch * len(dataloader) + i)
            total_loss = 0
    
    # Save the model after each epoch
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), os.path.join(args.model_save_path, f'model_epoch_{epoch}.pth'))
    else:
        torch.save(model.state_dict(), os.path.join(args.model_save_path, f'model_epoch_{epoch}.pth'))
    
    elapsed_time = time.time() - start_time
    print(f'Epoch {epoch} completed in {elapsed_time:.2f}s')

writer.close()
