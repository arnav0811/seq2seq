import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data_loader import ConalaDataset, collate
from seq2seq_model import Seq2SeqModel
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        intents = batch['intents'].to(device)
        snippets = batch['snippets'].to(device)
        intent_lengths = batch['intent_lengths']
        snippet_lengths = batch['snippet_lengths']

        # Forward pass
        outputs = model(intents, snippets, intent_lengths, snippet_lengths)

        # Ignore the <SOS> token that begins and then calculate loss
        targets = snippets[:, 1:].contiguous().view(-1)
        outputs = outputs[:, :-1].contiguous().view(-1, outputs.size(-1))
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Prevents exploding gradients - from bahdanau paper
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            intents = batch['intents'].to(device)
            snippets = batch['snippets'].to(device)
            intent_lengths = batch['intent_lengths']
            snippet_lengths = batch['snippet_lengths']

            outputs = model(intents, snippets, intent_lengths, snippet_lengths)

            targets = snippets[:, 1:].contiguous().view(-1)
            outputs = outputs[:, :-1].contiguous().view(-1, outputs.size(-1))
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required = True, help = 'Path to CoNaLa JSON file')
    parser.add_argument('--use_attention', action='store_true', help = 'Use attention mechanism')
    parser.add_argument('--epochs', type=int, default=8, help = 'Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help = 'Batch size')
    parser.add_argument('--lr', type=float, default=0.7, help = 'Initial learning rate ')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading dataset")
    dataset = ConalaDataset(args.data_path)

    # Train split
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle = True, collate_fn = collate)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle = False, collate_fn = collate)

    model = Seq2SeqModel(vocab_size = len(dataset.vocab), use_attention = args.use_attention).to(device)

    print("Using attention" if args.use_attention else "Not using attention")
    print(f"Initial LR: {args.lr}")

    criterion = nn.CrossEntropyLoss(ignore_index = 0)
    optimizer = optim.SGD(model.parameters(), lr = args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    train_losses = []
    validation_losses = []
    learning_rates = []
    best_validation_loss = float('inf')

    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{args.epochs} | Learning Rate: {current_lr:.4f}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        validation_loss = validate(model, validation_loader, criterion, device)
        
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        learning_rates.append(current_lr)

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}")

        if epoch >= 4:  
            scheduler.step()

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            model_name = 'saved_models/model_with_attention.pth' if args.use_attention else 'saved_models/model_without_attention.pth'
            torch.save(model.state_dict(), model_name)
            print(f"New model saved as {model_name}")

    # Plot with learning rate info
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curves
    ax1.plot(train_losses, label='Training Loss', marker='o')
    ax1.plot(validation_losses, label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Curves {"(with Attention)" if args.use_attention else "(Baseline - Without Attention)"}')
    ax1.legend()
    ax1.grid(True)
    
    # Learning rate schedule
    ax2.plot(learning_rates, label='Learning Rate', marker='d', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale('log')  # Log scale for LR

    plot_name = 'training_curves_attention.png' if args.use_attention else 'training_curves_baseline.png'
    plt.tight_layout()
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Best validation loss: {best_validation_loss:.4f}")

if __name__ == "__main__":
    main()