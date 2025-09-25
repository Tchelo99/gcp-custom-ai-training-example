import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleCNN

def train_model(args):
    """Trains a CNN model for image classification."""
    
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset
    try:
        # Assuming the data is structured with subdirectories for each class (e.g., 'cats', 'dogs')
        train_dataset = datasets.ImageFolder(root=args.data_path, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    except Exception as e:
        print(f"Error loading data from {args.data_path}: {e}")
        print("Please ensure your data is in a directory with subdirectories for each class (e.g., 'data/train/cats', 'data/train/dogs').")
        return

    # Initialize the model, loss function, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Move data to the selected device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}")

    # Save the model
    if args.model_dir:
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
        print(f"Model saved to {args.model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("AIP_MODEL_DIR"), help="Directory to save the model.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the training image data directory.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    args = parser.parse_args()
    train_model(args)