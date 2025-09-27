import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
from datetime import datetime
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

    # Set up MLflow tracking
    mlflow.set_tracking_uri(uri=args.mlflow_tracking_uri)
    mlflow.set_experiment("SimpleCNN_Image_Classification")
    run_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"SimpleCNN_{run_date}"):
        # Log parameters
        mlflow.log_params({
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "device": str(device),
            "image_size": "224x224",
            "optimizer": "Adam",
            "model_type": "SimpleCNN",
            "num_classes": len(train_dataset.classes),
            "dataset_size": len(train_dataset)
        })
        
        # Log dataset class names
        mlflow.log_param("class_names", train_dataset.classes)
        
        # Training loop
        best_loss = float('inf')
        best_model_state = None
        
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, labels) in enumerate(train_loader):
                # Move data to the selected device
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Log batch metrics every 10 batches
                if i % 10 == 0:
                    batch_loss = loss.item()
                    batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
                    print(f"Epoch {epoch+1}/{args.epochs}, Batch {i}, Loss: {batch_loss:.4f}, Acc: {batch_acc:.2f}%")
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            
            # Log epoch metrics
            mlflow.log_metrics({
                "epoch_loss": epoch_loss,
                "epoch_accuracy": epoch_acc,
                "learning_rate": args.learning_rate  # Track LR in case of scheduling
            }, step=epoch)
            
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
            
            # Track best model (save state dict in memory)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                # Store the best model state dict in memory
                best_model_state = model.state_dict().copy()
                
                # Log best model metrics
                mlflow.log_metrics({
                    "best_loss": best_loss,
                    "best_accuracy": epoch_acc,
                    "best_epoch": epoch
                })
        
        # Load the best model state for final logging
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Create a sample input for model signature inference
        model.eval()  # Set to evaluation mode for inference
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 224, 224).to(device)
            sample_output = model(sample_input)
            
            # Infer signature for MLflow model logging
            signature = infer_signature(
                sample_input.cpu().numpy(),
                sample_output.cpu().numpy()
            )
        
        # Log the final model to MLflow (best model based on loss)
        mlflow.pytorch.log_model(
            model,
            "model",
            signature=signature,
            input_example=sample_input.cpu().numpy(),
            registered_model_name=f"SimpleCNN_{run_date}" if args.register_model else None
        )
        
        # Log final training summary
        mlflow.log_metrics({
            "final_loss": epoch_loss,
            "final_accuracy": epoch_acc,
            "total_epochs": args.epochs,
            "total_batches": len(train_loader) * args.epochs
        })
        
        print(f"Training completed. Best loss: {best_loss:.4f}")
        print(f"Model logged to MLflow with run ID: {mlflow.active_run().info.run_id}")
        
        if args.register_model:
            print(f"Model registered in MLflow Model Registry as: SimpleCNN_{run_date}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimpleCNN with MLflow tracking")
    parser.add_argument("--data-path", type=str, required=True, 
                       help="Path to the training image data directory.")
    parser.add_argument("--learning-rate", type=float, default=0.001, 
                       help="Learning rate for the optimizer.")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, 
                       help="Number of training epochs.")
    parser.add_argument("--mlflow-tracking-uri", type=str, 
                       default="your_mlflow_tracking_uri",
                       help="MLflow tracking server URI.")
    parser.add_argument("--register-model", action="store_true",
                       help="Register the trained model in MLflow Model Registry.")
    
    args = parser.parse_args()
    train_model(args)