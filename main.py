import argparse, os, sys
import torch
from models import ProbabilisticCNN
from src.make_dataset import download_videos_from_s3, compute_and_cache_features
from src.data_loader import create_hybrid_data_loaders
from src.trainer import train_hybrid_system
from src.evaluate import evaluate_probabilistic_model

def main(args):

    # 1. Get the absolute path of the folder containing main.py
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 2. Change the working directory to that folder
    os.chdir(project_root)

    # 3. Add this folder to Python's import path (so 'import src' works)
    if project_root not in sys.path:
        sys.path.append(project_root)

    print(f"Working Directory set to: {os.getcwd()}")
    # --- Prepare Data ---
    print("Downloading video and preprocessing data...")
    current_working_dir = os.getcwd()
    download_dir = os.path.join(current_working_dir, 'data', 'raw')
    print(f"Download dir: {download_dir}")
    if not os.path.exists(download_dir):
        download_videos_from_s3(download_dir)

    # --- CONFIGURATION ---

    VIDEO_DIR = download_dir    # Where your videos are
    CACHE_PATH= os.path.join(current_working_dir, 'data', 'processed', 'processed_pose_data.pt')
    MODEL_NAME = 'yolov8n-pose.pt'
    if not os.path.exists(CACHE_PATH):
        compute_and_cache_features(VIDEO_DIR, CACHE_PATH, MODEL_NAME)

    print("Loading data...")
    train_loader, test_loader = create_hybrid_data_loaders(CACHE_PATH, 
        batch_size=args.batch_size
    )

    # --- FUNCTION CALL 2: Initialize Model ---
    print("Initializing model...")
    model = ProbabilisticCNN()
    
    # --- FUNCTION CALL 3: Train or Evaluate ---
    if args.mode == 'train':
        print("Starting training...")

        model = train_hybrid_system(train_loader, test_loader, model, batch_size=args.batch_size)

        #train_model(model, train_loader, epochs=args.epochs)
        
        # Save the model after training
        torch.save(model.state_dict(), "pCCN_model.pth")
        
    elif args.mode == 'evaluate':
        print("Starting evaluation...")
        model.load_state_dict(torch.load("pCCN_model.pth"))
        evaluate_probabilistic_model(model, test_loader)

if __name__ == "__main__":
    # This block handles the command line arguments
    parser = argparse.ArgumentParser(description="Probabilistic Push-Up Counter")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    #parser.add_argument('--data_dir', type=str, default='./processed')
    
    args = parser.parse_args()
    
    # Kick off the main function
    main(args)