import argparse
import os
import utils
import numpy as np
import torch
import torch.nn as nn
import metrics
from torch.utils.data import DataLoader
from typing import Tuple
from dataset import Tut_Dataset
from model import SED, SaveBestModel
from sklearn.preprocessing import StandardScaler

def train(
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim,
    frames_per_second: int,
    model: torch.nn.Module,
    loss: torch.nn.BCELoss,
    threshold: float
) -> Tuple[float, float]:
    """
    Function responsible for the training step.
    Args:
        device (torch.device): the device (cpu or cuda).
        train_loader (torch.utils.data.DataLoader): the training dataloader.
        optimizer (torch.nn.optim): the optimizer that will be used.
        epoch (int): the current epoch.
        model (torch.nn.Module): the Voxseg module.
    Returns:
        Tuple[float, float]: current epoch training loss and accuracy.
    """
    model.train()
    training_er = 0
    training_f1 = 0

    for data, target in train_loader:
        data = data.unsqueeze(1)
        data = data.permute(0, 3, 1, 2)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        _output = output.permute(0, 2, 1)
        _target = target.permute(0, 2, 1)

        l = loss(_output, _target)
        l.backward()
        optimizer.step()

        output = (output > threshold) * 1.0
        scores = metrics.compute_scores(pred=output.detach().numpy(),
                                        y=target.detach().numpy(),
                                        frames_in_1_sec=frames_per_second)

        training_er += scores["er_overall_1sec"]
        training_f1 += scores["f1_overall_1sec"]
    
    training_er /= len(train_loader)
    training_f1 /= len(train_loader)
    return training_er, training_f1

def evaluate(
    device: torch.device,
    validation_loader: DataLoader,
    model: torch.nn.Module,
    frames_per_second: int,
    threshold: float
) -> Tuple[float, float]:
    model.eval()
    validation_er = 0
    validation_f1 = 0
    
    with torch.no_grad():
        for data, target in validation_loader:
            data = data.unsqueeze(1)
            data = data.permute(0, 3, 1, 2)

            data, target = data.to(device), target.to(device)
            output = model(data)
            output = (output > threshold) * 1.0
            
            scores = metrics.compute_scores(pred=output.detach().numpy(),
                                            y=target.detach().numpy(),
                                            frames_in_1_sec=frames_per_second)
            validation_er += scores["er_overall_1sec"]
            validation_f1 += scores["f1_overall_1sec"]
    
    validation_er /= len(validation_loader)
    validation_f1 /= len(validation_loader)
    return validation_er, validation_f1
   
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--data_path", type=str, required=True)
    parse.add_argument("--output_dir", type=str, required=True)
    parse.add_argument("--sample_rate", type=int)
    parse.add_argument("--window_size", type=int)
    parse.add_argument("--n_fft", type=int)
    parse.add_argument("--hop_length", type=int)
    parse.add_argument("--n_mels", type=int)
    parse.add_argument("--threshold", type=float)
    parse.set_defaults(
        sample_rate=44100,
        window_size=256,
        n_fft=2048,
        n_mels=40,
        threshold=0.5
    )
    args = parse.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    _valid_sample_rates = [8000, 16000, 24000, 44000, 44100]
    assert os.path.exists(args.data_path) and os.path.exists(os.path.join(args.data_path, "evaluation_setup")) \
           and os.path.exists(os.path.join(args.data_path, "audio")) and \
           os.path.exists(os.path.join(args.data_path, "meta")), "Please use a valid data path"
    assert args.window_size > 0, "Please use a valid value for window size"
    assert args.n_mels > 0, "Please use a valid value for filter mels"
    assert args.n_fft > 0, "Please use a valid value for n_fft"
    assert args.sample_rate in _valid_sample_rates, f"Please use a valid value for sample rate ({_valid_sample_rates})"
    assert args.hop_length is None or args.hop_length > 0, "Please use a valid value for hop_length"
    
    if args.hop_length is None:
        args.hop_length = args.n_fft // 2

    _folds = [1, 2, 3, 4]
    _frames_per_second = int(args.sample_rate/(args.n_fft/2.0))

    for fold in _folds:
        scaler = StandardScaler()
        print("\n--------------------------------")
        print(f"FOLD {fold}")
        print("--------------------------------\n")

        _files_path = os.path.join(args.data_path, "evaluation_setup")
        _train_path = os.path.join(_files_path, f"street_fold{fold}_train.txt")
        _evaluate_path = os.path.join(_files_path, f"street_fold{fold}_evaluate.txt")
        _test_path = os.path.join(_files_path, f"street_fold{fold}_test.txt")
        _meta_folder_path = os.path.join(args.data_path, "meta")
        
        print("Extracting training features...")       
        train_df = utils.open_file(path=_train_path,
                                   meta_path=_meta_folder_path)
        
        train_features, encoded_matrix_train = utils.extract_features(path=args.data_path,
                                                                      sample_rate=args.sample_rate,
                                                                      n_fft=args.n_fft,
                                                                      hop_length=args.hop_length,
                                                                      n_mels=args.n_mels,
                                                                      df=train_df,
                                                                      window_size=args.window_size)
        
        train_features = train_features.astype(np.float32)
        encoded_matrix_train = encoded_matrix_train.astype(np.float32)
        
        # Scaling the training features
        train_features = train_features.reshape(-1, train_features.shape[-1])
        train_features = scaler.fit_transform(train_features)
        train_features = train_features.reshape(-1, args.window_size, train_features.shape[-1])
        
        train_dataset = Tut_Dataset(X=train_features,
                                    y=encoded_matrix_train)
        
        train_dataloader = utils.create_dataloader(dataset=train_dataset,
                                                   batch_size=128)
        
        print("Extracting evaluation features...") 
        evaluate_df = utils.open_file(path=_evaluate_path,
                                      meta_path=_meta_folder_path)
        
        evaluate_features, encoded_matrix_evaluate = utils.extract_features(path=args.data_path,
                                                                            sample_rate=args.sample_rate,
                                                                            n_fft=args.n_fft,
                                                                            hop_length=args.hop_length,
                                                                            n_mels=args.n_mels,
                                                                            df=evaluate_df,
                                                                            window_size=args.window_size)
        
        evaluate_features = evaluate_features.astype(np.float32)
        encoded_matrix_evaluate = encoded_matrix_evaluate.astype(np.float32)
        
        # Scaling the evaluation features
        evaluate_features = evaluate_features.reshape(-1, evaluate_features.shape[-1])
        evaluate_features = scaler.transform(evaluate_features)
        evaluate_features = evaluate_features.reshape(-1, args.window_size, evaluate_features.shape[-1])
        
        evaluate_dataset = Tut_Dataset(X=evaluate_features,
                                       y=encoded_matrix_evaluate)
        
        evaluate_dataloader = utils.create_dataloader(dataset=evaluate_dataset,
                                                      batch_size=128)
        
        print("Extracting test features...") 
        test_df = utils.open_file(path=_test_path,
                                  meta_path=_meta_folder_path,
                                  is_test=True)
        
        test_features, encoded_matrix_test = utils.extract_features(path=args.data_path,
                                                                    sample_rate=args.sample_rate,
                                                                    n_fft=args.n_fft,
                                                                    hop_length=args.hop_length,
                                                                    n_mels=args.n_mels,
                                                                    df=test_df,
                                                                    window_size=args.window_size)
        
        test_features = test_features.astype(np.float32)
        encoded_matrix_test = encoded_matrix_test.astype(np.float32)
        
        # Scaling the test features
        test_features = test_features.reshape(-1, test_features.shape[-1])
        test_features = scaler.transform(test_features)
        test_features = test_features.reshape(-1, args.window_size, test_features.shape[-1])
        
        test_dataset = Tut_Dataset(X=test_features,
                                   y=encoded_matrix_test)
        
        test_dataloader = utils.create_dataloader(dataset=test_dataset,
                                                  batch_size=128)
        
        # Create the model
        epochs = 150
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SED()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-03, eps=1e-07)
        loss = nn.BCELoss()
        save_best_model = SaveBestModel(output_dir=args.output_dir,
                                        model_name=f"best_model_fold{fold}")
        
        print("\nTraining the model...\n")

        # Training loop
        for epoch in range(1, epochs + 1):
            train_er, train_f1 = train(
                device=device,
                train_loader=train_dataloader,
                optimizer=optimizer,
                frames_per_second=_frames_per_second,
                model=model,
                loss=loss,
                threshold=args.threshold
            )

            validation_er, validation_f1 = evaluate(
                device=device,
                validation_loader=evaluate_dataloader,
                model=model,
                frames_per_second=_frames_per_second,
                threshold=args.threshold
            )

            print(f"Train Epoch: {epoch}/{epochs}")
            
            save_best_model(
                current_valid_er=validation_er,
                current_valid_f1=validation_f1,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
            )