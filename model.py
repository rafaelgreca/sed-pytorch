import torch
import torch.nn as nn
import os
from typing import Tuple, List


class TimeDistributed(nn.Module):
    """
    Mimics the Keras TimeDistributed layer.

    All credits to: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346
    """

    def __init__(
        self, module: torch.nn.Module, batch_first: bool, layer_name: str
    ) -> None:
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
        self.layer_name = layer_name

    def forward(self, x: torch.tensor) -> torch.tensor:
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-3), x.size(-2), x.size(-1))

        y = self.module(x_reshape)

        if self.layer_name == "convolutional" or self.layer_name == "max_pooling":
            # We have to reshape Y
            if self.batch_first:
                y = y.contiguous().view(
                    x.size(0), x.size(1), y.size(-3), y.size(-2), y.size(-1)
                )
            else:
                y = y.view(-1, x.size(1), y.size(-1))  # FIXME
        else:
            # We have to reshape Y
            if self.batch_first:
                y = y.contiguous().view(x.size(0), x.size(1), y.size(-1))
            else:
                y = y.view(-1, x.size(1), y.size(-1))  # FIXME

        return y


class Extract_GRU_Output(nn.Module):
    """
    Extracts only the output from the BiLSTM layer.
    """

    def forward(self, x: torch.tensor) -> torch.tensor:
        output, _ = x
        return output


class SED(nn.Module):
    """
    The Convolutional Recurrent Neural Network (CRNN) model created for
    the Sound Event Detection (SED) task.
    """

    def __init__(self) -> None:
        super(SED, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(
                in_channels=40, out_channels=128, kernel_size=(3, 3), padding="same"
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 5)),
            nn.Dropout(p=0.5),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )
        self.temporal_processing = nn.Sequential(
            nn.GRU(
                input_size=6,
                hidden_size=32,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            ),
            Extract_GRU_Output(),
            nn.GRU(
                input_size=64,
                hidden_size=32,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            ),
            Extract_GRU_Output(),
        )
        self.classification = nn.Sequential(
            TimeDistributed(
                nn.Linear(in_features=64, out_features=16),
                batch_first=True,
                layer_name="linear",
            ),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            TimeDistributed(
                nn.Linear(in_features=16, out_features=6),
                batch_first=True,
                layer_name="linear",
            ),
            nn.Sigmoid(),
        )
        self.feature_extraction.apply(self.init_weights)
        self.temporal_processing.apply(self.init_weights)
        self.classification.apply(self.init_weights)

    def init_weights(self, m: nn.Module) -> nn.Module:
        """
        Initalize all the weights in the PyTorch model to be the same as Keras.
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        if isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
            nn.init.xavier_uniform_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)
            nn.init.zeros_(m.bias_ih_l0)
            nn.init.zeros_(m.bias_hh_l0)

    def forward(self, X: torch.tensor) -> torch.tensor:
        X = self.feature_extraction(X)
        X = X.permute(0, 3, 2, 1)
        X = X.view(X.shape[0], 256, -1)
        X = self.temporal_processing(X)
        X = self.classification(X)
        return X


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, output_dir: str, model_name: str) -> None:
        self.best_valid_er = 1
        self.best_valid_f1 = 0.0
        self.output_dir = output_dir
        self.model_name = model_name
        self.best_folds_f1 = [0.0, 0.0, 0.0, 0.0]
        self.best_folds_er = [1.0, 1.0, 1.0, 1.0]

    def __call__(
        self,
        current_valid_er: float,
        current_valid_f1: float,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim,
        fold: int,
    ) -> None:
        if current_valid_er < self.best_valid_er:
            self.best_valid_er = current_valid_er
            self.best_valid_f1 = current_valid_f1
            print("\nSaving model...")
            print(f"Epoch: {epoch}")
            print(f"Validation F1: {current_valid_f1:1.6f}")
            print(f"Validation ER: {current_valid_er:1.6f}\n")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(self.output_dir, f"{self.model_name}_fold{fold}.pth"),
            )
            self.best_folds_f1[fold-1] = current_valid_f1
            self.best_folds_er[fold-1] = current_valid_er

    def get_best_metrics(self) -> Tuple[List, List]:
        return self.best_folds_f1, self.best_folds_er


class EarlyStopping:
    """
    The Early Stopping class (used to avoid overfitting during training).
    """

    def __init__(self, tolerance: int, delta: float = 0.0) -> None:
        self.tolerance = tolerance
        self.early_stop = False
        self.counter = 0
        self.delta = delta
        self.best_er = None

    def __call__(self, validation_er) -> None:
        if self.best_er is None:
            self.best_er = validation_er
        elif validation_er < self.best_er:
            self.counter += 1

            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.best_er = validation_er
            self.counter = 0
