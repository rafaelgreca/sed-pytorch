import os
import pandas as pd
import numpy as np
import librosa
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
from glob import glob


def _mel_spectogram(
    audio: np.ndarray, sample_rate: int, n_fft: int, hop_length: int, n_mels: int
) -> np.ndarray:
    """
    Extracts the mel spectogram.

    Args:
        audio (np.ndarray): the audio data array.
        sample_rate (int): the sample rate of the audio.
        n_fft (int): the number of fft.
        hop_length (int): the hop length.
        n_mels (int): the number of mels.

    Returns:
        np.ndarray: the mel spectogram.
    """
    spectogram, _ = librosa.core.spectrum._spectrogram(
        y=audio, n_fft=n_fft, hop_length=hop_length, power=1
    )
    mel = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    return np.log(np.dot(mel, spectogram))


def _encoded_matrix(
    features: np.ndarray, sample_rate: int, hop_length: int, df: pd.DataFrame
) -> np.ndarray:
    """_summary_

    Args:
        features (np.ndarray): the features array.
        sample_rate (int): the sample rate of the audio.
        hop_length (int): the hop length.
        df (pd.DataFrame): the dataframe with the events timestamps
                           (created by the open_file function).

    Returns:
        np.ndarray: the encoded matrix representation.
    """
    map_events = {
        "brakes squeaking": 0,
        "car": 1,
        "children": 2,
        "large vehicle": 3,
        "people speaking": 4,
        "people walking": 5,
    }
    encoded_matrix = np.zeros((features.shape[0], len(map_events)))
    unique_events = df["label"].unique().tolist()

    for event in unique_events:
        temp_df = df[df["label"] == event].reset_index(drop=True)

        event_starts = np.array(
            [(s * sample_rate) / hop_length for s in temp_df["start"].tolist()]
        )
        event_starts = np.floor(event_starts).astype(int)

        event_ends = np.array(
            [(e * sample_rate) / hop_length for e in temp_df["end"].tolist()]
        )
        event_ends = np.ceil(event_ends).astype(int)

        for (s, e) in zip(event_starts, event_ends):
            encoded_matrix[s : e + 1, map_events[event]] = 1

    return encoded_matrix


def extract_features(
    path: str,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    df: pd.DataFrame,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts features from the audio (log mel energy) and creates the
    encoded matrix with the targets timestamps.

    Args:
        path (str): the root folder's directory path.
        sample_rate (int): the sample rate of the audio.
        n_fft (int): number of fft (fast fourier transform).
        hop_length (int): the hop length.
        n_mels (int): number of mels.
        df (pd.DataFrame): the dataframe returned by the open_file function.
        window_size (int): the size of the window.

    Returns:
        Tuple[np.ndarray, np.ndarray]: the log mel energy features array and
                                       the encoded matrix representation.
    """
    unique_files = df["file_path"].unique().tolist()
    mels = np.array([])
    encoded_matrixes = np.array([])

    for file in unique_files:
        temp_df = df[df["file_path"] == file].reset_index(drop=True)
        audio, sr = librosa.load(os.path.join(path, file), mono=True, sr=None)

        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

        mel = _mel_spectogram(
            audio=audio,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        ).T

        if mels.size == 0:
            mels = mel
        else:
            mels = np.concatenate([mels, mel], axis=0)

        matrix = _encoded_matrix(
            features=mel, sample_rate=sample_rate, hop_length=hop_length, df=temp_df
        )

        if encoded_matrixes.size == 0:
            encoded_matrixes = matrix
        else:
            encoded_matrixes = np.concatenate([encoded_matrixes, matrix], axis=0)

    mels = _split_in_windows(features=mels, window_size=window_size)
    encoded_matrixes = _split_in_windows(
        features=encoded_matrixes, window_size=window_size
    )

    return mels, encoded_matrixes


def _split_in_windows(features: np.ndarray, window_size: int) -> np.ndarray:
    """
    Split the features into windows.
    E.g: window_size = 256
         (25600, 40) -> (100, 256, 40)

    Args:
        features (np.ndarray): the features (or encoded matrix) array.
        window_size (int): the size of the window.

    Returns:
        np.ndarray: the reshaped features (or encoded matrix) array.
    """
    if features.shape[0] % window_size:
        features = features[: -(features.shape[0] % window_size), :]

    features = features.reshape(
        (features.shape[0] // window_size, window_size, features.shape[1])
    )
    return features


def open_file(path: str, meta_path: str, is_test: bool = False) -> pd.DataFrame:
    """
    Reads the training/validation/test txt file (inside the evaluation_setup) folder
    for a specific fold.

    Args:
        path (str): the fold folder's (evaluation_setup) directory path.
        meta_path (str): the meta folder's directory path.
        is_test (bool): boolean flag to indicate if it's the test file.

    Returns:
        pd.DataFrame: _description_
    """
    if not is_test:
        df = pd.read_csv(path, header=None, sep="\t")
        df = df.drop(columns=[1])
        df = df.rename(columns={0: "file_path", 2: "start", 3: "end", 4: "label"})
        df = df.drop_duplicates()
        df = df.dropna().reset_index(drop=True)
    else:
        meta_df = _read_meta_path(path=meta_path)
        df = pd.read_csv(path, header=None, sep="\t")
        _unique_files = df[0].unique().tolist()

        df = meta_df[meta_df["file_path"].isin(_unique_files)]
        df = df.reset_index(drop=True)

    return df


def _read_meta_path(path: str) -> pd.DataFrame:
    """
    Reads all the annotation files inside the meta folder.

    Args:
        path (str): the meta folder's directory path.

    Returns:
        pd.DataFrame: the dataframe containing all the annotations.
    """
    annotation_files = glob(path + "/**/*.ann", recursive=True)
    annotation_df = pd.DataFrame()

    for annotation in annotation_files:
        temp_df = pd.read_csv(annotation, header=None, sep="\t")
        annotation_df = pd.concat([annotation_df, temp_df], axis=0)

    annotation_df = annotation_df.drop(columns=[1, 5, 6])
    annotation_df = annotation_df.reset_index(drop=True)
    annotation_df = annotation_df.rename(
        columns={0: "file_path", 2: "start", 3: "end", 4: "label"}
    )

    return annotation_df


def create_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    """
    Creates a new DataLoader.

    Args:
        dataset (Dataset): the custom dataset in the PyTorch format.
        batch_size (int): the batch size.

    Returns:
        DataLoader: the new DataLoader that will be used to
                    train/validate/test the model.
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    return dataloader
