import numpy as np
from typing import Dict

#####################
# Scoring functions
#
# Code blocks taken from Toni Heittola's repository: http://tut-arg.github.io/sed_eval/
#
# Implementation of the Metrics in the following paper:
# Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen, 'Metrics for polyphonic sound event detection',
# Applied Sciences, 6(6):162, 2016
#####################


def f1_overall_framewise(O: np.ndarray, T: np.ndarray):
    """
    Auxiliary function to calculate the F1 Score (F1).

    Args:
        O (np.ndarray): the output array.
        T (np.ndarray): the target array.

    Returns:
        float: the F1 Score value.
    """
    if len(O.shape) == 3:
        O = O.reshape(O.shape[0] * O.shape[1], O.shape[2])
        T = T.reshape(T.shape[0] * T.shape[1], T.shape[2])

    TP = ((2 * T - O) == 1).sum()
    Nref, Nsys = T.sum(), O.sum()

    prec = float(TP) / float(Nsys + np.finfo(np.float).eps)
    recall = float(TP) / float(Nref + np.finfo(np.float).eps)
    f1_score = 2 * prec * recall / (prec + recall + np.finfo(np.float).eps)
    return f1_score


def er_overall_framewise(O: np.ndarray, T: np.ndarray) -> float:
    """
    Auxiliary function to calculate the Error Rate (ER).

    Args:
        O (np.ndarray): the output array.
        T (np.ndarray): the target array.

    Returns:
        float: the Error Rate value.
    """
    if len(O.shape) == 3:
        O = O.reshape(O.shape[0] * O.shape[1], O.shape[2])
        T = T.reshape(T.shape[0] * T.shape[1], T.shape[2])

    FP = np.logical_and(T == 0, O == 1).sum(1)
    FN = np.logical_and(T == 1, O == 0).sum(1)

    S = np.minimum(FP, FN).sum()
    D = np.maximum(0, FN - FP).sum()
    I = np.maximum(0, FP - FN).sum()

    Nref = T.sum()
    ER = (S + D + I) / (Nref + 0.0)
    return ER


def f1_overall_1sec(O: np.ndarray, T: np.ndarray, block_size: int) -> float:
    """
    Computes the F1 Score.

    Args:
        O (np.ndarray): the output array.
        T (np.ndarray): the target array.
        block_size (int): the block size (total frames in one second).

    Returns:
        float: the F1 Score.
    """
    if len(O.shape) == 3:
        O = O.reshape(O.shape[0] * O.shape[1], O.shape[2])
        T = T.reshape(T.shape[0] * T.shape[1], T.shape[2])

    new_size = int(np.ceil(O.shape[0] / block_size))
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i, :] = np.max(
            O[
                int(i * block_size) : int(i * block_size + block_size - 1),
            ],
            axis=0,
        )
        T_block[i, :] = np.max(
            T[
                int(i * block_size) : int(i * block_size + block_size - 1),
            ],
            axis=0,
        )
    return f1_overall_framewise(O_block, T_block)


def er_overall_1sec(O: np.ndarray, T: np.ndarray, block_size: int) -> float:
    """
    Computes the Error Rate.

    Args:
        O (np.ndarray): the output array.
        T (np.ndarray): the target array.
        block_size (int): the block size (total frames in one second).

    Returns:
        float: the Error Rate.
    """
    if len(O.shape) == 3:
        O = O.reshape(O.shape[0] * O.shape[1], O.shape[2])
        T = T.reshape(T.shape[0] * T.shape[1], T.shape[2])

    new_size = int(O.shape[0] / block_size)
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i, :] = np.max(
            O[
                int(i * block_size) : int(i * block_size + block_size - 1),
            ],
            axis=0,
        )
        T_block[i, :] = np.max(
            T[
                int(i * block_size) : int(i * block_size + block_size - 1),
            ],
            axis=0,
        )
    return er_overall_framewise(O_block, T_block)


def compute_scores(pred: np.ndarray, y: np.ndarray, frames_in_1_sec: int = 50) -> Dict:
    """
    Compute the scores (F1 Score and Error Rate).

    Args:
        pred (np.ndarray): the prediction array.
        y (np.ndarray): the target array.
        frames_in_1_sec (int, optional): the total frames in one second. Defaults to 50.

    Returns:
        Dict: the F1 Score and Error Rate.
    """
    scores = dict()
    scores["f1_overall_1sec"] = f1_overall_1sec(pred, y, frames_in_1_sec)
    scores["er_overall_1sec"] = er_overall_1sec(pred, y, frames_in_1_sec)
    return scores
