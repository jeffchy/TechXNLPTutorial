import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


def pad_dataset(query, seq_max_len, pad_idx):

    lengths = []
    new_query = []
    new_query_inverse = []
    for q in query:
        length = len(q)
        if length <= 0:
            continue
        if length > seq_max_len:
            q = q[: seq_max_len]
            length = seq_max_len
        else:
            remain = seq_max_len - length
            remain_arr = np.repeat(pad_idx, remain)
            q = np.concatenate((q, remain_arr))
            assert len(q) == seq_max_len

        new_query.append(q)
        lengths.append(length)

    return new_query, lengths


def eval_acc(sent_pred, sent_true):
    assert len(sent_pred) == len(sent_true)
    acc = accuracy_score(torch.LongTensor(sent_true), sent_pred)
    return acc


def flatten(input, length):
    """
    :param input: B x L x ?
    :param length: B
    :return:
    """

    B, L = input.size()[0], input.size()[1]
    # fraction = 1
    flattened = torch.cat([input[i,:length[i]] for i in range(B)], dim=0)

    return flattened


def eval_seq_token(seq_label_pred, seq_label_true, o_idx=0):
    """
    :param seq_label_pred: B x L
    :param seq_label_true: B x L
    :param seq_len: B
    :return:
    """
    assert len(seq_label_pred) == len(seq_label_true)

    correct = 0
    tp = 0
    fp = 0
    fn = 0

    for i in range(len(seq_label_pred)):
        sp = seq_label_pred[i]
        st = seq_label_true[i]

        if sp == st:
            correct += 1
            if sp != o_idx:
                tp += 1
        else:
            if sp != o_idx:
                fp += 1
            if st != o_idx:
                fn += 1


    all_tokens = len(seq_label_pred)
    accuracy = correct / all_tokens
    precision = tp / (tp + fp) if (tp + fp != 0) else 0
    recall = tp / (tp + fn) if (tp + fn != 0) else 0
    f1 = 2 * precision * recall / (precision + recall) if  (precision + recall != 0) else 0

    return accuracy, precision, recall, f1