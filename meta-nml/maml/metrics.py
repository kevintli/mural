import numpy as np
from scipy.special import logsumexp, softmax

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def nll(logits, labels):
    probs = softmax(logits, axis=-1)
    log_probs = np.log(probs)
    return -np.array([log_probs[i, j] for i, j in enumerate(labels)])

def acc(logits, labels):
    predictions = np.argmax(logits, axis=-1)
    return (predictions == labels)

def brier(logits, labels):
    probs = softmax(logits, axis=-1)
    # one hot labels
    one_hot_labels = one_hot(labels, 10)
    return np.mean(np.sum(np.square(probs - one_hot_labels), axis=-1))

def comp(logits):
    return np.log(np.exp(logits).sum(axis=-1))

def entropy(logits):
    probs = softmax(logits, axis=-1)
    return -(probs * logits).sum(axis=-1)

def calibrations(logits, labels=None, n_buckets=20, uniform_buckets=False):
    probs = softmax(logits, axis=-1)
    predict_probs = np.max(probs, axis=-1)
    buckets = [[] for i in range(n_buckets)]
    sorted_confs = np.sort(predict_probs)
    index_separation = len(predict_probs) // n_buckets
    boundaries = [sorted_confs[i * index_separation] for i in range(n_buckets)]

    for j, p in enumerate(predict_probs):
        if not uniform_buckets:
            for index, boundary in enumerate(boundaries):
                if index == n_buckets - 1 or p < boundaries[index+1]:
                    buckets[index].append(j)
                    break
        else:
            index = int(p * 100)  // 5
            if index == 20:
                index = 19
        buckets[index].append(j)

    if labels is not None:
        accs = []
        for j, bucket_idxs in enumerate(buckets):
            idxs = np.array(bucket_idxs)
            if len(idxs) > 0:
                bucket_acc = sum(np.argmax(logits[idxs], axis=-1) == labels[idxs].numpy()) / len(idxs)
                accs.append(bucket_acc)
            else:
                accs.append(0)
        return buckets, accs, predict_probs

    return buckets, predict_probs

def ece(logits, labels, accs, n_buckets=20):
    buckets, confs = calibrations(logits, n_buckets=n_buckets)
    results = []
    count = 0
    sum_ce = 0.
    mean_confs = []
    for i, bucket in enumerate(buckets):
        count += len(bucket)
        if len(bucket) == 0:
            pass
            results.append(0)
            mean_confs.append(0)
        else:
            total = sum([accs[j] for j in bucket])
            total_conf = sum([confs[j] for j in bucket])
            mean_acc = total / len(bucket)
            mean_conf = total_conf / len(bucket)
            results.append(mean_acc)
            mean_confs.append(mean_conf)
            # print(i, len(bucket), mean_acc, mean_conf)
            sum_ce += len(bucket) * np.abs(mean_conf - mean_acc)
            # print(sum_ce)
    results = np.array(results)
    mean_confs = np.array(mean_confs) # linestyle='--', marker='o'
    return sum_ce / count
