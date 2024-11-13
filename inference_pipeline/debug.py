"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it), 
    Davide Zoni (davide.zoni@polimi.it)
"""

import numpy as np


def errorCount(gt, predictions, stride):
    '''
    Returns the number of errors between the ground truth and the predictions.
    An error is defined as a CO found in the prediction but not in the ground truth, or vice versa.

    Parameters
    ----------
    gt : numpy.ndarray
        The ground truth CPs.
    predictions : numpy.ndarray
        The predictions.
    stride : int
        The stride used during Sliding Window Classification.

    Returns
    -------
    int
        The number of false positives, i.e., the number of CPs found in the predictions but not in the ground truth. 
    int
        The number of false negatives, i.e., the number of CPs found in the ground truth but not in the predictions.
    '''
    fp = len(falsePositives(gt, predictions, stride))
    fn = len(falseNegatives(gt, predictions, stride))
    return fp, fn


def errorRate(gt, predictions, stride):
    '''
    Returns the error rate between the ground truth and the predictions.
    The error rate is defined as the number of errors divided by the number of CPs in the ground truth.
    '''
    fp, fn = errorCount(gt, predictions, stride)
    return fp / len(gt), fn / len(gt)


def _findClosestGt(gt, predictions):
    '''
    Returns the closest ground truth CPs to the predictions.
    '''
    closest_gt = []
    for pred in predictions:
        distances = np.linalg.norm(gt - pred.reshape(1, -1), axis=0)
        closest_gt.append(gt[np.argmin(distances)])

    return closest_gt


def errorDistance(gt, predictions, stride, relative='gt'):
    '''
    Returns the error distance between the ground truth and the predictions.
    The error distance is defined as the Euclidean distance between the closest ground truth CP and the prediction.

    Parameters
    ----------
    gt : numpy.ndarray
        The ground truth CPs.
    predictions : numpy.ndarray
        The predictions.
    stride : int
        The stride used during Sliding Window Classification.
    relative : str, optional
        The reference frame to compute the error distance (default is 'gt').
        If 'gt', the error distance is computed in the ground truth reference frame.
        If 'pred', the error distance is computed in the predictions reference frame.

    Returns
    -------
    list
        The error distances for each prediction from the ground truth.
    '''
    if relative == 'pred':
        gt = gt//stride
    elif relative == 'gt':
        predictions = predictions*stride
    else:
        raise ValueError("relative must be 'gt' or 'pred'")

    closest_gt = _findClosestGt(gt, predictions)
    errors = []
    for pred, gt_ in zip(predictions, closest_gt):
        error = gt_ - pred
        errors.append(error)
    return errors


def __nonUniqueClosestGt(gt):
    for i, x in enumerate(gt):
        if gt[:i].count(x) == 0 and gt[i+1:].count(x) > 0:
            yield i, gt.count(x)


def falsePositives(gt, predictions, stride):
    '''
    Returns the number of false positive corresponding to non-true CPs.
    An error is defined as a CP found in the prediction but not in the ground truth.

    Parameters
    ----------
    gt : numpy.ndarray
        The ground truth CPs.
    predictions : numpy.ndarray
        The predictions.
    stride : int
        The stride used during Sliding Window Classification.

    Returns
    -------
    The false positives.
    '''

    false_positives = []
    closest_gt = _findClosestGt(gt//stride, predictions)
    errors = np.abs(errorDistance(gt, predictions,
                    stride=stride, relative='pred'))

    non_unique_closest_gt = __nonUniqueClosestGt(closest_gt)

    for i in non_unique_closest_gt:
        idx, num = i
        min_error = min(errors[idx:idx+num])
        for j in range(num):
            if errors[idx+j] > min_error:
                false_positives.append(predictions[idx+j])

    return false_positives


def falseNegatives(gt, predictions, stride):
    '''
    Returns the number of false negative corresponding to true CPs.
    An error is defined as a CP found in the ground truth but not in the predictions.

    Parameters
    ----------
    gt : numpy.ndarray
        The ground truth CPs.
    predictions : numpy.ndarray
        The predictions.
    stride : int
        The stride used during Sliding Window Classification.

    Returns
    -------
    The false negatives.
    '''

    false_negatives = []
    closest_gt = _findClosestGt(gt//stride, predictions)

    false_negatives = np.setdiff1d(gt//stride, closest_gt)

    return false_negatives
