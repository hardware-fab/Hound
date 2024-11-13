"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it), 
    Davide Zoni (davide.zoni@polimi.it)
"""

import numpy as np


def majorityFilter(array, kernel_size=None):
    """
    Apply a majority filter to a 1D array of integers.

    Parameters
    ----------
    array : np.ndarray
      The input array of interger. Value are limited from 0 to 2.
    kernel_size : int, optioanl
      The size of the window (default is None, which is equal to the length of the array).

    Returns
    -------
    The output array after applying the majority filter.
    """

    if kernel_size is None:
        value_counts = np.bincount(array)
        # Find the majority value (index of the maximum count)
        majority_value = np.argmax(value_counts)

        return np.full_like(array, majority_value)

    # Pad the array to handle edge cases
    pad_width = kernel_size // 2
    padded_array = np.pad(array, (pad_width, pad_width), mode='symmetric')

    output_array = np.zeros_like(array)

    for i in range(0, len(array)):
        window = padded_array[i:i + kernel_size]

        # Count the occurrences of each value
        value_counts = np.bincount(window)

        # Find the majority value (index of the maximum count)
        majority_value = np.argmax(value_counts)

        # Set the output element to the majority value
        output_array[i] = majority_value

    return output_array


def extractEdges(square_signal: np.ndarray,
                 min_distance: int,
                 first: bool) -> np.ndarray:
    """
    Get the starting sample for each CPs present in the square wave signal.

    Parameters
    ----------
    `square_signal` : np.ndarray
        The square wave signal.
    `min_distance` : int
        The minimum distance between two CPs.
    `first` : bool
        If True, the last edge is not detected.

    Returns
    -------
    The starting sample for each CPs present in the square wave signal.
    """

    edges = []
    prev_falling_edge_index = - min_distance

    for i in range(1, len(square_signal)):
        if square_signal[i] == 0 and square_signal[i - 1] != 0 and i - prev_falling_edge_index > min_distance:
            edges.append(i)
            prev_falling_edge_index = i

    if first:
        return np.asarray(edges)
    else:
        # Last edge is already detected in the first loop
        return np.asarray(edges)[:-1]


def minCPsLenght(startCPs: np.ndarray) -> int:
    '''
    Compute the minimum distance between two following CPs.

    Parameters
    ----------
    `startCPs` : array_like
        The starting sample for each CPs.

    Returns
    -------
    The minimum distance between two following CPs.
    '''

    # Handle empty list case
    if len(startCPs) == 0:
        return float('inf')

    min_distance = float('inf')
    np.sort(startCPs)
    for i in range(1, len(startCPs)):
        distance = startCPs[i] - startCPs[i - 1]
        min_distance = min(min_distance, distance)
    return min_distance


def percentileCPsLenght(startCPs: np.ndarray, percentile: int = 5) -> int:
    '''
    Compute the percentile distance between two following CPs.

    Parameters
    ----------
    `startCPs` : array_like
        The starting sample for each CPs.
    `percentile` : int, optional
        The percentile to compute. Values must be between 0 and 100 (default is 5).

    Returns
    -------
    The percentile distance between two following CPs.
    '''

    # Handle empty list case
    if len(startCPs) == 0:
        return 0

    distances = []
    np.sort(startCPs)
    for i in range(1, len(startCPs)):
        distances.append(startCPs[i] - startCPs[i - 1])

    return np.percentile(distances, percentile)


def removeLastRound(classification, min_distance):
    '''
    Remove the last round of CPs from the classification output.

    Parameters
    ----------
    `classification` : np.ndarray
        The sliding windows classification output.
    `min_distance` : int
        The minimum distance between two CPs.

    Returns
    -------
    The updated classification output with the last round of CPs removed.
    '''

    min_distance = int(min_distance)
    idx = np.where(classification[:-min_distance] == 0)[0]
    
    for i in range(0, len(idx)):
        if majorityFilter(classification[idx[i]+1:idx[i]+min_distance+1])[0] == 2:
            classification[idx[i]] = 1

    return classification


def segment(classification: np.ndarray,
            major_filter_size: int,
            stride: int,
            avg_cp_lenght: int) -> np.ndarray:
    '''
    Segment the classification output to obtain the start samples of the CPs.

    Parameters
    ----------
    `classification` : np.ndarray
        The sliding windows classification output.
    `major_filter_size` : int
        The initial size of the majority filter. It will be reduced during the segmentation.
    `stride` : int
        The stride used to slide the window in sliding windows classification.
    `avg_cp_lenght` : int
        The initial minimum distance between two CPs. It will be refine during the segmentation.
        Rule of thumb: cipher average lenght.

    Returns
    -------
    The start samples of the CPs.
    '''
    CPs = []
    offsets = []
    
    classifications_to_polish = [classification.copy()]
    
    min_distance = avg_cp_lenght // stride
    
    while(major_filter_size>=1):
    
        polished_classifications = _polish(classifications_to_polish, major_filter_size, min_distance)

        newCPs = _extract(polished_classifications, min_distance, offsets, first=len(CPs)==0)
        CPs.extend(newCPs)
        CPs = sorted(CPs)
        
        major_filter_size, min_distance, classifications_to_polish, offsets = _refine(classification, major_filter_size, 
                                                                                      avg_cp_lenght // stride, CPs)
    
    CPs = _finalizeSegmentation(classification, avg_cp_lenght // stride, CPs)
    
    return CPs

def _polish(classifications: list[np.ndarray],
            major_filter_size: int,
            min_distance: int) -> np.ndarray:
    '''
    Polish the classification output.
    It applies a majority filter and removes the last round of CPs.
    
    Parameters
    ----------
    `classification` : list[np.ndarray]
        The sliding windows classification output.
    `major_filter_size` : int
        The kernel size of the majority filter.
    `min_distance` : int
        The minimum distance between two CPs.

    Returns
    -------
    The filtered classification output.
    '''
    outs = []
    for classification in classifications:
        out = np.argmax(classification, axis=1)
        if major_filter_size > 1:
            out = majorityFilter(out, major_filter_size)
        
        out = removeLastRound(out, min_distance)
        
        outs.append(out)
    return outs

def _extract(classifications, avg_cp_lenght, offsets, first):
    '''
    Extract the starting sample for each CPs present in the classification output.
    
    Parameters
    ----------
    `classifications` : list[np.ndarray]
        The sliding windows classification output.
    `avg_cp_lenght` : int
        The average distance between two CPs.
    `offsets` : list[int]
        The offset to apply to the starting sample.
    `first` : bool
        If True, the last edge is not detected.
    
    Returns
    -------
    The starting sample for each CPs present in the classification output.
    '''
    startCPs = []
    
    if first:
        offsets = [0] * len(classifications)
        
    for classification, offset in zip(classifications, offsets):
        newStarts = extractEdges(classification, avg_cp_lenght, first) + offset
        startCPs.extend(newStarts.tolist())
        
    return startCPs

def _refine(classification, major_filter_size, avg_cp_lenght, CPs):
    '''
    Refine the segmentation parameters to avoid false positive CPs.
    
    Parameters
    ----------
    `classifications` : np.ndarray
        The sliding windows classification output.
    `major_filter_size` : int
        The kernel size of the majority filter.
    `avg_cp_lenght` : int
        The average distance between two CPs.
    `CPs` : np.ndarray
        The starting sample for each CPs.
    
    Returns
    -------
    The refined kernel size of the majority filter, the refined classification output and the refined offsets.
    '''
    
    major_filter_size = _refineMajorFilterSize(major_filter_size)
    min_distance = min(minCPsLenght(CPs), avg_cp_lenght)
    
    sub_classifications = []
    offsets = []
     
    if len(CPs) == 0:
        sub_classifications.append(classification)
        offsets.append(0)
        return major_filter_size, min_distance*0.8, sub_classifications, offsets
    
    # If there can still be a CP at the beginning of the signal
    if CPs[0] > min_distance:
        sub_classifications.append(classification[: CPs[0]])
        offsets.append(0)

    # If there can be two CPs consecutives
    for i in range(1, len(CPs)):
        if CPs[i] - CPs[i - 1] > 2*min_distance:
            offset = int(min_distance*0.8)
            sub_classifications.append(classification[CPs[i-1]+offset: CPs[i]+1])
            offsets.append(CPs[i-1]+offset)

    # If there can still be a CP at the end of the signal
    if len(classification) - CPs[-1] > 2*min_distance:
        offset = int(min_distance*0.8)
        sub_classifications.append(classification[CPs[-1]+offset:])
        offsets.append(CPs[-1]+offset)
     
    return major_filter_size, min_distance*0.8, sub_classifications, offsets

def _refineMajorFilterSize(major_filter_size):
    if major_filter_size > 9:
        return major_filter_size // 10
    elif major_filter_size > 1:
        return 1
    elif major_filter_size <= 1:
        return 0

def _finalizeSegmentation(classification, avg_cp_lenght, CPs):
    should_continue = True
    newCPs = []

    avg_cp_lenght = min(int(percentileCPsLenght(CPs, 95)), avg_cp_lenght)

    while should_continue or len(newCPs) > 0:
        should_continue = False

        for i in range(1, len(CPs)):
            newCPs = []
            # If we have at least two CPs consecutives
            if (CPs[i] - CPs[i - 1] > 1.8*avg_cp_lenght) and \
                    majorityFilter(np.argmax(classification[CPs[i-1]+avg_cp_lenght:CPs[i-1]+2*avg_cp_lenght], axis=1))[1] != 2:
                newCPs = [CPs[i-1]+avg_cp_lenght]
                CPs.extend(newCPs)
        CPs = sorted(CPs)

    return CPs
