def kadane(signal):
    """
    Apply Kadane’s algorithm to find the subarray with maximum sum.
    Returns (start_index, end_index, max_sum)
    """
    max_sum = current_sum = signal[0]
    start = end = temp_start = 0

    for i in range(1, len(signal)):
        if current_sum < 0:
            current_sum = signal[i]
            temp_start = i
        else:
            current_sum += signal[i]

        if current_sum > max_sum:
            max_sum = current_sum
            start, end = temp_start, i

    return start, end, max_sum
