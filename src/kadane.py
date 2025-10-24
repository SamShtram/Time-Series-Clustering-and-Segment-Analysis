def kadane(signal):
    """Find maximum subarray (Kadane’s Algorithm)."""
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
