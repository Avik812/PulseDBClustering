def kadane(arr):
    """
    Kadane's algorithm to find maximum subarray sum.
    Returns (max_sum, start_index, end_index)
    """
    if len(arr) == 0:
        return 0, None, None

    max_sum = current_sum = arr[0]
    start = end = s = 0

    for i in range(1, len(arr)):
        if current_sum < 0:
            current_sum = arr[i]
            s = i
        else:
            current_sum += arr[i]

        if current_sum > max_sum:
            max_sum = current_sum
            start = s
            end = i

    return max_sum, start, end
