def kadane(arr):
    # arr: 1D iterable (numeric)
    max_sum = float('-inf')
    curr = 0
    start = end = temp_start = 0
    for i, v in enumerate(arr):
        curr += v
        if curr > max_sum:
            max_sum = curr
            start, end = temp_start, i
        if curr < 0:
            curr = 0
            temp_start = i + 1
    return max_sum, start, end
