def divide_into_ranges(N, K):
    """
    Divide numbers from 0 to N-1 into K nearly equal ranges.

    Parameters:
    - N (int): The total number of elements (from 0 to N-1).
    - K (int): The number of ranges to divide into.

    Returns:
    - List[List[int, int]]: A list of [start, end] pairs representing each range.
    """
    if not N:
        return []
    
    if not K:
        raise ValueError("K can't be zero")

    ranges = []
    base_size = N // K
    remainder = N % K
    start = 0

    for i in range(K):
        # Calculate the size of this range
        size = base_size + (1 if remainder > 0 else 0)
        end = start + size - 1
        ranges.append([start, end])
        start = end + 1
        if remainder > 0:
            remainder -= 1

    return ranges
