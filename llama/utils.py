import torch

def calc_skip_pattern(interleave, max_seq_len):
    skip_plot = []
    for i in range(0, max_seq_len // 10):
        skip_plot.append(0)

    j = max_seq_len // 10
    while j < max_seq_len:
        # skip_plot.extend(list(range(0, max_height)))
        # skip_pattern = list([0, 1, 2, 4, 6, 8])
        skip_pattern = [10, 12, 14, 12, 10]
        # skip_pattern = [0, 0, 0]
        skip_plot.extend(skip_pattern)
        skip_plot.extend(list([0] * interleave))
        j += (len(skip_pattern) + interleave)
    return skip_plot

    
