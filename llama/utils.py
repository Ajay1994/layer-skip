import torch

def calc_skip_pattern(interleave, max_seq_len, random):
    if random == True:
        skip_plot = list([4] * (max_seq_len + 1))
        print(skip_plot[:200])
        return skip_plot
    else:
        skip_plot = []
        for i in range(0, max_seq_len // 64):
            skip_plot.append(0)
        j = max_seq_len // 64
        while j < max_seq_len:
            skip_pattern = [5,5,5,5,5]
            skip_plot.extend(skip_pattern)
            skip_plot.extend(list([0] * interleave))
            j += (len(skip_pattern) + interleave)
        print(skip_plot[:200])
        return skip_plot

"""
# skip_plot.extend(list(range(0, max_height)))
[12, 12, 11, 12, 12] # 25%
[14, 14, 15, 14, 14] # 30%
[0, 0, 0, 0, 0] # 0%
[5,5,5,5,5] #10%
            # skip_pattern = list([0, 1, 2, 4, 6, 8])
            # skip_pattern = [10, 12, 14, 12, 10] #25%
            # skip_pattern = [14, 16, 18, 16, 14] #33%
            # skip_pattern = [3, 5, 7, 5, 3] 
            # skip_pattern = [14, 14, 14, 14, 14] #33%
            # skip_pattern = [2, 2, 2, 2, 2]
            # skip_pattern = [0, 0, 0]
            # skip_pattern = [12, 12, 13, 12, 12]
            # skip_pattern = [10, 10, 10, 10, 10]
"""



    
