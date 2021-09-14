import math as mt

def distance_euclid(p, q, dim):
    # p, d ... points, dim ... dimension
    sum_squared = 0
    for i in range(dim):
        sum_squared += (p[i] - q[i]) ** 2
    return mt.sqrt(sum_squared)