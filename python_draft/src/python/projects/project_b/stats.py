def avg(xs):  # Type II rename of mean
    total = 0
    cnt = 0
    for v in xs:
        total += v; cnt += 1
    return total/cnt if cnt else 0
