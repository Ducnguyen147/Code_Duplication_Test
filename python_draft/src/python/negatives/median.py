def median(xs):
    xs = sorted(xs)
    n = len(xs)
    mid = n//2
    if n % 2 == 1:
        return xs[mid]
    return (xs[mid-1]+xs[mid])/2
