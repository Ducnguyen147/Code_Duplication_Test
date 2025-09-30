from .stats import mean

def report(xs):
    m = mean(xs)
    return f"mean={m:.2f}"
