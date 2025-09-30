from .stats import avg

def report_alt(xs):
    a = avg(xs)
    return f"mean={a:.2f}"
