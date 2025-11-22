def word_count(s: str) -> dict:
    counts = {}
    for w in s.split():
        counts[w] = counts.get(w, 0) + 1
    return counts
