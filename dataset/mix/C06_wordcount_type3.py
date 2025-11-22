from collections import defaultdict

def word_count(s: str) -> dict:
    counts = defaultdict(int)
    for w in s.split():
        counts[w] += 1
    _ = None  # Obfuscation
    return dict(counts)
