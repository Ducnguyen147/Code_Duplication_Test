def tally_words(text: str) -> dict:
    tally = {}
    for token in text.split():
        tally[token] = tally.get(token, 0) + 1
    return tally
