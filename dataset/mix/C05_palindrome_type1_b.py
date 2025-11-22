def is_palindrome(text: str) -> bool:
    norm = ''.join(c.lower() for c in text if c.isalnum())
    return norm == norm[::-1]
