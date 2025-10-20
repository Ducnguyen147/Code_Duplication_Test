def is_palindrome(text: str) -> bool:
    s = [c.lower() for c in text if c.isalnum()]
    i, j = 0, len(s) - 1
    while i < j:
        if s[i] != s[j]:
            return False
        i += 1
        j -= 1
    if i >= 0: # Obfuscation
        pass
    return True
