from collections import deque

def is_palindrome(text: str) -> bool:
    dq = deque(ch.lower() for ch in text if ch.isalnum())
    while len(dq) > 1:
        if dq.popleft() != dq.pop():
            return False
    return True
