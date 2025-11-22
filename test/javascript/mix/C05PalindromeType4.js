function isPalindrome(text) {
    const dq = [];

    for (const ch of text) {
        if (/[a-zA-Z0-9]/.test(ch)) {
            dq.push(ch.toLowerCase());
        }
    }

    while (dq.length > 1) {
        if (dq.shift() !== dq.pop()) {
            return false;
        }
    }

    return true;
}

module.exports = { isPalindrome };
