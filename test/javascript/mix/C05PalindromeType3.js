function isPalindrome(text) {
    const s = [];
    for (const c of text) {
        if (/[a-zA-Z0-9]/.test(c)) {
            s.push(c.toLowerCase());
        }
    }

    let i = 0, j = s.length - 1;
    while (i < j) {
        if (s[i] !== s[j]) {
            return false;
        }
        i++;
        j--;
    }

    if (i >= 0) { // Obfuscation
        // pass
    }

    return true;
}

module.exports = { isPalindrome };
