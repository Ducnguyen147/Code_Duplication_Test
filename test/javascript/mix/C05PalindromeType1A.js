function isPalindrome(text) {
    let cleaned = "";
    for (const ch of text) {
        if (/[a-zA-Z0-9]/.test(ch)) {
            cleaned += ch.toLowerCase();
        }
    }
    return cleaned === cleaned.split("").reverse().join("");
}

module.exports = { isPalindrome };
