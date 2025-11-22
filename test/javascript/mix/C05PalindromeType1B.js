// Added comments and white spaces
function isPalindrome(text) {
    let cleaned = "";

    for (const ch of text) {
        if (/[a-zA-Z0-9]/.test(ch)) {
            cleaned += ch.toLowerCase();
        }
    }

    const cleanedStr = cleaned;
    return cleanedStr === cleanedStr.split("").reverse().join("");
}

module.exports = { isPalindrome };
