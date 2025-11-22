def count_modulo(input_list):
    freq = {}
    for x in input_list:
        r = x % 3
        freq[r] = freq.get(r, 0) + 1
    return freq
