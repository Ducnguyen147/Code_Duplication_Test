def squares_positive(input_list):
    def gen():
        for x in input_list:
            if x >= 0:
                yield x * x

    return list(gen())
