def aggregate_lines(lines):
    acc = {}
    for line in lines:
        key, val = line.strip().split(',')
        acc[key] = acc.get(key, 0) + float(val)
    return acc
