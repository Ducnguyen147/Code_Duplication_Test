def extract_and_sort(input_list):
    # input_list: List of dicts, e.g., [{"value":5},{"value":2},...]
    extracted = []
    for item in input_list:
        if "value" in item:
            extracted.append(item["value"])
    extracted.sort(reverse=True)
    return extracted
