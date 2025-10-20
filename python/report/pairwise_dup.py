import json
from collections import defaultdict

def calculate_pairwise_duplication(data: dict):
    """
    Given a jscpd JSON report, compute duplicate-line counts and ratios
    for each pair of distinct files.
    """
    # total lines per source file
    sources = data["statistics"]["formats"]["python"]["sources"]
    total_lines = {name: info["lines"] for name, info in sources.items()}

    # accumulate duplicated line numbers per pair
    pairwise = defaultdict(lambda: defaultdict(set))

    for dup in data["duplicates"]:
        f1 = dup["firstFile"]["name"]
        f2 = dup["secondFile"]["name"]

        # Ignore duplicate fragments within the same file
        if f1 == f2:
            continue

        pair = tuple(sorted((f1, f2)))

        # prepare sets for both files in the pair
        _ = pairwise[pair][f1]
        _ = pairwise[pair][f2]

        # extract start/end lines
        start1, end1 = dup["firstFile"]["start"], dup["firstFile"]["end"]
        start2, end2 = dup["secondFile"]["start"], dup["secondFile"]["end"]

        # clamp ranges to the valid line numbers for each file
        max1 = total_lines.get(f1, 0)
        max2 = total_lines.get(f2, 0)
        lines1 = set(range(max(1, start1), min(end1, max1) + 1)) if max1 else set()
        lines2 = set(range(max(1, start2), min(end2, max2) + 1)) if max2 else set()

        pairwise[pair][f1].update(lines1)
        pairwise[pair][f2].update(lines2)

    # build results
    results = []
    for (fileA, fileB), lines_dict in pairwise.items():
        dupA = len(lines_dict[fileA])
        dupB = len(lines_dict[fileB])
        ratioA = dupA / total_lines[fileA] if total_lines[fileA] else 0
        ratioB = dupB / total_lines[fileB] if total_lines[fileB] else 0

        results.append({
            "pair": (fileA, fileB),
            "duplicate_lines": {fileA: dupA, fileB: dupB},
            "ratio": {fileA: ratioA, fileB: ratioB}
        })
    return results


# Example usage:
if __name__ == "__main__":
    with open("jscpd-report.json", "r") as f:
        jscpd_data = json.load(f)

    pairs = calculate_pairwise_duplication(jscpd_data)
    for entry in pairs:
        fileA, fileB = entry["pair"]
        dup_lines = entry["duplicate_lines"]
        ratios = entry["ratio"]
        print(f"Pair: {fileA} & {fileB}")
        print(f"  Duplicate lines in {fileA}: {dup_lines[fileA]} "
              f"({ratios[fileA]:.2%} of total lines)")
        print(f"  Duplicate lines in {fileB}: {dup_lines[fileB]} "
              f"({ratios[fileB]:.2%} of total lines)\n")
