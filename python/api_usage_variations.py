# ----------------------------
# API Usage Variations
# ----------------------------

# Original requests usage
def fetch_data_1(url):
    import requests
    response = requests.get(url)
    return response.json()

# Type IV equivalent with different HTTP library
def fetch_data_2(url):
    from urllib.request import urlopen
    import json
    with urlopen(url) as response:
        return json.load(response)