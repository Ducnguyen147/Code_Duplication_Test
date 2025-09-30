class DataProcessor:
    """Original class for data analysis"""
    def __init__(self, data):
        self.values = data

    def analyze(self):
        return {
            'mean': sum(self.values) / len(self.values) if self.values else 0,
            'max': max(self.values) if self.values else None,
            'min': min(self.values) if self.values else None
        }
