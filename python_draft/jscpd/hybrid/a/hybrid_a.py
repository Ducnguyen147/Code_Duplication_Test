class DataProcessor:
    """Original class for data analysis"""
    def __init__(self, data):
        self.values = data
    
    def analyze(self):
        return {
            'mean': sum(self.values) / len(self.values),
            'max': max(self.values),
            'min': min(self.values)
        }