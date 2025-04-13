# ----------------------------
# Complex Hybrid Example
# Combining Multiple Clone Types
# ----------------------------

class DataProcessor:
    """Original class implementation"""
    def __init__(self, data):
        self.values = data
    
    def analyze(self):
        return {
            'mean': sum(self.values)/len(self.values),
            'max': max(self.values),
            'min': min(self.values)
        }

# Hybrid variant - Type II, III, and IV changes
class InfoHandler:
    """Modified analysis implementation"""
    def __init__(self, numbers):
        self.dataset = numbers  # Type II change
        
    def compute_stats(self):  # Type III structural change
        total = 0
        cnt = 0
        current_max = float('-inf')
        current_min = float('inf')
        
        for n in self.dataset:  # Type IV algorithmic change
            total += n
            cnt += 1
            if n > current_max:
                current_max = n
            if n < current_min:
                current_min = n
                
        return {
            'average': total/cnt if cnt else 0,
            'maximum': current_max,
            'minimum': current_min
        }