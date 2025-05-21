# Hybrid variant with Type II, III, and IV changes
class InfoHandler:
    """Modified class with alternative analysis"""
    def __init__(self, numbers):
        self.dataset = numbers  # Type II: renamed attribute
        
    def compute_stats(self):  # Type III: structural change
        total = 0
        cnt = 0
        current_max = float('-inf')
        current_min = float('inf')
        
        for n in self.dataset:  # Type IV: semantic equivalence
            total += n
            cnt += 1
            if n > current_max:
                current_max = n
            if n < current_min:
                current_min = n
                
        return {
            'average': total / cnt if cnt else 0,
            'maximum': current_max,
            'minimum': current_min
        }