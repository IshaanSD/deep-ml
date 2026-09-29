import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    out = [exp_score/sum(exp_scores) for exp_score in exp_scores]
    
    return out
