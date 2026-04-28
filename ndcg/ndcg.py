import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    # Write code here

    scores = relevance_scores[:k]

    def dcg(scores):
        total = 0.0
        for i, rel in enumerate(scores):
            rank = i + 1
            total += (2 ** rel - 1) / math.log2(rank + 1)
        return total

    dcg_score = dcg(scores)

    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg_score = dcg(ideal_scores)

    if idcg_score == 0:
        return 0.0

    return dcg_score / idcg_score