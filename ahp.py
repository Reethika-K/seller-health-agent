import numpy as np
from ahp_config import MEESHO, CATEGORIES, METRICS

RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}  # Saaty's random index

def ahp_weights(A):
    A = np.array(A, dtype=float)
    values, vectors = np.linalg.eig(A)
    k = np.argmax(values.real)
    lambda_max = values[k].real
    vec = vectors[:, k].real
    weights = np.abs(vec) / np.abs(vec).sum()
    return weights, lambda_max

def consistency_ratio(A, lambda_max):
    n = len(A)
    ci = (lambda_max - n) / (n - 1)
    if RI[n] == 0:
        return 0.0
    return ci / RI[n]

def metric_weights(platform_matrices):
    final = {}
    cat_w, lam = ahp_weights(platform_matrices["categories"])
    for cat, cw in zip(CATEGORIES, cat_w):
        local_w, lam = ahp_weights(platform_matrices[cat])
        for metric, lw in zip(METRICS[cat], local_w):
            final[metric] = cw * lw
    return final

if __name__ == "__main__":
    # Level 1: the 6x6 category matrix
    weights, lam = ahp_weights(MEESHO["categories"])
    cr = consistency_ratio(MEESHO["categories"], lam)
    for n, w in zip(CATEGORIES, weights):
        print(f"{n:22s} {w:.3f}")
    print(f"CR = {cr:.3f}  ->  {'OK' if cr <= 0.1 else 'inconsistent, revise matrix'}\n")

    # Level 2: check each category matrix's consistency
    for cat in CATEGORIES:
        w, lam = ahp_weights(MEESHO[cat])
        print(f"{cat:22s} local weights {w.round(3)}  CR = {consistency_ratio(MEESHO[cat], lam):.3f}")

    # Final weight of every metric
    final = metric_weights(MEESHO)
    print("\nsum of weights:", round(sum(final.values()), 4))
    for metric, w in sorted(final.items(), key=lambda x: -x[1]):
        print(f"{metric:24s}{w:.3f}")