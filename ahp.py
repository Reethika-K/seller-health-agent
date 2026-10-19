import numpy as np

RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}  # Saaty's random index

#ORDER: Marketing, Logistics, Customer Experience, Seller support, Finance, Inventory
#Level 1: Compare within categories. For Meesho, Marketing gets the highest score
MEESHO = [
    [1,   1,   2,   3,   5,   2],
    [1,   1,   1,   2,   4,   2],
    [1/2, 1,   1,   2,   3,   2],
    [1/3, 1/2, 1/2, 1,   2,   1],
    [1/5, 1/4, 1/3, 1/2, 1,   1/2],
    [1/2, 1/2, 1/2, 1,   2,   1],
]

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

if __name__ == "__main__":
    weights, lam = ahp_weights(MEESHO)
    cr = consistency_ratio(MEESHO, lam)

    names = ["Marketing", "Logistics", "Customer Experience",
             "Seller support", "Finance", "Inventory"]
    for n, w in zip(names, weights):
        print(f"{n:22s} {w:.3f}")
    print(f"\nlambda_max = {lam:.3f}")
    print(f"CR = {cr:.3f}  ->  {'OK' if cr <= 0.1 else 'inconsistent, revise matrix'}")