import numpy as np
import matplotlib.pyplot as plt

def _diag_matrix(d):
    """Creează matrice diagonală dintr-un vector."""
    n = len(d)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = d[i]
    return A
E = np.loadtxt("E.txt")
n, N= np.shape(E)
print(n)
print(N)
plt.figure(figsize=(10, 6))

for i in range(6):
    plt.subplot(2, 3, i + 1)

    img = E[:, i].reshape((100, 100), order="F")
    plt.imshow(img, cmap="jet")
    plt.title(f"Imagine originala {i + 1}")
    plt.colorbar(shrink = 0.9)

plt.tight_layout()
plt.show()
n, N = E.shape
m = np.mean(E, axis=1, keepdims=True)
A = E - m
z, Z= np.shape(A)
print(z)
print(Z)


U, s, Vt = np.linalg.svd(A, full_matrices=False)


S = _diag_matrix(s)
print ("diag merge")
Sigma = S / np.sqrt(N - 1)

plt.figure(figsize=(12, 6))

for i in range(10):
    eps = np.random.randn(N, 1)

    # decoder
    v = m + U @ Sigma @ eps

    plt.subplot(2, 5, i + 1)

    img_new = v.reshape((100, 100), order="F")
    plt.imshow(img_new, cmap="jet")
    plt.title(f"New image {i + 1}")
    plt.colorbar(shrink= 0.5)

plt.tight_layout()
plt.show()