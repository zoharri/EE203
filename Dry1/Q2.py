import numpy as np

def calc_psi(N, X):
    psi_table = np.zeros([N, X])
    psi_table[0, :] = 1

    for x in range(X):
        for n in range(1, N):
            if n > x:
                psi_table[n, x] = 0
            else:
                psi_table[n, x] = np.sum(psi_table[n-1, :x])
    return psi_table[N-1, X-1]
    #return psi_table

if __name__ == "__main__":
    print(calc_psi(12, 800))