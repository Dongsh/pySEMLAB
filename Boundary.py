import numpy as np


def BoundaryMatrix(wgll, NELXY, iglob, jac1D, side):
    NGLL = len(wgll)
    NELX = NELXY[0]
    NELY = NELXY[1]

    if side == "L":
        eB = np.array(range(0, NELY - 1)) * NELX + 1
        igll = 1
        jgll = np.array(range(0, NGLL))
    elif side == "R":
        eB = np.array(range(0, NELY - 1)) * NELX + NELX
        igll = NGLL
        jgll = np.array(range(0, NGLL))
    elif side == "T":
        eB = (NELY - 1) * NELX + np.array(range(1, NELX))
        igll = np.array(range(0, NGLL))
        jgll = NGLL
    else:
        eB = np.array(range(0, NELX))
        igll = np.array(range(0, NGLL))
        jgll = 0

    NELB = len(eB)
    ng = NELB * (NGLL - 1) + 1
    iB = np.zeros([ng, 1])
    B = np.zeros([ng, 1])
    jB = np.zeros([NGLL, NELB])
    # print(iglob[:, :, 1])
    # print(eB)
    for e in range(0, NELB):
        ip = (NGLL - 1) * (e) + np.array(range(0, NGLL))

        iB[ip] = iglob[igll, jgll, eB[e]][:, np.newaxis]
        # print(iglob[igll, jgll-1, eB[e]])
        jB[:, e] = ip

        B[ip] = (jac1D * np.array(wgll) + B[ip].T).T
    return B, iB, jB
