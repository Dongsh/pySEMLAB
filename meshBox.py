import numpy as np


def GetGLL(*par):
    if len(par) > 1:
        kind = par[1]
        prefix = kind[:3]
    else:
        prefix = "gll"
    ngll = par[0]
    name = "gll_xwh/" + prefix + "_" + str(ngll) + ".tab"
    with open(name, 'r') as file:
        array = []
        for line in file:
            array.append([float(x) for x in line.split()])
    file.close()
    x = array[0][:]
    w = array[1][:]
    h = array[2:][:]
    return x, w, h


def meshBox(LX, LY, NELX, NELY, NGLL):
    dxe = LX / NELX
    dye = LY / NELY
    XGLL = GetGLL(NGLL)[0]

    iglob = np.zeros([NGLL, NGLL, NELX * NELY + 1])
    nglob = (NELX * (NGLL - 1) + 1) * (NELY * (NGLL - 1) + 1)
    x = np.zeros([nglob, 1])
    y = np.zeros([nglob, 1])
    e = 0
    last_iglob = 0
    igL = np.array(range(1, NGLL * (NGLL - 1) + 1)).reshape([NGLL, NGLL - 1]).T
    igB = np.array(range(1, NGLL * (NGLL - 1) + 1)).reshape([NGLL - 1, NGLL]).T
    igLB = np.array(range(1, (NGLL - 1) * (NGLL - 1) + 1)).reshape([NGLL - 1, NGLL - 1]).T
    xgll = np.tile([0.5 * (x + 1) for x in XGLL], [1, NGLL]).reshape([NGLL, NGLL]).T
    ygll = dye * xgll.T
    xgll = dye * xgll

    for ey in range(NELY):
        for ex in range(NELX):
            e = e + 1

            if e == 1:
                ig = np.array(range(0, NGLL * NGLL)).reshape([NGLL, NGLL]).T+1
            else:
                # print(e)
                if ey == 0:
                    # print(iglob[NGLL - 1, :, e - 1].shape)
                    ig[0, :] = iglob[NGLL - 1, :, e - 1]  # left edge
                    ig[1:NGLL, :] = last_iglob + igL
                elif ex == 0:
                    # print(iglob[:, NGLL, e - NELX].shaope)
                    # print(igB.shape)
                    ig[:, 0] = iglob[:, NGLL - 1, e - NELX]
                    ig[:, 1:NGLL] = last_iglob + igB

                else:
                    ig[0, :] = iglob[NGLL - 1, :, e - 1]
                    ig[:, 0] = iglob[:, NGLL - 1, e - NELX]
                    ig[1:NGLL, 1:NGLL] = last_iglob + igLB

            iglob[:, :, e] = ig
            last_iglob = ig[NGLL - 1, NGLL - 1]
            x[ig - 1] = dxe * ex + xgll[:, :, np.newaxis]
            y[ig - 1] = dye * ey + ygll[:, :, np.newaxis]

    iglob = iglob[:, :, 1:NELX * NELY + 1]
    return iglob, x, y


def repmat(mat, x, y):
    return np.tile(mat, [x, y])


def FindNearestNode(xin, yin, X, Y):
    nseis = len(xin)
    dist = np.zeros([nseis, 1])
    iglob = np.zeros([nseis, 1])
    for k in range(nseis):
        dist[k] = np.min((X - xin[k]) ** 2 + (Y - yin[k]) ** 2)
        iglob[k] = np.argmin((X - xin[k]) ** 2 + (Y - yin[k]) ** 2)
    dist = np.sqrt(dist)

    xout = X[iglob.astype(int)]
    yout = Y[iglob.astype(int)]
    return xout, yout, iglob, dist
