# Spectral Element Method via Dongsh refer to SEMLAB

import numpy as np
import meshBox
import Boundary
import Friction
import matplotlib.pyplot as plt

if __name__ == '__main__':
    LX = 12.5e3
    LY = 7.5e3

    NELX = 25
    NELY = 15
    P = 4  # Polynomial Degree

    ABC_B = 0
    ABC_R = 2
    ABC_T = 2
    ABC_L = 0

    dxe = LX / NELX
    dye = LY / NELY
    NEL = NELX * NELY
    NGLL = P + 1
    # NGLL = 9
    # XGLL = meshBox.GetGLL(NGLL)[0]
    iglob, x, y = meshBox.meshBox(LX=LX, LY=LY, NELX=NELX, NELY=NELY, NGLL=NGLL)

    nglob = len(x)

    RHO = 2670.
    VS = 3464.
    MU = RHO * VS ** 2

    ETA = 0.1
    PML_A = 10
    PML_N = 2

    xgll, wgll, H = meshBox.GetGLL(NGLL)
    

    Ht = np.array(H).T
    W = np.array(wgll)[:, np.newaxis] * np.array(wgll)[np.newaxis, :]
    M = np.zeros([nglob, 1])
    rho = np.zeros([NGLL, NGLL])
    mu = np.zeros([NGLL, NGLL])
    NT = 0
    TT = 12
    CFL = 0.6

    dt = np.inf
    dx_dxi = 0.5 * dxe
    dy_deta = 0.5 * dye
    jac = dx_dxi * dy_deta
    coefint1 = jac / dx_dxi
    coefint2 = jac / dy_deta

    for ey in range(NELY):
        for ex in range(NELX):
            e = (ey) * NELX + ex-1
            ig = iglob[:, :, e].astype(np.int32) - 1

            rho[:, :] = RHO
            mu[:, :] = MU

            M[ig] = M[ig] + (np.multiply(W, rho) * jac)[:, :, np.newaxis]
            vs = np.sqrt(mu / rho)
            if dxe < dye:
                vs = np.maximum(vs[0:NGLL - 1, :], vs[1:NGLL, :])
                dx = meshBox.repmat(np.diff(xgll) * 0.5 * dxe, 1, NGLL)
            else:
                vs = np.maximum(vs[:, 0:NGLL - 1], vs[:, 1:NGLL])
                dx = meshBox.repmat(np.diff(xgll).T * 0.5 * dxe, NGLL, 1)

            dtloc = dx / vs

            dt = np.min(np.append(np.array(dtloc.flatten(order='F')), np.array(dt)))

    dt = CFL * dt
    
    if ETA:
        dt = dt / np.sqrt(1 + 2 * ETA)

    half_dt = 0.5 * dt

    if NT == 0:
        NT = np.ceil(TT / dt)

    f = np.zeros([nglob, 1])
    v = np.zeros([nglob, 1])
    d = np.zeros([nglob, 1])

    time = np.array(range(0, NT.astype(np.int32))).T * dt

    impedance = RHO * VS
    if ABC_L == 1:
        BcLC, iBcL, noUsed = Boundary.BoundaryMatrix(wgll, [NELX, NELY], iglob, dy_deta, 'L')
        BcLC = impedance * BcLC
        M[iBcL] = M[iBcL] + half_dt * BcLC

    if ABC_R == 1:
        BcRC, iBcR, noUsed = Boundary.BoundaryMatrix(wgll, [NELX, NELY], iglob, dy_deta, 'R');
        BcRC = impedance * BcRC
        M[iBcR] = M[iBcR] + half_dt * BcRC

    if ABC_T == 1:
        BcTC, iBcT, noUsed = Boundary.BoundaryMatrix(wgll, [NELX, NELY], iglob, dx_dxi, 'T')
        BcTC = impedance * BcTC
        M[iBcT] = M[iBcT] + half_dt * BcTC

    # ------- finished : PML ---------
    anyPML = np.array(np.array([ABC_T, ABC_R]) == 2).any()
    # print(anyPML)
    if anyPML:

        ePML = []
        NEL_PML = 0
        if ABC_R > 1:
            ePML = np.arange(NELX, NEL + 1, NELX)
        if ABC_T > 1:
            ePML = np.append(ePML, range(NEL - NELX, NEL))
        ePML = np.unique(ePML)
        NEL_PML = len(ePML)
        # print(iglob[:, :,26 ])
        iglob_PML = iglob[:, :, ePML - 1]
        # print(iglob_PML[:, :, 0])
        
        iPML, dum, iglob_PML = np.unique(iglob_PML.flatten(order='F'), return_index=True, return_inverse=True)
#        plt.plot(iglob_PML.flatten(1))
#        plt.show()
        iglob_PML = iglob_PML.reshape([NGLL, NGLL, NEL_PML],order='F')
        nPML = len(ePML)
        axPML = np.zeros([nPML, 1])
        ayPML = np.zeros([nPML, 1])

        xp = x[iPML.astype(int) - 1]
        yp = y[iPML.astype(int) - 1]
        lx = LX - dxe
        ly = LY - dye

        if ABC_R:
            axPML = PML_A * VS / dxe * ((xp - lx) / dxe) ** PML_N * (xp >= lx)
        if ABC_T:
            ayPML = PML_A * VS / dye * ((yp - ly) / dye) ** PML_N * (yp >= ly)

#        del xp, yp

        ahsumPML = 0.5 * (axPML + ayPML)
        aprodPML = axPML * ayPML
        asinvPML = 1 / (1 + dt * ahsumPML)

        s_x = np.zeros([NGLL, NGLL, NEL_PML])
        s_y = np.zeros([NGLL, NGLL, NEL_PML])
    else:
        NEL_PML = 0
        ePML = 0
        # iglob_PML = iglob[:, :, ePML]

    # raise RuntimeError("Pause")
    # Fault Settings
    FltB, iFlt, noUsed = Boundary.BoundaryMatrix(wgll=wgll, NELXY=[NELX, NELY], iglob=iglob, jac1D=dx_dxi, side='B')

    # print(FltB)
    FltN = len(iFlt)
    #
    FltZ = np.array(M[iFlt.astype(int)-1].squeeze() / FltB.T).squeeze() / dt
    FltX = x[iFlt.astype(int)]
    FltV = np.zeros([FltN, NT.astype(int)])
    FltD = np.zeros([FltN, NT.astype(int)])
    # # % background stress
    FltNormalStress = 120e6
    FltInitStress = meshBox.repmat(70e6, FltN, 1)
    # % nucleation
    isel = np.where(np.abs(np.array(FltX).squeeze()) <= 1.5e3)  # isel = find(abs(FltX) <= 1.5e3);
    # print(np.array(FltX).squeeze())
    FltInitStress[isel] = 81.6e6
    # % friction
    FltFriction = Friction.Friction()
    FltState = np.zeros([1, FltN])
    FltFriction.MUs = meshBox.repmat(0.677, FltN, 1)
    FltFriction.MUd = meshBox.repmat(0.525, FltN, 1)
    FltFriction.Dc = 0.4
    # # % barrier
    L_BARRIER = 15e3 / 2
    # # isel = find(abs(FltX) > L_BARRIER);
    isel = np.where(np.abs(np.array(FltX).squeeze()) > L_BARRIER)  # isel = find(abs(FltX) > L_BARRIER);
    FltFriction.MUs[isel] = 1e4
    FltFriction.MUd[isel] = 1e4
    FltFrictionW = (FltFriction.MUs - FltFriction.MUd) / FltFriction.Dc
    FltStrength = Friction.friction(u=FltState, f=FltFriction) * FltNormalStress - FltInitStress  # strength excess

    if ETA:

        NEL_ETA = min(np.float(NELX), np.ceil(L_BARRIER / dxe) + 2)
        x1 = 0.5 * (1 + np.array(xgll).T)
        eta_tapper = np.exp(-np.pi * x1 ** 2)
        eta = ETA * dt * meshBox.repmat(eta_tapper, NGLL, 1)
    else:
        NEL_ETA = 0

    OUTxseis = np.arange(0, 10e3 + 1, 0.5e3).T
    OUTnseis = len(OUTxseis)
    OUTyseis = meshBox.repmat(2e3, OUTnseis, 1)

    OUTxseis, OUTyseis, OUTiglob, OUTdseis = meshBox.FindNearestNode(xin=OUTxseis, yin=OUTyseis, X=x, Y=y)
    OUTv = np.zeros([OUTnseis, NT.astype(int)])
    OUTdt = np.floor(0.5 / dt)
    OUTit = 0
    # OUTindx =               # Plot2dSnapshot

    # Solve Equation
    
    for it in range(0, int(NT)):
        d = d + dt * v
        FltD[:, it] = 2 * d[iFlt.astype(int)-1].squeeze()

        f[:] = 0

        ep = 0
        # eep = ePML[ep]
        eep = ePML[ep]

        # eepn = eN

        for e in range(NEL):
            isPML = e == eep
            isETA = e <= NEL_ETA

            ig = iglob[:, :, e]

            if isPML:
                igPML = iglob_PML[:, :, ep]
                ax = axPML[igPML.astype(int) - 1]
                ay = ayPML[igPML.astype(int) - 1]

                vloc = v[ig.astype(int) - 1]
                dloc = d[ig.astype(int) - 1] - half_dt * vloc
                locx = vloc + np.multiply(ay, dloc)
                locy = vloc + np.multiply(ax, dloc)

                sx = MU * Ht * locx.squeeze() / dx_dxi
                sy = MU * locy.squeeze() * H / dy_deta
                sx = np.multiply(dt * sx + (1 - half_dt * ax).squeeze(), s_x[:, :, ep]) / (1 + half_dt * ax).squeeze()
                sy = np.multiply(dt * sy + (1 - half_dt * ay).squeeze(), s_y[:, :, ep]) / (1 + half_dt * ay).squeeze()

                s_x[:, :, ep] = sx
                s_y[:, :, ep] = sy

                ep = ep + 1
                if ep <= NEL_PML:
                    eep = ePML[ep]
                else:
                    eep = 0

            else:
                if isETA:
                    local = d[ig.astype(int) - 1].squeeze() + np.multiply(eta, v[ig.astype(int) - 1].squeeze())
                else:
                    local = d[ig.astype(int) - 1]
                

                sx = MU * np.dot(Ht, local.squeeze()) / dx_dxi
                sy = MU * np.dot(local.squeeze(), H) / dy_deta
                
                
            d_xi = W * sx
            d_xi =np.dot( H, d_xi )* coefint1
            d_eta = W * sy
            d_eta =coefint2* np.dot(d_eta , Ht )
            local = d_xi + d_eta
#            print(np.max(local))

#            f[ig.astype(int) - 1] = (f[ig.astype(int) - 1].squeeze() - local)[:, :, np.newaxis]
            f[ig.astype(int).flatten(order='F') - 1] = (f[ig.astype(int).flatten(order='F') - 1].squeeze() - local.flatten(order='F'))[:, np.newaxis]
            
        if ABC_L == 1:
            f[iBcL-1] = f[iBcL-1] - np.multiply(BcLC, v[iBcL-1])
        if ABC_R == 1:
            f[iBcR-1] = f[iBcR-1] - np.multiply(BcRC, v[iBcR-1])
        if ABC_T == 1:
            f[iBcT-1] = f[iBcT-1] - np.multiply(BcTC, v[iBcT-1])

        if anyPML:
            tmp = np.multiply(ahsumPML, v[iPML.astype(int) - 1].T) + np.multiply(aprodPML, d[iPML.astype(int) - 1])
            f[iPML.astype(int) - 1] = np.multiply(aprodPML.squeeze(), d[iPML.astype(int) - 1].squeeze())[:, np.newaxis]

        FltState = np.maximum(FltD[:, it], FltState)
        FltStrength = (Friction.friction(FltState.T, FltFriction) * FltNormalStress).squeeze() - FltInitStress.squeeze()
        FltVFree = v[iFlt.astype(int)-1].squeeze() + dt * f[iFlt.astype(int)-1].squeeze() / M[iFlt.astype(int)-1].squeeze()

        # raise RuntimeError('Stop')
        TauStick = np.multiply(FltZ, FltVFree)

        Tau = np.minimum(TauStick, FltStrength)

        f[iFlt.astype(int)-1] = (f[iFlt.astype(int)-1].squeeze() - np.multiply(FltB.squeeze(), Tau).squeeze())[:, np.newaxis, np.newaxis]

        v = v + dt * f / M

        # if anyPML
        FltV[:, it] = 2 * v[iFlt.astype(int)].squeeze()

        # STEP 4 OUT

        OUTv[:, it] = v[OUTiglob.astype(int)].squeeze()

        if np.mod(it, OUTdt) == 0:
            OUTit = OUTit + 1

            # FIGURE 1

            # FIGURE 2

            # test[it] = max(abs(v))
        plt.cla()
        plt.scatter(x,y,c=v)
        plt.pause(0.0001)

