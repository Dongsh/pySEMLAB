import numpy as np


class Friction:
    kind = 1
    MUs = []
    MUd = []
    Dc = []
    W = []
    p = []

    def __init__(self):
        self.kind = 1
        self.MUd = []
        self.MUs = []
        self.Dc = []
        self.W = []
        self.p = []


def friction(u, f: Friction):
    if f.kind == 1:                     # Linear slip weakening friction law
        W = (np.array(f.MUs) - np.array(f.MUd)) / f.Dc
        mu = np.maximum(np.array(f.MUs) - W * u, f.MUd)
    elif f.kind == 2:                   # -- Chambon's non linear law --
        u = u / (np.array(f.p) * np.array(f.Dc))
        mu = f.MUd + (np.array(f.MUs) - np.array(f.MUd)) / (1 + u) ** np.array(f.p)
    elif f.kind == 3:                   # -- Abercrombie and Rice non-linear law --
        u = u / np.array(f.Dc)
        # mu = f.MUs - (f.MUs - f.MUd)* (u. * (u <= 1) + (f.p - 1 + u. ^ f.p) / f.p. * (u > 1));
        mu = np.array(f.MUs) - (np.array(f.MUs) - np.array(f.MUd)) * (
            u * (u <= 1) + (np.array(f.p) - 1 + u ** np.array(f.p)) / np.array(f.p) * (u > 1))
    else:
        raise RuntimeError("friction unknown kind")
    return mu
