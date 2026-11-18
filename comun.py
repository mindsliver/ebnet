"""
comun.py -- Maquinaria compartida por los tres casos del manual.
================================================================

Contiene SOLO lo que no depende del caso:

  * MLP            : el tronco de la red, identico en los tres casos.
  * HermiteC1      : interpolante cubico C^1, DIFERENCIABLE en torch.
                     Es la pieza que permite meter una rho obtenida por
                     integracion numerica dentro del grafo de autograd.
  * d1, d2         : derivadas por autograd, con create_graph=True.
  * entrenar       : bucle Adam + refinamiento L-BFGS.
  * error_rel_l2   : metrica de evaluacion.

Convencion de todo el manual: float64. Una PINN en float32 se estanca
alrededor de 1e-4 y uno cree que el metodo no funciona.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)


# =====================================================================
# 1. Tronco de la red
# =====================================================================
class MLP(nn.Module):
    """Perceptron multicapa con tanh e inicializacion de Xavier.

    out_dim = 1 en los casos reales (familias I y II).
    out_dim = 2 en el caso cuantico (parte real e imaginaria).

    El tronco es IDENTICO en los cuatro modelos de cada escalera: lo unico
    que cambia entre modelos es que se le mete por delante (el embedding)
    y que se le pone por detras (la reconstruccion).
    """

    def __init__(self, in_dim: int = 2, out_dim: int = 1,
                 width: int = 64, depth: int = 4):
        super().__init__()
        capas = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            capas += [nn.Linear(width, width), nn.Tanh()]
        capas += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*capas)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =====================================================================
# 2. Interpolante de Hermite C^1  (la pieza critica del pipeline)
# =====================================================================
class HermiteC1:
    r"""Interpolante cubico de Hermite por tramos sobre una malla UNIFORME.

    PROBLEMA QUE RESUELVE
    ---------------------
    En los casos II y III la escala rho(t) no tiene forma cerrada: sale de
    integrar una EDO. Pero la red necesita rho(t) y, sobre todo, autograd
    necesita derivar a traves de rho(t) para formar el residuo. Un
    `numpy.interp` rompe el grafo; un spline de scipy tampoco es derivable
    por torch.

    La solucion es tabular (t_k, f_k, f'_k) con un integrador de alta
    precision y reconstruir con la base de Hermite, que usa el VALOR y la
    DERIVADA en cada nodo. El resultado es C^1 y esta hecho de operaciones
    de torch, asi que autograd lo deriva exactamente.

    Sobre s = (t - t_k)/h en [0,1]:

        h00 = 2s^3 - 3s^2 + 1        h10 = s^3 - 2s^2 + s
        h01 = -2s^3 + 3s^2           h11 = s^3 - s^2

        f(t) = h00 f_k + h*h10 f'_k + h01 f_{k+1} + h*h11 f'_{k+1}

    POR QUE C^1 Y NO C^0
    --------------------
    El residuo de la EDP contiene rho_punto. Con un interpolante solo C^0 la
    derivada es constante por tramos y discontinua: el residuo adquiere
    saltos artificiales en cada nodo y el entrenamiento se atasca. C^1 es el
    minimo que sirve.

    PRECISION
    ---------
    El error del interpolante es O(h^4) en el valor y O(h^3) en la derivada.
    Con n=2001 nodos sobre [0,T] eso queda muy por debajo del error de la
    red, de modo que el embedding no es el cuello de botella.
    """

    def __init__(self, ts, fs, dfs):
        ts = np.asarray(ts, dtype=float)
        h = ts[1] - ts[0]
        if not np.allclose(np.diff(ts), h, rtol=1e-10, atol=0.0):
            raise ValueError("HermiteC1 exige malla uniforme.")
        self.t0 = float(ts[0])
        self.h = float(h)
        self.n = len(ts)
        self.f = torch.as_tensor(np.asarray(fs, dtype=float), dtype=DTYPE)
        self.df = torch.as_tensor(np.asarray(dfs, dtype=float), dtype=DTYPE)

    def __call__(self, t):
        # Indice del tramo. .long() lo desacopla de autograd (es un entero),
        # de modo que el gradiente fluye solo por s, que es lo correcto.
        u = (t - self.t0) / self.h
        k = torch.clamp(torch.floor(u).long(), 0, self.n - 2)
        s = u - k.to(DTYPE)                      # s en [0,1], derivable

        fk, fk1 = self.f[k], self.f[k + 1]
        dk, dk1 = self.df[k], self.df[k + 1]

        s2, s3 = s * s, s * s * s
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2
        return h00 * fk + self.h * h10 * dk + h01 * fk1 + self.h * h11 * dk1


# =====================================================================
# 3. Derivadas por autograd
# =====================================================================
def d1(f, x):
    """df/dx conservando el grafo (imprescindible para derivar otra vez)."""
    return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]


def d2(f, x):
    """d2f/dx2 en dos pasos."""
    return d1(d1(f, x), x)


# =====================================================================
# 4. Entrenamiento
# =====================================================================
@dataclass
class Cfg:
    n_adam: int = 2500
    n_lbfgs: int = 200
    lr: float = 2e-3
    n_col: int = 1500      # puntos de colocacion (residuo)
    n_ic: int = 300        # puntos de condicion inicial
    n_bc: int = 150        # puntos de frontera
    lam_ic: float = 10.0   # peso de la condicion inicial
    lam_bc: float = 10.0   # peso de la frontera
    seed: int = 0
    log_every: int = 500
    verbose: bool = False


def entrenar(model, perdida, lotes, cfg: Cfg):
    """Adam con decaimiento coseno y refinamiento final L-BFGS.

    `perdida(model, lote) -> (escalar, dict de partes)`
    `lotes(gen) -> lote`                (se vuelve a muestrear en cada paso)

    El mismo bucle, con el mismo presupuesto, para los cuatro modelos de la
    escalera: si el optimizador cambia entre modelos la comparacion no dice
    nada sobre el embedding.
    """
    gen = torch.Generator().manual_seed(cfg.seed + 12345)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.n_adam, eta_min=1e-5)

    t0 = time.time()
    for k in range(cfg.n_adam):
        lote = lotes(gen)
        L, partes = perdida(model, lote)
        opt.zero_grad(set_to_none=True)
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sch.step()
        if cfg.verbose and k % cfg.log_every == 0:
            print(f"      it={k:5d}  L={float(L):.3e}  " +
                  "  ".join(f"{a}={b:.2e}" for a, b in partes.items()))

    if cfg.n_lbfgs > 0:
        lote = lotes(gen)                      # lote FIJO para L-BFGS
        lb = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=cfg.n_lbfgs,
            tolerance_grad=1e-12, tolerance_change=1e-14,
            history_size=50, line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad(set_to_none=True)
            L, _ = perdida(model, lote)
            L.backward()
            return L

        try:
            lb.step(closure)
        except RuntimeError:
            pass                               # L-BFGS a veces aborta; no es fatal

    return time.time() - t0


# =====================================================================
# 5. Evaluacion
# =====================================================================
def error_rel_l2(pred, exact) -> float:
    """||pred - exact||_2 / ||exact||_2 sobre la malla de evaluacion."""
    num = torch.linalg.vector_norm(pred - exact)
    den = torch.linalg.vector_norm(exact)
    return float(num / den)


def malla(L: float, t_lo: float, t_hi: float, nx: int = 400, nt: int = 100):
    xs = torch.linspace(-L, L, nx)
    ts = torch.linspace(t_lo, t_hi, nt)
    X, T = torch.meshgrid(xs, ts, indexing="ij")
    return X.reshape(-1, 1), T.reshape(-1, 1)


def tabla(filas, titulo: str, ref: str | None = None):
    """Imprime la escalera de ablacion con la razon respecto al peldano previo."""
    print(f"\n  {titulo}")
    print("  " + "-" * 62)
    print(f"  {'modelo':<14}{'params':>8}{'rel L2':>12}{'vs PINN':>10}{'vs previo':>12}")
    print("  " + "-" * 62)
    base = filas[0][2]
    prev = None
    for nombre, npar, err in filas:
        g0 = base / err
        gp = (prev / err) if prev else float("nan")
        gp_s = "    --" if prev is None else f"{gp:9.2f}x"
        print(f"  {nombre:<14}{npar:>8d}{err:>12.3e}{g0:>9.2f}x{gp_s:>12}")
        prev = err
    print("  " + "-" * 62)
    if ref:
        print(f"  {ref}")
