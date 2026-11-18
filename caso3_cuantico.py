"""
caso3_cuantico.py -- CASO 3:  c = +1
=====================================

        i psi_t = -1/2 psi_xx + 1/2 omega^2(t) x^2 psi

El oscilador cuantico con frecuencia dependiente del tiempo. Es la rama
OPUESTA de la constante de Pinney (c = +1) y por eso cambia lo que se
cuelga del tronco: aqui lo que carga la ganancia es la FASE CUADRATICA, no
el prefactor de amplitud.

Dos diferencias practicas respecto de los casos 1 y 2:

  (a) la incognita es COMPLEJA. La red tiene dos salidas (parte real e
      imaginaria) y el residuo se parte en dos residuos reales.

  (b) el control honesto no es otra red: es la APROXIMACION ADIABATICA, que
      es lo que usaria un fisico. Se prueba en un regimen donde no tiene
      derecho a funcionar, y se comprueba que efectivamente falla.

Bloques:
    python caso3_cuantico.py --bloque 1   # integrar Pinney; rho Y rho-punto interpolados
    python caso3_cuantico.py --bloque 2   # verificar la solucion exacta y medir la adiabaticidad
    python caso3_cuantico.py --bloque 3   # el residuo complejo, partido en dos
    python caso3_cuantico.py --bloque 4   # entrenar la escalera
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from scipy.integrate import solve_ivp

from comun import (DTYPE, Cfg, HermiteC1, MLP, d1, entrenar, n_params, tabla)

SQ = math.pi ** (-0.25)          # normalizacion del estado base


# =====================================================================
# BLOQUE 1 -- Pinney con c = +1
# =====================================================================
class Cuantico:
    r"""REGLA 1 (rama de Pinney).  c = +1:

            rho'' + omega^2(t) rho = 1 / rho^3 ,

        con rho(0) = 1/sqrt(omega(0)) y rho'(0) = 0, que es la eleccion que
        arranca en el estado base instantaneo. La rama c > 0 es oscilatoria
        (la trampa cuantica respira), frente a la expansion monotona de
        c < 0 en difusion.

        OJO: aqui rho NO es la raiz de un momento de una densidad de
        probabilidad difusiva; es la escala del paquete cuantico, y el reloj
        es theta' = 1/rho^2, no D/rho^2.

    REGLA 2 (que da rho).  Ya no hay identidad de momentos cerrada como en
        los casos 1 y 2: rho sale de integrar Pinney. Sigue siendo a priori
        (solo necesita omega y los datos iniciales), que es lo que importa.

    REGLA 3 (que se cuelga).  c > 0 => aparece la FASE de Lewis-Riesenfeld:

            psi = rho^{-1/2} exp( i rho' x^2 / (2 rho) ) phi(xi, theta)

        El prefactor rho^{-1/2} conserva la norma; la exponencial es la fase
        de calibracion cuadratica. Ese factor es el que justifica el nombre
        Ermakov, y es el peldano decisivo de este caso.
    """

    def __init__(self, w0=1.0, amp=0.7, L=6.0, T=1.0, n_nodos=4001):
        self.w0, self.amp = w0, amp
        self.L, self.T = L, T
        self._construir(n_nodos)

    # --- frecuencia -------------------------------------------------
    def _w_np(self, t):
        return self.w0 + self.amp * np.sin(2 * np.pi * t)

    def omega(self, t):
        return self.w0 + self.amp * torch.sin(2 * math.pi * t)

    def theta_ad(self, t):
        """int_0^t omega ds, en forma cerrada. Solo la usa el control adiabatico."""
        w = 2 * math.pi
        return self.w0 * t + (self.amp / w) * (1.0 - torch.cos(w * t))

    # --- integrar Pinney --------------------------------------------
    def _construir(self, n):
        r"""Integra y = [rho, rho', theta] y tabula DOS interpolantes C^1.

        POR QUE DOS. La fase contiene rho' explicitamente, y el residuo
        exige derivar la fase respecto de t. Si rho' se obtuviera derivando
        por autograd el interpolante de rho, seria solo C^0 y su derivada
        tendria saltos en cada nodo. Interpolando rho' por separado, con su
        propia derivada exacta rho'' = 1/rho^3 - omega^2 rho, tanto rho como
        rho' quedan C^1 y el residuo sale limpio.
        """
        ts = np.linspace(0.0, self.T, n)
        r0 = 1.0 / math.sqrt(self._w_np(0.0))

        def rhs(t, y):
            rho, drho, _ = y
            return [drho,
                    1.0 / rho ** 3 - self._w_np(t) ** 2 * rho,
                    1.0 / rho ** 2]

        sol = solve_ivp(rhs, (0.0, self.T), [r0, 0.0, 0.0], t_eval=ts,
                        rtol=1e-12, atol=1e-14, method="DOP853")
        assert sol.success, sol.message
        rho, drho, th = sol.y
        d2rho = 1.0 / rho ** 3 - self._w_np(ts) ** 2 * rho

        self.ts, self.rho_nodos = ts, rho
        self._rho = HermiteC1(ts, rho, drho)         # C^1
        self._drho = HermiteC1(ts, drho, d2rho)      # C^1 tambien
        self._theta = HermiteC1(ts, th, 1.0 / rho ** 2)
        self.rho0 = float(rho[0])
        self.theta_T = float(th[-1])
        self.thad_T = float(self.theta_ad(torch.tensor(self.T, dtype=DTYPE)))

    def rho(self, t):
        return self._rho(t)

    def drho(self, t):
        return self._drho(t)

    def theta(self, t):
        return self._theta(t)

    # --- solucion exacta: estado base transportado -------------------
    def psi_exact(self, x, t):
        r"""psi = pi^{-1/4} rho^{-1/2} exp(i rho' x^2/(2 rho) - i theta/2 - x^2/(2 rho^2))

        Es el estado base del oscilador estatico, llevado por el marco de
        Ermakov. El -theta/2 es la fase dinamica con E_0 = 1/2.
        """
        rho, drho, th = self.rho(t), self.drho(t), self.theta(t)
        env = SQ * rho ** (-0.5) * torch.exp(-(x ** 2) / (2 * rho ** 2))
        S = drho * x ** 2 / (2 * rho) - th / 2
        return env * torch.cos(S), env * torch.sin(S)

    def psi_init(self, x):
        env = SQ * self.rho0 ** (-0.5) * torch.exp(-(x ** 2) / (2 * self.rho0 ** 2))
        return env, torch.zeros_like(env)

    # --- el control honesto: aproximacion adiabatica -----------------
    def psi_adiab(self, x, t):
        """psi_ad = (omega/pi)^{1/4} exp(-omega x^2/2) exp(-i int omega /2).

        Es el estado base INSTANTANEO con su fase dinamica. Vale cuando
        |omega'|/omega^2 << 1.
        """
        w = self.omega(t)
        env = (w / math.pi) ** 0.25 * torch.exp(-w * x ** 2 / 2)
        S = -self.theta_ad(t) / 2
        return env * torch.cos(S), env * torch.sin(S)


def bloque1(p: Cuantico):
    print("BLOQUE 1 -- Pinney con c = +1, y los dos interpolantes")
    print(f"  omega(t) = {p.w0} + {p.amp} sin(2 pi t)  =>  omega en "
          f"[{p.w0 - p.amp:.2f}, {p.w0 + p.amp:.2f}]")
    print(f"  rho(0) = 1/sqrt(omega(0)) = {p.rho0:.6f},  rho'(0) = 0")
    print(f"  rho oscila en [{p.rho_nodos.min():.4f}, {p.rho_nodos.max():.4f}]"
          f"  =>  respira un factor {p.rho_nodos.max() / p.rho_nodos.min():.3f}")
    print(f"  theta(T) = int dt/rho^2 = {p.theta_T:.6f}")

    # verificar Pinney por autograd a traves de los interpolantes
    t = torch.linspace(0.02, p.T - 0.02, 777).reshape(-1, 1).clone().requires_grad_(True)
    rho = p.rho(t)
    drho_ag = d1(rho, t)
    drho_in = p.drho(t)
    print(f"\n  (a) coherencia de los dos interpolantes:")
    print(f"      max |d(rho)/dt por autograd - interpolante de rho'| = "
          f"{float((drho_ag - drho_in).detach().abs().max()):.2e}")
    d2rho = d1(drho_in, t)
    res = d2rho + p.omega(t) ** 2 * rho - 1.0 / rho ** 3
    esc = float((1.0 / rho ** 3).detach().abs().mean())
    print(f"  (b) residuo de Pinney  rho'' + omega^2 rho - 1/rho^3:")
    print(f"      max |residuo| / escala = {float(res.detach().abs().max()) / esc:.2e}")
    return float(res.detach().abs().max()) / esc


# =====================================================================
# BLOQUE 2 -- la solucion exacta, la norma, y cuan mala es la adiabatica
# =====================================================================
def residuo_sch(fr, fi, x, t, p: Cuantico):
    """Parte el residuo complejo en dos residuos reales.

        i psi_t + 1/2 psi_xx - 1/2 omega^2 x^2 psi = 0,   psi = a + i b

        =>  r1 = -b_t + 1/2 a_xx - 1/2 omega^2 x^2 a
            r2 =  a_t + 1/2 b_xx - 1/2 omega^2 x^2 b
    """
    a_t, b_t = d1(fr, t), d1(fi, t)
    a_xx, b_xx = d1(d1(fr, x), x), d1(d1(fi, x), x)
    V = 0.5 * p.omega(t) ** 2 * x ** 2
    return (-b_t + 0.5 * a_xx - V * fr), (a_t + 0.5 * b_xx - V * fi)


def bloque2(p: Cuantico):
    print("\nBLOQUE 2 -- la solucion de referencia y el control adiabatico")

    # (a) la solucion exacta cumple la EDP
    xs = torch.linspace(-p.L, p.L, 401).reshape(-1, 1)
    ts = torch.linspace(0.05, p.T - 0.05, 41).reshape(1, -1)
    X = xs.repeat(1, ts.shape[1]).reshape(-1, 1).clone().requires_grad_(True)
    Tt = ts.repeat(xs.shape[0], 1).reshape(-1, 1).clone().requires_grad_(True)
    a, b = p.psi_exact(X, Tt)
    r1, r2 = residuo_sch(a, b, X, Tt, p)
    print(f"  (a) residuo de Schrodinger sobre la solucion exacta: "
          f"max = {float(torch.maximum(r1.detach().abs(), r2.detach().abs()).max()):.2e}")
    print("      (limitado por el interpolante, no por la formula)")

    # (b) norma conservada
    xg = torch.linspace(-12.0, 12.0, 60001).reshape(-1, 1)
    print(f"\n  (b) {'t':>6}{'norma ||psi||^2':>18}{'<x^2>':>12}")
    for tv in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tt = torch.full_like(xg, tv)
        a, b = p.psi_exact(xg, tt)
        dens = (a ** 2 + b ** 2).squeeze()
        nrm = float(torch.trapezoid(dens, xg.squeeze()))
        x2 = float(torch.trapezoid((xg.squeeze() ** 2) * dens, xg.squeeze()))
        print(f"      {tv:>6.2f}{nrm:>18.12f}{x2:>12.6f}")

    # (c) cuan NO adiabatico es el regimen
    tg = torch.linspace(0.0, p.T, 20001).reshape(-1, 1).clone().requires_grad_(True)
    w = p.omega(tg)
    dw = d1(w, tg)
    ratio = (dw.abs() / w ** 2).detach()
    print(f"\n  (c) max |omega'| / omega^2 = {float(ratio.max()):.4f}")
    print("      Muy por encima de 1: la aproximacion adiabatica no tiene")
    print("      derecho a funcionar en este regimen. Comprobemoslo.")

    # (d) error de la aproximacion adiabatica contra la solucion exacta
    xf = torch.linspace(-p.L, p.L, 400).reshape(-1, 1).repeat(1, 100).reshape(-1, 1)
    tf = torch.linspace(0.0, p.T, 100).reshape(1, -1).repeat(400, 1).reshape(-1, 1)
    with torch.no_grad():
        ae, be = p.psi_exact(xf, tf)
        aa, ba = p.psi_adiab(xf, tf)
        num = torch.sqrt(((aa - ae) ** 2 + (ba - be) ** 2).sum())
        den = torch.sqrt((ae ** 2 + be ** 2).sum())
    print(f"  (d) error relativo L2 de la aproximacion adiabatica: "
          f"{float(num / den):.3e}")
    print("      Este es el numero contra el que hay que comparar la red. No")
    print("      contra cero, y no contra una PINN sin estructura.")
    return float(ratio.max()), float(num / den)


# =====================================================================
# BLOQUE 3 -- los cuatro modelos
# =====================================================================
class _Base(torch.nn.Module):
    def __init__(self, p: Cuantico, width=64, depth=4):
        super().__init__()
        self.p = p
        self.mlp = MLP(2, 2, width, depth)      # dos salidas: Re y Im
        self._L, self._T = p.L, p.T
        self._thT, self._thadT = p.theta_T, p.thad_T

    def _split(self, z):
        f = self.mlp(z)
        return f[:, 0:1], f[:, 1:2]


class M1_QPINN(_Base):
    """Peldano 1: PINN compleja estandar.  psi = NN(x/L, t/T)"""
    nombre = "QPINN"

    def forward(self, x, t):
        return self._split(torch.cat([x / self._L, t / self._T], dim=1))


class M2_QPINNad(_Base):
    """Peldano 2: EL CONTROL HONESTO -- coordenadas adiabaticas.

        psi = omega^{1/4} NN( x sqrt(omega), theta_ad / theta_ad(T) )

    Es la estructura que sugiere la fisica de libro. No lleva nada de
    Ermakov. Si el regimen fuera adiabatico, ganaria.
    """
    nombre = "QPINN-ad"

    def forward(self, x, t):
        w = self.p.omega(t)
        z = torch.cat([x * torch.sqrt(w), self.p.theta_ad(t) / self._thadT], dim=1)
        fr, fi = self._split(z)
        amp = w ** 0.25
        return amp * fr, amp * fi


class M3_QEBsinfase(_Base):
    """Peldano 3: coordenadas de Ermakov + amplitud, SIN fase.

        psi = rho^{-1/2} NN(xi, theta)
    """
    nombre = "QEB-sinfase"

    def forward(self, x, t):
        rho = self.p.rho(t)
        fr, fi = self._split(torch.cat([x / rho, self.p.theta(t) / self._thT], dim=1))
        amp = rho ** (-0.5)
        return amp * fr, amp * fi


class M4_QEBNet(_Base):
    """Peldano 4: EB-Net completa, CON la fase de Lewis-Riesenfeld.

        psi = rho^{-1/2} exp( i rho' x^2 / (2 rho) ) NN(xi, theta)

    Multiplicar por e^{iS} mezcla las dos salidas de la red:
        Re = amp (cos S fr - sin S fi)
        Im = amp (sin S fr + cos S fi)
    """
    nombre = "QEB-Net"

    def forward(self, x, t):
        rho, drho = self.p.rho(t), self.p.drho(t)
        fr, fi = self._split(torch.cat([x / rho, self.p.theta(t) / self._thT], dim=1))
        S = drho * x ** 2 / (2 * rho)
        cs, sn = torch.cos(S), torch.sin(S)
        amp = rho ** (-0.5)
        return amp * (cs * fr - sn * fi), amp * (sn * fr + cs * fi)


ESCALERA = [M1_QPINN, M2_QPINNad, M3_QEBsinfase, M4_QEBNet]


def lotes_factory(p: Cuantico, cfg: Cfg):
    def lotes(gen):
        xc = (torch.rand(cfg.n_col, 1, generator=gen) * 2 - 1) * p.L
        tc = torch.rand(cfg.n_col, 1, generator=gen) * p.T
        xi = (torch.rand(cfg.n_ic, 1, generator=gen) * 2 - 1) * p.L
        ti = torch.zeros(cfg.n_ic, 1)
        sb = torch.randint(0, 2, (cfg.n_bc, 1), generator=gen).to(DTYPE) * 2 - 1
        return xc, tc, xi, ti, sb * p.L, torch.rand(cfg.n_bc, 1, generator=gen) * p.T
    return lotes


def perdida_factory(p: Cuantico, cfg: Cfg):
    """Perdida compleja: los dos residuos reales suman, y la condicion
    inicial se impone sobre parte real E imaginaria por separado."""
    def perdida(model, lote):
        xc, tc, xi, ti, xb, tb = lote
        xg = xc.clone().requires_grad_(True)
        tg = tc.clone().requires_grad_(True)
        a, b = model(xg, tg)
        r1, r2 = residuo_sch(a, b, xg, tg, p)
        l_r = (r1 ** 2).mean() + (r2 ** 2).mean()

        a0, b0 = model(xi, ti)
        e0r, e0i = p.psi_init(xi)
        l_ic = ((a0 - e0r) ** 2).mean() + ((b0 - e0i) ** 2).mean()

        ab, bb = model(xb, tb)
        ebr, ebi = p.psi_exact(xb, tb)
        l_bc = ((ab - ebr) ** 2).mean() + ((bb - ebi) ** 2).mean()

        L = l_r + cfg.lam_ic * l_ic + cfg.lam_bc * l_bc
        return L, {"res": float(l_r.detach()), "ic": float(l_ic.detach()),
                    "bc": float(l_bc.detach())}
    return perdida


def bloque3(p: Cuantico, cfg: Cfg):
    print("\nBLOQUE 3 -- la perdida compleja sobre modelos sin entrenar")
    lotes, perdida = lotes_factory(p, cfg), perdida_factory(p, cfg)
    lote = lotes(torch.Generator().manual_seed(7))
    print(f"    {'modelo':<14}{'params':>8}{'L total':>12}"
          f"{'residuo':>12}{'inicial':>12}{'frontera':>12}")
    for cls in ESCALERA:
        torch.manual_seed(0)
        m = cls(p)
        L, q = perdida(m, lote)
        print(f"    {cls.nombre:<14}{n_params(m):>8d}{float(L.detach()):>12.3e}"
              f"{q['res']:>12.3e}{q['ic']:>12.3e}{q['bc']:>12.3e}")


# =====================================================================
# BLOQUE 4 -- entrenar
# =====================================================================
def evaluar(model, p: Cuantico):
    with torch.no_grad():
        xf = torch.linspace(-p.L, p.L, 400).reshape(-1, 1).repeat(1, 100).reshape(-1, 1)
        tf = torch.linspace(0.0, p.T, 100).reshape(1, -1).repeat(400, 1).reshape(-1, 1)
        a, b = model(xf, tf)
        ae, be = p.psi_exact(xf, tf)
        num = torch.sqrt(((a - ae) ** 2 + (b - be) ** 2).sum())
        den = torch.sqrt((ae ** 2 + be ** 2).sum())
        return float(num / den)


def bloque4(p: Cuantico, cfg: Cfg, err_adiab=None):
    print("\nBLOQUE 4 -- entrenamiento de la escalera")
    lotes, perdida = lotes_factory(p, cfg), perdida_factory(p, cfg)
    filas = []
    for cls in ESCALERA:
        torch.manual_seed(cfg.seed)
        m = cls(p)
        seg = entrenar(m, perdida, lotes, cfg)
        err = evaluar(m, p)
        print(f"    {cls.nombre:<14} entrenado en {seg:6.1f} s   rel L2 = {err:.3e}")
        filas.append((cls.nombre, n_params(m), err))
    ref = ("LEER ASI: el salto que importa es QEB-sinfase -> QEB-Net, es decir\n"
           "  la fase cuadratica. Las coordenadas solas NO bastan aqui, al\n"
           "  contrario que en el caso 1.")
    if err_adiab is not None:
        ref += f"\n  Referencia analitica: la aproximacion adiabatica da {err_adiab:.3e}."
    tabla(filas, "CASO 3 (c=+1): escalera de ablacion", ref=ref)
    return filas


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bloque", type=int, default=0)
    ap.add_argument("--adam", type=int, default=2500)
    args = ap.parse_args()

    p = Cuantico()
    cfg = Cfg(n_adam=args.adam, lam_ic=10.0, lam_bc=10.0, seed=0)

    print("=" * 68)
    print("CASO 3:  i psi_t = -1/2 psi_xx + 1/2 omega^2(t) x^2 psi    (c = +1)")
    print("=" * 68)
    err_ad = None
    if args.bloque in (0, 1):
        bloque1(p)
    if args.bloque in (0, 2):
        _, err_ad = bloque2(p)
    if args.bloque in (0, 3):
        bloque3(p, cfg)
    if args.bloque in (0, 4):
        bloque4(p, cfg, err_ad)


if __name__ == "__main__":
    main()
