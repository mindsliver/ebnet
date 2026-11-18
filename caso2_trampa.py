"""
caso2_trampa.py -- CASO 2:  c = -1,  omega != 0
================================================

        u_t = D(t) u_xx + gamma(t) (x u)_x ,    gamma > 0

Difusion dentro de una trampa cuya rigidez se modula: pinzas opticas a
potencia variable, un coloide en un campo oscilante. Es el caso DECISIVO,
por dos razones:

  (a) el termino de arrastre gamma(t)(xu)_x contiene x EXPLICITAMENTE, de
      modo que NINGUNA reparametrizacion del tiempo s = s(t) lo puede
      absorber. La degeneracion del caso 1 desaparece y el control trivial
      deja de ser suficiente: de hecho, empeora.

  (b) rho ya NO tiene forma cerrada. Hay que integrar una EDO y meter el
      resultado en el grafo de autograd. Aqui es donde se usa el
      interpolante de Hermite C^1, y es el paso que mas problemas da.

Bloques:
    python caso2_trampa.py --bloque 1   # integrar la EDO de momentos y verificar el interpolante
    python caso2_trampa.py --bloque 2   # verificar Pinney y la reduccion
    python caso2_trampa.py --bloque 3   # la perdida con el termino de arrastre
    python caso2_trampa.py --bloque 4   # entrenar la escalera
    python caso2_trampa.py --bloque 5   # por que el control trivial no puede funcionar
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from scipy.integrate import solve_ivp

from comun import (DTYPE, Cfg, HermiteC1, MLP, d1, d2, entrenar, error_rel_l2,
                   malla, n_params, tabla)


# =====================================================================
# BLOQUE 1 -- rho por integracion + interpolante diferenciable
# =====================================================================
class Trampa:
    r"""u_t = D(t) u_xx + gamma(t) (x u)_x.

    REGLA 1 (rama de Pinney). Sigue siendo c < 0, pero ahora omega != 0:
    derivando la relacion rho rho' = D - gamma rho^2 se obtiene

        rho'' + (gamma' - gamma^2) rho - D'/rho = -D^2/rho^3,

    es decir una ecuacion de Pinney con omega^2 = gamma' - gamma^2 y
    constante negativa. La trampa entra en omega, no en c.

    REGLA 2 (identidad de momentos, lema 4). dm2/dt = 2D - 2 gamma m2, que
    es LINEAL. Sigue siendo a priori: rho = sqrt(m2) se obtiene integrando
    una EDO escalar desde m2(0), sin tocar la solucion de la EDP.

    REGLA 3 (que se cuelga). c < 0 => sin fase; prefactor rho^-1.
    """

    def __init__(self, D0=1.0, a=0.5, g0=1.0, b=0.8, fase=math.pi / 3,
                 L=6.0, T=1.0, m2_0=0.25, n_nodos=2001):
        self.D0, self.a = D0, a
        self.g0, self.b, self.fase = g0, b, fase
        self.L, self.T, self.m2_0 = L, T, m2_0
        self.alpha = 1.0
        self._construir(n_nodos)

    # --- coeficientes, en numpy (para la EDO) y en torch (para la red) ---
    def _D_np(self, t):
        return self.D0 + self.a * np.sin(2 * np.pi * t)

    def _g_np(self, t):
        return self.g0 * (1.0 + self.b * np.sin(2 * np.pi * t + self.fase))

    def _dD_np(self, t):
        return 2 * np.pi * self.a * np.cos(2 * np.pi * t)

    def _dg_np(self, t):
        return self.g0 * self.b * 2 * np.pi * np.cos(2 * np.pi * t + self.fase)

    def D(self, t):
        return self.D0 + self.a * torch.sin(2 * math.pi * t)

    def gamma(self, t):
        return self.g0 * (1.0 + self.b * torch.sin(2 * math.pi * t + self.fase))

    def tau(self, t):
        """tau = int_0^t D. Solo la usa el modelo de control PINN-tau."""
        w = 2 * math.pi
        return self.D0 * t + (self.a / w) * (1.0 - torch.cos(w * t))

    # --- integracion de la EDO de momentos y tabulado ----------------
    def _construir(self, n):
        r"""Integra  [m2, theta]  y tabula (rho, rho') y (theta, theta').

        Las derivadas nodales NO se obtienen por diferencias finitas: se
        evaluan con las propias ecuaciones,

            rho' = (D - gamma m2) / rho          (de rho rho' = D - gamma rho^2)
            theta' = D / m2

        Esto es importante: el interpolante de Hermite es exacto en valor Y
        derivada en los nodos, asi que al alimentarlo con las derivadas
        exactas se hereda la precision del integrador.
        """
        ts = np.linspace(0.0, self.T, n)

        def rhs(t, y):
            m2, _ = y
            return [2 * self._D_np(t) - 2 * self._g_np(t) * m2,
                    self._D_np(t) / m2]

        sol = solve_ivp(rhs, (0.0, self.T), [self.m2_0, 0.0], t_eval=ts,
                        rtol=1e-12, atol=1e-14, method="DOP853")
        assert sol.success, sol.message
        m2, th = sol.y[0], sol.y[1]
        rho = np.sqrt(m2)
        drho = (self._D_np(ts) - self._g_np(ts) * m2) / rho     # derivada exacta
        dth = self._D_np(ts) / m2                               # derivada exacta

        self.ts, self.m2_nodos = ts, m2
        self._rho = HermiteC1(ts, rho, drho)
        self._theta = HermiteC1(ts, th, dth)
        self.rho0 = float(rho[0])
        self.rho_T = float(rho[-1])
        self.theta_T = float(th[-1])
        self.tau_T = float(self.tau(torch.tensor(self.T, dtype=DTYPE)))
        self._m2 = HermiteC1(ts, m2, 2 * self._D_np(ts) - 2 * self._g_np(ts) * m2)

    def rho(self, t):
        return self._rho(t)

    def theta(self, t):
        return self._theta(t)

    # --- solucion exacta: gaussiana de varianza m2(t) ----------------
    def u_exact(self, x, t):
        s2 = self._m2(t)
        return torch.exp(-(x ** 2) / (2 * s2)) / torch.sqrt(2 * math.pi * s2)

    def u_init(self, x):
        return torch.exp(-(x ** 2) / (2 * self.m2_0)) / math.sqrt(2 * math.pi * self.m2_0)


def bloque1(p: Trampa):
    print("BLOQUE 1 -- rho por integracion, y el interpolante diferenciable")
    print(f"  D(t) = {p.D0} + {p.a} sin(2 pi t)")
    print(f"  gamma(t) = {p.g0} (1 + {p.b} sin(2 pi t + pi/3))")
    print(f"  m2(0) = {p.m2_0:.4f}  =>  rho(0) = {p.rho0:.6f}")
    print(f"  rho(T) = {p.rho_T:.6f},   theta(T) = {p.theta_T:.6f}")
    print(f"  rho oscila en [{math.sqrt(p.m2_nodos.min()):.4f}, "
          f"{math.sqrt(p.m2_nodos.max()):.4f}]  (la trampa la hace respirar)")

    # (a) el interpolante reproduce los nodos
    tt = torch.as_tensor(p.ts, dtype=DTYPE).reshape(-1, 1)
    rho_i = p.rho(tt).squeeze().numpy()
    err_nodos = np.abs(rho_i - np.sqrt(p.m2_nodos)).max()
    print(f"\n  (a) error del interpolante en los nodos: {err_nodos:.2e}")

    # (b) el interpolante entre nodos, contra una reintegracion mas fina
    ts_f = np.linspace(0.0, p.T, 7777)

    def rhs(t, y):
        m2, _ = y
        return [2 * p._D_np(t) - 2 * p._g_np(t) * m2, p._D_np(t) / m2]
    ref = solve_ivp(rhs, (0.0, p.T), [p.m2_0, 0.0], t_eval=ts_f,
                    rtol=1e-13, atol=1e-15, method="DOP853")
    rho_ref = np.sqrt(ref.y[0])
    rho_int = p.rho(torch.as_tensor(ts_f, dtype=DTYPE).reshape(-1, 1)).squeeze().numpy()
    print(f"  (b) error del interpolante ENTRE nodos: "
          f"{np.abs(rho_int - rho_ref).max():.2e}")

    # (c) LA VERIFICACION QUE IMPORTA: autograd a traves del interpolante
    tg = torch.linspace(0.02, p.T - 0.02, 999).reshape(-1, 1).clone().requires_grad_(True)
    rho_t = p.rho(tg)
    drho_ag = d1(rho_t, tg)                         # derivada por autograd
    drho_ex = (p.D(tg) - p.gamma(tg) * rho_t ** 2) / rho_t   # derivada exacta
    # OJO con la metrica: rho' cruza cero cuando D = gamma rho^2, de modo
    # que el error RELATIVO PUNTUAL se dispara ahi sin que nada este mal.
    # Lo honesto es medir el error absoluto contra la ESCALA de rho'.
    dif = (drho_ag - drho_ex).detach().abs()
    esc = drho_ex.detach().abs().max()
    rel = float(dif.max() / esc)
    print(f"  (c) autograd d(rho)/dt vs la derivada exacta:")
    print(f"      max |diferencia| = {float(dif.max()):.2e}   "
          f"(escala de rho' = {float(esc):.3f})")
    print(f"      error relativo a esa escala = {rel:.2e}")
    print("      Si este numero no es pequenio, el embedding esta roto y la red")
    print("      entrenara contra un residuo equivocado SIN dar ningun error.")
    return rel


# =====================================================================
# BLOQUE 2 -- Pinney y la reduccion
# =====================================================================
def bloque2(p: Trampa):
    print("\nBLOQUE 2 -- rho cumple Pinney, y la solucion reducida es estacionaria")

    # (a) Pinney, verificado EN LOS NODOS con algebra exacta.
    #     Derivando rho rho' = D - gamma rho^2 se obtiene
    #        rho'' = (D' - gamma' rho^2 - 2 gamma rho rho' - rho'^2) / rho
    #     y sustituyendo rho' = (D - gamma rho^2)/rho queda
    #        rho'' + (gamma' - gamma^2) rho - D'/rho = -D^2/rho^3 .
    #     Esta comprobacion no usa autograd, asi que sale a precision de
    #     maquina: verifica la DERIVACION, no el interpolante.
    ts = p.ts[1:-1]
    m2 = p.m2_nodos[1:-1]
    rho_n = np.sqrt(m2)
    D, g = p._D_np(ts), p._g_np(ts)
    dD, dg = p._dD_np(ts), p._dg_np(ts)
    drho_n = (D - g * m2) / rho_n
    d2rho_n = (dD - dg * m2 - 2 * g * rho_n * drho_n - drho_n ** 2) / rho_n
    izq = d2rho_n + (dg - g ** 2) * rho_n - dD / rho_n
    der = -(D ** 2) / rho_n ** 3
    esc = np.abs(der).mean()
    print("  (a) Pinney  rho'' + (gamma'-gamma^2) rho - D'/rho = -D^2/rho^3")
    print(f"      (en los nodos, algebra exacta)  max |izq-der|/escala = "
          f"{np.abs(izq - der).max() / esc:.2e}")

    # (a-bis) la misma identidad, pero derivando el interpolante por autograd.
    #     Sale peor porque la SEGUNDA derivada de un Hermite C^1 solo es
    #     O(h^2) y salta en los nodos. Es normal; no es un error.
    t = torch.linspace(0.05, p.T - 0.05, 500).reshape(-1, 1).clone().requires_grad_(True)
    rho = p.rho(t)
    d2rho = d1(d1(rho, t), t)
    gt, Dt = p.gamma(t), p.D(t)
    res_ag = d2rho + (d1(gt, t) - gt ** 2) * rho - d1(Dt, t) / rho + Dt ** 2 / rho ** 3
    print(f"      (por autograd sobre el interpolante)  mediana = "
          f"{float(res_ag.detach().abs().median()) / float(esc):.2e}")

    # (b) v = rho * u  es independiente de theta
    xi = torch.linspace(-5.0, 5.0, 11).reshape(-1, 1)
    vs = []
    for tv in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tt = torch.full_like(xi, tv)
        rho_v = p.rho(tt)
        vs.append(rho_v ** p.alpha * p.u_exact(xi * rho_v, tt))
    V = torch.cat(vs, dim=1)
    var = float((V.max(dim=1).values - V.min(dim=1).values).max())
    print(f"  (b) variacion maxima de v(xi,theta) a lo largo de theta: {var:.2e}")
    print("      => con el embedding la red aprende un perfil FIJO, aunque la")
    print("         densidad fisica este oscilando con la trampa.")
    return var


# =====================================================================
# BLOQUE 3 -- modelos y perdida
# =====================================================================
class _Base(torch.nn.Module):
    def __init__(self, p: Trampa, width=64, depth=4):
        super().__init__()
        self.p = p
        self.mlp = MLP(2, 1, width, depth)
        self._L, self._T = p.L, p.T
        self._tauT, self._thT = p.tau_T, p.theta_T


class M1_PINN(_Base):
    nombre = "PINN"

    def forward(self, x, t):
        return self.mlp(torch.cat([x / self._L, t / self._T], dim=1))


class M2_PINNtau(_Base):
    """EL CONTROL. Aqui se espera que NO ayude, e incluso que estorbe:
    tau absorbe D pero no puede hacer nada contra gamma(t)(xu)_x."""
    nombre = "PINN-tau"

    def forward(self, x, t):
        return self.mlp(torch.cat([x / self._L, self.p.tau(t) / self._tauT], dim=1))


class M3_EBcoords(_Base):
    nombre = "EB-coords"

    def forward(self, x, t):
        rho = self.p.rho(t)
        return self.mlp(torch.cat([x / rho, self.p.theta(t) / self._thT], dim=1))


class M4_EBNet(_Base):
    """El prefactor rho^-1 es lo que aqui carga la ganancia."""
    nombre = "EB-Net"

    def forward(self, x, t):
        rho = self.p.rho(t)
        v = self.mlp(torch.cat([x / rho, self.p.theta(t) / self._thT], dim=1))
        return v / rho ** self.p.alpha


ESCALERA = [M1_PINN, M2_PINNtau, M3_EBcoords, M4_EBNet]


def residuo(model, x, t):
    """r = u_t - D u_xx - gamma (x u)_x,  con (x u)_x = u + x u_x."""
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = model(x, t)
    u_t = d1(u, t)
    u_x = d1(u, x)
    u_xx = d1(u_x, x)
    p = model.p
    return u_t - p.D(t) * u_xx - p.gamma(t) * (u + x * u_x)


def lotes_factory(p: Trampa, cfg: Cfg):
    def lotes(gen):
        xc = (torch.rand(cfg.n_col, 1, generator=gen) * 2 - 1) * p.L
        tc = torch.rand(cfg.n_col, 1, generator=gen) * p.T
        xi = (torch.rand(cfg.n_ic, 1, generator=gen) * 2 - 1) * p.L
        ti = torch.zeros(cfg.n_ic, 1)
        sb = torch.randint(0, 2, (cfg.n_bc, 1), generator=gen).to(DTYPE) * 2 - 1
        return xc, tc, xi, ti, sb * p.L, torch.rand(cfg.n_bc, 1, generator=gen) * p.T
    return lotes


def perdida_factory(p: Trampa, cfg: Cfg):
    def perdida(model, lote):
        xc, tc, xi, ti, xb, tb = lote
        l_r = (residuo(model, xc, tc) ** 2).mean()
        l_ic = ((model(xi, ti) - p.u_init(xi)) ** 2).mean()
        l_bc = ((model(xb, tb) - p.u_exact(xb, tb)) ** 2).mean()
        L = l_r + cfg.lam_ic * l_ic + cfg.lam_bc * l_bc
        return L, {"res": float(l_r.detach()), "ic": float(l_ic.detach()),
                    "bc": float(l_bc.detach())}
    return perdida


def bloque3(p: Trampa, cfg: Cfg):
    print("\nBLOQUE 3 -- la perdida, ahora con el termino de arrastre")
    lotes, perdida = lotes_factory(p, cfg), perdida_factory(p, cfg)
    gen = torch.Generator().manual_seed(7)
    lote = lotes(gen)
    print(f"    {'modelo':<12}{'params':>8}{'L total':>12}"
          f"{'residuo':>12}{'inicial':>12}{'frontera':>12}")
    for cls in ESCALERA:
        torch.manual_seed(0)
        m = cls(p)
        L, q = perdida(m, lote)
        print(f"    {cls.nombre:<12}{n_params(m):>8d}{float(L.detach()):>12.3e}"
              f"{q['res']:>12.3e}{q['ic']:>12.3e}{q['bc']:>12.3e}")


# =====================================================================
# BLOQUE 4 -- entrenar
# =====================================================================
def evaluar(model, p: Trampa):
    with torch.no_grad():
        xf, tf = malla(p.L, 0.0, p.T)
        return error_rel_l2(model(xf, tf), p.u_exact(xf, tf))


def bloque4(p: Trampa, cfg: Cfg):
    print("\nBLOQUE 4 -- entrenamiento de la escalera")
    lotes, perdida = lotes_factory(p, cfg), perdida_factory(p, cfg)
    filas = []
    for cls in ESCALERA:
        torch.manual_seed(cfg.seed)
        m = cls(p)
        seg = entrenar(m, perdida, lotes, cfg)
        err = evaluar(m, p)
        print(f"    {cls.nombre:<12} entrenado en {seg:6.1f} s   rel L2 = {err:.3e}")
        filas.append((cls.nombre, n_params(m), err))
    tabla(filas, "CASO 2 (c=-1, omega!=0): escalera de ablacion",
          ref="LEER ASI: aqui el control PINN-tau deja de ayudar, y el salto\n"
              "  EB-coords -> EB-Net (el prefactor rho^-1) es el que carga la\n"
              "  ganancia. Compare con el caso 1, donde ese mismo salto era casi nulo.")
    return filas


# =====================================================================
# BLOQUE 5 -- por que ninguna reparametrizacion del tiempo sirve
# =====================================================================
def bloque5(p: Trampa):
    print("\nBLOQUE 5 -- por que el control trivial no puede funcionar")
    print("  La EDP es   u_t = D(t) u_xx + gamma(t) (x u)_x .")
    print("  Un cambio s = s(t) convierte u_t en s'(t) u_s, de modo que puede")
    print("  normalizar UN coeficiente, no dos; y el termino de arrastre lleva x")
    print("  explicitamente, asi que ninguna funcion de t lo absorbe.")
    xs = torch.linspace(-p.L, p.L, 2001).reshape(-1, 1)
    print(f"\n    {'t':>6}{'||D u_xx||':>14}{'||gamma (xu)_x||':>18}{'razon':>10}")
    for tv in [0.1, 0.3, 0.5, 0.7, 0.9]:
        tt = torch.full_like(xs, tv).clone().requires_grad_(True)
        xg = xs.clone().requires_grad_(True)
        u = p.u_exact(xg, tt)
        u_x = d1(u, xg)
        u_xx = d1(u_x, xg)
        dif = (p.D(tt) * u_xx).detach()
        arr = (p.gamma(tt) * (u + xg * u_x)).detach()
        nd, na = float(dif.norm()), float(arr.norm())
        print(f"    {tv:>6.2f}{nd:>14.4f}{na:>18.4f}{na / nd:>10.3f}")
    print("    El arrastre es del mismo orden que la difusion: no es una")
    print("    correccion pequenia que se pueda ignorar.")


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bloque", type=int, default=0)
    ap.add_argument("--adam", type=int, default=2500)
    args = ap.parse_args()

    p = Trampa()
    cfg = Cfg(n_adam=args.adam, lam_ic=10.0, lam_bc=10.0, seed=0)

    print("=" * 68)
    print("CASO 2:  u_t = D(t) u_xx + gamma(t) (x u)_x     (c = -1, omega != 0)")
    print("=" * 68)
    if args.bloque in (0, 1):
        bloque1(p)
    if args.bloque in (0, 2):
        bloque2(p)
    if args.bloque in (0, 3):
        bloque3(p, cfg)
    if args.bloque in (0, 4):
        bloque4(p, cfg)
    if args.bloque in (0, 5):
        bloque5(p)


if __name__ == "__main__":
    main()
