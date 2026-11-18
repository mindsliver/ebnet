"""
caso1_difusion.py -- CASO 1:  c = -1,  omega = 0
=================================================

        u_t = D(t) u_xx ,      x in [-L, L],   t in [0, T]

Este es el caso DEGENERADO, y por eso es el primero: ensena el pipeline
completo y, al mismo tiempo, ensena por que la escalera de ablacion no es
opcional. Aqui rho tiene FORMA CERRADA, de modo que no hace falta ni
integrador ni interpolante: es el caso donde montar el embedding cuesta
cinco lineas.

Se ejecuta por bloques:

    python caso1_difusion.py --bloque 1     # el problema y la identidad de momentos
    python caso1_difusion.py --bloque 2     # verificar la reduccion a Ornstein-Uhlenbeck
    python caso1_difusion.py --bloque 3     # inspeccionar la funcion de perdida
    python caso1_difusion.py --bloque 4     # entrenar la escalera de ablacion
    python caso1_difusion.py --bloque 5     # el peso implicito rho^-3
    python caso1_difusion.py                # todos
"""
from __future__ import annotations

import argparse
import math

import torch

from comun import (DTYPE, Cfg, MLP, d1, d2, entrenar, error_rel_l2, malla,
                   n_params, tabla)


# =====================================================================
# BLOQUE 1 -- El problema, y rho a priori
# =====================================================================
class Difusion:
    r"""u_t = D(t) u_xx con D(t) = D0 + a sin(2 pi f t).

    Se elige esta D porque tau(t) = int_0^t D tiene forma cerrada. Con
    cualquier otra D el pipeline es el mismo pero tau sale por cuadratura.

    REGLA 1 (rama de Pinney).  c = -1 y omega = 0, de modo que la ecuacion
    de Pinney  rho'' + omega^2 rho = c/rho^3  se reduce a rho'' = -1/rho^3,
    cuya solucion con los datos que impone el problema es la raiz del
    segundo momento.

    REGLA 2 (identidad de momentos, lema 1).  m2(t) = m2(0) + 2 tau(t)
    EXACTAMENTE. De aqui sale todo: rho = sqrt(m2) se puede calcular ANTES
    de entrenar, solo con la condicion inicial y con D. No hace falta red
    auxiliar ni conocer la solucion.

    REGLA 3 (que se le cuelga al tronco).  c < 0 => no hay fase; lo que se
    cuelga es el prefactor de amplitud rho^-alpha con alpha = 1 (dimension
    espacial 1, masa conservada).
    """

    def __init__(self, D0=1.0, a=0.5, f=1.0, L=5.0, T=1.0, sigma0=0.5):
        assert D0 - abs(a) > 0.0, "D(t) debe ser > 0 (parabolicidad)"
        self.D0, self.a, self.f = D0, a, f
        self.L, self.T, self.sigma0 = L, T, sigma0
        self.alpha = 1.0
        self.m2_0 = sigma0 ** 2
        self.rho0 = math.sqrt(self.m2_0)
        tT = torch.tensor(T, dtype=DTYPE)
        self.tau_T = float(self.tau(tT))
        self.theta_T = float(self.theta(tT))

    # --- coeficiente y tiempo difusivo -------------------------------
    def D(self, t):
        return self.D0 + self.a * torch.sin(2 * math.pi * self.f * t)

    def tau(self, t):
        """tau(t) = int_0^t D(s) ds. OJO: tiene unidades de longitud^2."""
        w = 2 * math.pi * self.f
        return self.D0 * t + (self.a / w) * (1.0 - torch.cos(w * t))

    # --- escala de Ermakov-Pinney y su reloj -------------------------
    def rho(self, t):
        return torch.sqrt(self.m2_0 + 2.0 * self.tau(t))

    def theta(self, t):
        return torch.log(self.rho(t) / self.rho0)

    # --- solucion exacta (nucleo de calor reescalado) ----------------
    def u_exact(self, x, t):
        s2 = self.m2_0 + 2.0 * self.tau(t)
        return torch.exp(-(x ** 2) / (2 * s2)) / torch.sqrt(2 * math.pi * s2)

    def u_init(self, x):
        return self.u_exact(x, torch.zeros_like(x))


def bloque1(p: Difusion):
    print("BLOQUE 1 -- el problema y la escala a priori")
    print(f"  D(t) = {p.D0} + {p.a} sin(2 pi t)   =>  D en "
          f"[{p.D0 - abs(p.a):.2f}, {p.D0 + abs(p.a):.2f}]")
    print(f"  dominio: x en [-{p.L}, {p.L}],  t en [0, {p.T}]")
    print(f"  m2(0) = sigma0^2 = {p.m2_0:.4f}   =>  rho(0) = {p.rho0:.4f}")
    print(f"  tau(T) = {p.tau_T:.6f}  (unidades de longitud^2)")
    print(f"  rho(T) = {float(p.rho(torch.tensor(p.T))):.6f}")
    print(f"  theta(T) = log(rho_T/rho_0) = {p.theta_T:.6f}  (adimensional)")

    # --- verificacion numerica de la identidad de momentos -----------
    # m2(t) = int x^2 u dx / int u dx, calculado sobre la solucion exacta,
    # contra la prediccion m2(0) + 2 tau(t).
    xs = torch.linspace(-40.0, 40.0, 400001).reshape(-1, 1)
    print("\n  verificacion de la identidad de momentos  m2(t) = m2(0) + 2 tau(t):")
    print(f"    {'t':>6}{'m2 por cuadratura':>22}{'m2(0)+2tau':>16}{'dif rel':>12}")
    peor = 0.0
    for tv in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tt = torch.full_like(xs, tv)
        u = p.u_exact(xs, tt)
        masa = torch.trapezoid(u.squeeze(), xs.squeeze())
        m2_num = float(torch.trapezoid((xs ** 2 * u).squeeze(), xs.squeeze()) / masa)
        m2_teo = float(p.m2_0 + 2 * p.tau(torch.tensor(tv, dtype=DTYPE)))
        dif = abs(m2_num - m2_teo) / m2_teo
        peor = max(peor, dif)
        print(f"    {tv:>6.2f}{m2_num:>22.12f}{m2_teo:>16.12f}{dif:>12.2e}")
    print(f"    peor diferencia relativa: {peor:.2e}  "
          f"(limitada por la cuadratura, no por la identidad)")
    return peor


# =====================================================================
# BLOQUE 2 -- La reduccion: por que el embedding ayuda
# =====================================================================
def bloque2(p: Difusion):
    r"""Comprueba que v(xi,theta) = rho * u(rho xi, t) NO depende de theta.

    Teoria: con xi = x/rho y theta = log(rho/rho_0), la EDP de coeficientes
    variables se vuelve el operador AUTONOMO de Ornstein-Uhlenbeck

        v_theta = v_xixi + xi v_xi + v = d_xi( v_xi + xi v ).

    Para este dato inicial la solucion reducida es ademas un PUNTO FIJO de
    ese operador, es decir constante en theta. Eso es exactamente lo que la
    red tiene que aprender cuando se le da el embedding: un perfil fijo en
    lugar de una familia que se ensancha. Esta es toda la razon del metodo.
    """
    print("\nBLOQUE 2 -- la solucion reducida es estacionaria")
    xi = torch.linspace(-6.0, 6.0, 13).reshape(-1, 1)
    print(f"    {'t':>6}{'theta':>10}   v(xi,theta) en xi = -6 .. 6 (muestra)")
    vs = []
    for tv in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tt = torch.full_like(xi, tv)
        rho = p.rho(tt)
        v = rho ** p.alpha * p.u_exact(xi * rho, tt)      # v = rho^alpha u
        vs.append(v)
        th = float(p.theta(torch.tensor(tv, dtype=DTYPE)))
        print(f"    {tv:>6.2f}{th:>10.4f}   " +
              " ".join(f"{float(v[i]):.6f}" for i in (3, 5, 6, 7, 9)))
    V = torch.cat(vs, dim=1)
    var = float((V.max(dim=1).values - V.min(dim=1).values).max())
    print(f"    variacion maxima de v a lo largo de theta: {var:.3e}")

    # el operador de OU aplicado a v tambien se anula
    xi_g = torch.linspace(-6.0, 6.0, 2001).reshape(-1, 1).clone().requires_grad_(True)
    v = torch.exp(-(xi_g ** 2) / 2) / math.sqrt(2 * math.pi)
    flujo = d1(v, xi_g) + xi_g * v                 # v_xi + xi v
    res_ou = d1(flujo, xi_g)                       # d_xi( v_xi + xi v )
    print(f"    max |d_xi(v_xi + xi v)| sobre el perfil reducido: "
          f"{float(res_ou.detach().abs().max()):.3e}")
    print("    => v_theta = 0 y el miembro derecho = 0: el perfil es punto fijo.")
    return var


# =====================================================================
# BLOQUE 3 -- Los cuatro modelos de la escalera
# =====================================================================
class _Base(torch.nn.Module):
    """Normalizaciones CONGELADAS en la construccion.

    Si mas tarde se evalua el modelo sobre otra D(t) o sobre un horizonte
    mas largo, estas constantes NO deben moverse: moverlas reescalaria la
    entrada respecto de lo que la red aprendio y falsearia la comparacion.
    """

    def __init__(self, p: Difusion, width=64, depth=4, out_dim=1, in_dim=2):
        super().__init__()
        self.p = p
        self.mlp = MLP(in_dim, out_dim, width, depth)
        self._L, self._T = p.L, p.T
        self._tauT, self._thT = p.tau_T, p.theta_T


class M1_PINN(_Base):
    """Peldano 1: PINN estandar.      u = NN(x/L, t/T)"""
    nombre = "PINN"

    def forward(self, x, t):
        return self.mlp(torch.cat([x / self._L, t / self._T], dim=1))


class M2_PINNtau(_Base):
    """Peldano 2: EL CONTROL.         u = NN(x/L, tau/tau_T)

    Solo reparametriza el tiempo. No lleva ninguna estructura de Ermakov.
    Es el modelo que decide si el resultado del caso 1 significa algo.
    """
    nombre = "PINN-tau"

    def forward(self, x, t):
        return self.mlp(torch.cat([x / self._L, self.p.tau(t) / self._tauT], dim=1))


class M3_EBcoords(_Base):
    """Peldano 3: coordenadas de Ermakov SIN prefactor.  u = NN(xi, theta)"""
    nombre = "EB-coords"

    def forward(self, x, t):
        rho = self.p.rho(t)
        return self.mlp(torch.cat([x / rho, self.p.theta(t) / self._thT], dim=1))


class M4_EBNet(_Base):
    """Peldano 4: EB-Net completa.    u = rho^-alpha NN(xi, theta)

    La capa de embedding no tiene pesos: la dicta el operador del problema.
    """
    nombre = "EB-Net"

    def forward(self, x, t):
        rho = self.p.rho(t)
        v = self.mlp(torch.cat([x / rho, self.p.theta(t) / self._thT], dim=1))
        return v / rho ** self.p.alpha


ESCALERA = [M1_PINN, M2_PINNtau, M3_EBcoords, M4_EBNet]


# =====================================================================
# BLOQUE 3b -- La funcion de perdida, pieza por pieza
# =====================================================================
def residuo(model, x, t):
    """r = u_t - D(t) u_xx, por autograd y en coordenadas FISICAS.

    PUNTO CLAVE DE HONESTIDAD: los cuatro modelos minimizan este mismo
    residuo, en las mismas variables (x,t). NINGUN modelo se entrena sobre
    la ecuacion reducida. El embedding solo cambia como la red representa u;
    no cambia la ecuacion que se le exige cumplir.
    """
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = model(x, t)
    u_t = d1(u, t)
    u_xx = d2(u, x)
    return u_t - model.p.D(t) * u_xx


def lotes_factory(p: Difusion, cfg: Cfg):
    def lotes(gen):
        xc = (torch.rand(cfg.n_col, 1, generator=gen) * 2 - 1) * p.L
        tc = torch.rand(cfg.n_col, 1, generator=gen) * p.T
        xi = (torch.rand(cfg.n_ic, 1, generator=gen) * 2 - 1) * p.L
        ti = torch.zeros(cfg.n_ic, 1)
        sb = torch.randint(0, 2, (cfg.n_bc, 1), generator=gen).to(DTYPE) * 2 - 1
        xb, tb = sb * p.L, torch.rand(cfg.n_bc, 1, generator=gen) * p.T
        return xc, tc, xi, ti, xb, tb
    return lotes


def perdida_factory(p: Difusion, cfg: Cfg):
    r"""L = <r^2> + lam_ic <(u - u0)^2> + lam_bc <(u - u_exacta)^2>

    Los tres terminos y por que estan:

      * residuo   : impone la EDP en el interior. Es el unico termino que
                    contiene fisica; sin el la red interpola datos.
      * inicial   : fija la solucion. La EDP sola tiene infinitas
                    soluciones; sin este termino el optimizador encuentra
                    u = 0, que anula el residuo perfectamente.
      * frontera  : cierra el problema en x = +-L. Aqui usamos la solucion
                    exacta porque el dominio es truncado y la cola gaussiana
                    no es cero exactamente; con L = 5 y rho <= 1.5 vale ~1e-11,
                    asi que podria ponerse 0 sin cambiar nada.

    Los pesos lam NO son decorativos: ponderan terminos con unidades
    distintas. El residuo tiene unidades de u/tiempo y los otros dos de u.
    Se eligen por barrido sobre una semilla aparte y se aplica el MISMO
    valor a los cuatro modelos.
    """
    def perdida(model, lote):
        xc, tc, xi, ti, xb, tb = lote
        r = residuo(model, xc, tc)
        l_r = (r ** 2).mean()
        l_ic = ((model(xi, ti) - p.u_init(xi)) ** 2).mean()
        l_bc = ((model(xb, tb) - p.u_exact(xb, tb)) ** 2).mean()
        L = l_r + cfg.lam_ic * l_ic + cfg.lam_bc * l_bc
        return L, {"res": float(l_r.detach()), "ic": float(l_ic.detach()),
                    "bc": float(l_bc.detach())}
    return perdida


def bloque3(p: Difusion, cfg: Cfg):
    print("\nBLOQUE 3 -- la funcion de perdida sobre un modelo sin entrenar")
    torch.manual_seed(0)
    lotes = lotes_factory(p, cfg)
    perdida = perdida_factory(p, cfg)
    gen = torch.Generator().manual_seed(7)
    lote = lotes(gen)
    print(f"    {'modelo':<12}{'params':>8}{'L total':>12}"
          f"{'residuo':>12}{'inicial':>12}{'frontera':>12}")
    for cls in ESCALERA:
        m = cls(p)
        L, partes = perdida(m, lote)
        print(f"    {cls.nombre:<12}{n_params(m):>8d}{float(L.detach()):>12.3e}"
              f"{partes['res']:>12.3e}{partes['ic']:>12.3e}{partes['bc']:>12.3e}")
    print("    Los cuatro arrancan del mismo orden de magnitud y con el mismo")
    print("    numero de parametros: el embedding no anade pesos.")


# =====================================================================
# BLOQUE 4 -- Entrenar la escalera
# =====================================================================
def evaluar(model, p: Difusion, t_hi=None):
    with torch.no_grad():
        xf, tf = malla(p.L, 0.0, p.T if t_hi is None else t_hi)
        return error_rel_l2(model(xf, tf), p.u_exact(xf, tf))


def bloque4(p: Difusion, cfg: Cfg):
    print("\nBLOQUE 4 -- entrenamiento de la escalera de ablacion")
    lotes = lotes_factory(p, cfg)
    perdida = perdida_factory(p, cfg)
    filas = []
    for cls in ESCALERA:
        torch.manual_seed(cfg.seed)
        m = cls(p)
        seg = entrenar(m, perdida, lotes, cfg)
        err = evaluar(m, p)
        print(f"    {cls.nombre:<12} entrenado en {seg:6.1f} s   "
              f"rel L2 = {err:.3e}")
        filas.append((cls.nombre, n_params(m), err))
    tabla(filas, "CASO 1 (c=-1, omega=0): escalera de ablacion",
          ref="LEER ASI: el salto PINN -> PINN-tau es el control. Si se come la\n"
              "  mayor parte de la ganancia, esta familia NO sostiene ninguna\n"
              "  afirmacion sobre el embedding. Ese es el resultado del caso 1.")
    return filas


# =====================================================================
# BLOQUE 5 -- El peso implicito rho^-3
# =====================================================================
def bloque5(p: Difusion):
    r"""La identidad exacta del residuo

        u_t - D u_xx = (D / rho^3) [ v_theta - d_xi(v_xi + xi v) ]

    dice que minimizar el residuo FISICO equivale a minimizar el residuo
    REDUCIDO con un peso (D/rho^3)^2. Como rho crece, los tiempos tempranos
    pesan mas. Hay que saberlo: no es un hiperparametro que alguien ajusto,
    es una consecuencia del cambio de variables.
    """
    print("\nBLOQUE 5 -- el peso implicito que introduce el cambio de variables")
    r0 = float(p.rho(torch.tensor(0.0, dtype=DTYPE)))
    rT = float(p.rho(torch.tensor(p.T, dtype=DTYPE)))
    print(f"    rho(0) = {r0:.4f},  rho(T) = {rT:.4f},  rho_T/rho_0 = {rT / r0:.4f}")
    print(f"    el residuo fisico pondera t=0 frente a t=T por "
          f"(rho_T/rho_0)^3 = {(rT / r0) ** 3:.2f}")
    print("    Para comprobar que la ganancia NO viene de este peso, se repite")
    print("    el experimento con el residuo reponderado por rho^3/D (igual para")
    print("    los cuatro modelos) y se verifica que el ORDEN de la escalera")
    print("    sobrevive. Ver seccion 3.2 del articulo.")
    return (rT / r0) ** 3


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bloque", type=int, default=0)
    ap.add_argument("--adam", type=int, default=2500)
    args = ap.parse_args()

    p = Difusion()
    cfg = Cfg(n_adam=args.adam, lam_ic=10.0, lam_bc=10.0, seed=0)

    print("=" * 68)
    print("CASO 1:  u_t = D(t) u_xx      (c = -1, omega = 0)")
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
