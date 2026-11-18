"""
verificacion_simbolica.py -- BLOQUE 0 de los tres casos.
=========================================================

Antes de escribir una sola linea de red, conviene comprobar con algebra
simbolica que las identidades sobre las que se apoya todo el pipeline son
correctas. Son tres, una por caso, y las tres se verifican en segundos.

No necesita torch: solo sympy. Es lo primero que deberia correr un alumno.

    python verificacion_simbolica.py
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
I = sp.I


def banner(n, txt):
    print("\n" + "=" * 66)
    print(f"IDENTIDAD {n}: {txt}")
    print("=" * 66)


# =====================================================================
def caso1():
    r"""CASO 1. Con rho rho' = D y theta' = D/rho^2, el cambio
        u = rho^{-1} v(x/rho, theta)
    convierte el residuo fisico en el reducido segun la identidad EXACTA

        u_t - D u_xx = (D/rho^3) [ v_theta - d_xi(v_xi + xi v) ].

    Esta es la ecuacion (eq:resid) del manual, y es la que explica el peso
    implicito rho^{-3}.
    """
    banner(1, "reduccion de la difusion libre  (caso 1, c = -1, omega = 0)")
    rho = sp.Function("rho", positive=True)(t)
    D = sp.Function("D", positive=True)(t)
    th = sp.Function("theta", real=True)(t)
    v = sp.Function("v")

    xi = x / rho
    u = v(xi, th) / rho
    res_fis = sp.diff(u, t) - D * sp.diff(u, x, 2)
    # relaciones que definen el embedding
    res_fis = res_fis.subs(sp.Derivative(rho, t), D / rho).doit()
    res_fis = res_fis.subs(sp.Derivative(th, t), D / rho**2).doit()

    X, TH = sp.symbols("xi theta")
    vv = v(X, TH)
    reducido = sp.diff(vv, TH) - sp.diff(sp.diff(vv, X) + X * vv, X)
    esperado = (D / rho**3) * reducido.subs({X: xi, TH: th}).doit()

    dif = sp.simplify(sp.expand(res_fis - esperado))
    print("  u_t - D u_xx  -  (D/rho^3)[v_theta - d_xi(v_xi + xi v)]  =", dif)
    assert dif == 0, dif
    print("  OK. El operador reducido es el de Ornstein-Uhlenbeck, autonomo,")
    print("  y el prefactor D/rho^3 es la fuente del peso implicito.")


# =====================================================================
def caso2():
    r"""CASO 2. Con rho rho' = D - gamma rho^2 y theta' = D/rho^2, el MISMO
    cambio u = rho^{-1} v(x/rho, theta) reduce la ecuacion de la trampa al
    MISMO operador de Ornstein-Uhlenbeck.

    Es el punto que justifica decir que hay UNA construccion y no dos.
    """
    banner(2, "reduccion de la trampa  (caso 2, c = -1, omega != 0)")
    rho = sp.Function("rho", positive=True)(t)
    D = sp.Function("D", positive=True)(t)
    g = sp.Function("gamma", positive=True)(t)
    th = sp.Function("theta", real=True)(t)
    v = sp.Function("v")

    xi = x / rho
    u = v(xi, th) / rho
    res_fis = (sp.diff(u, t) - D * sp.diff(u, x, 2)
               - g * sp.diff(x * u, x))
    res_fis = res_fis.subs(sp.Derivative(rho, t), (D - g * rho**2) / rho).doit()
    res_fis = res_fis.subs(sp.Derivative(th, t), D / rho**2).doit()

    X, TH = sp.symbols("xi theta")
    vv = v(X, TH)
    reducido = sp.diff(vv, TH) - sp.diff(sp.diff(vv, X) + X * vv, X)
    esperado = (D / rho**3) * reducido.subs({X: xi, TH: th}).doit()

    dif = sp.simplify(sp.expand(res_fis - esperado))
    print("  (trampa) residuo fisico - (D/rho^3)[operador OU]  =", dif)
    assert dif == 0, dif
    print("  OK. Mismo operador reducido que el caso 1: la trampa entra por")
    print("  omega^2 = gamma' - gamma^2 en Pinney, no por el operador latente.")


# =====================================================================
def caso2_pinney():
    r"""CASO 2, segunda identidad: derivando rho rho' = D - gamma rho^2 se
    obtiene la forma de Pinney

        rho'' + (gamma' - gamma^2) rho - D'/rho = -D^2/rho^3 .
    """
    banner("2b", "forma de Pinney de la trampa")
    rho = sp.Function("rho", positive=True)(t)
    D = sp.Function("D", positive=True)(t)
    g = sp.Function("gamma", positive=True)(t)

    rel = sp.Eq(rho * sp.diff(rho, t), D - g * rho**2)
    d2 = sp.solve(sp.diff(rel.lhs - rel.rhs, t), sp.Derivative(rho, (t, 2)))[0]
    d2 = sp.simplify(d2.subs(sp.Derivative(rho, t), (D - g * rho**2) / rho))

    izq = d2 + (sp.diff(g, t) - g**2) * rho - sp.diff(D, t) / rho
    dif = sp.simplify(sp.expand(izq + D**2 / rho**3))
    print("  rho'' + (gamma'-gamma^2) rho - D'/rho + D^2/rho^3  =", dif)
    assert dif == 0, dif
    print("  OK. omega^2 = gamma' - gamma^2, y la constante de Pinney es negativa.")


# =====================================================================
def caso3():
    r"""CASO 3. El ansatz de Lewis-Riesenfeld para el estado base,

        psi = pi^{-1/4} rho^{-1/2} exp( i rho' x^2/(2 rho) - i theta/2
                                        - x^2/(2 rho^2) ),

    resuelve  i psi_t = -1/2 psi_xx + 1/2 omega^2 x^2 psi  EXACTAMENTE,
    dadas  rho'' + omega^2 rho = 1/rho^3  (Pinney con c = +1)  y
    theta' = 1/rho^2.
    """
    banner(3, "solucion exacta del oscilador cuantico  (caso 3, c = +1)")
    rho = sp.Function("rho", positive=True)(t)
    th = sp.Function("theta", real=True)(t)
    w = sp.Function("omega", positive=True)(t)

    psi = (sp.pi**sp.Rational(-1, 4) * rho**sp.Rational(-1, 2)
           * sp.exp(I * sp.diff(rho, t) * x**2 / (2 * rho)
                    - I * th / 2 - x**2 / (2 * rho**2)))
    res = (I * sp.diff(psi, t) + sp.Rational(1, 2) * sp.diff(psi, x, 2)
           - sp.Rational(1, 2) * w**2 * x**2 * psi)
    res = res.subs(sp.Derivative(th, t), 1 / rho**2).doit()
    res = res.subs(sp.Derivative(rho, (t, 2)), 1 / rho**3 - w**2 * rho).doit()
    dif = sp.simplify(sp.expand(sp.simplify(res / psi)))
    print("  [i psi_t + 1/2 psi_xx - 1/2 omega^2 x^2 psi] / psi  =", dif)
    assert dif == 0, dif
    print("  OK. La fase cuadratica e^{i rho' x^2/2rho} es imprescindible:")
    print("  sin ella la identidad NO se cumple (vease el ejercicio del manual).")


# =====================================================================
def caso3_sin_fase():
    """Contraste: si se quita la fase, la identidad falla. Esto es lo que
    mide el peldano QEB-sinfase de la escalera del caso 3."""
    banner("3b", "que pasa si se quita la fase cuadratica")
    rho = sp.Function("rho", positive=True)(t)
    th = sp.Function("theta", real=True)(t)
    w = sp.Function("omega", positive=True)(t)

    psi = (sp.pi**sp.Rational(-1, 4) * rho**sp.Rational(-1, 2)
           * sp.exp(-I * th / 2 - x**2 / (2 * rho**2)))        # SIN e^{iS}
    res = (I * sp.diff(psi, t) + sp.Rational(1, 2) * sp.diff(psi, x, 2)
           - sp.Rational(1, 2) * w**2 * x**2 * psi)
    res = res.subs(sp.Derivative(th, t), 1 / rho**2).doit()
    res = res.subs(sp.Derivative(rho, (t, 2)), 1 / rho**3 - w**2 * rho).doit()
    dif = sp.simplify(sp.expand(sp.simplify(res / psi)))
    print("  residuo/psi sin la fase =", sp.simplify(dif))
    print("  No es cero. El termino que sobra es justo el que la fase cancela.")


if __name__ == "__main__":
    caso1()
    caso2()
    caso2_pinney()
    caso3()
    caso3_sin_fase()
    print("\n" + "=" * 66)
    print("LAS CUATRO IDENTIDADES DEL MANUAL QUEDAN VERIFICADAS.")
    print("=" * 66)
