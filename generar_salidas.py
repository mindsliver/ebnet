r"""
generar_salidas.py -- corre los tres casos bloque por bloque y guarda la
salida literal de cada bloque en ../salidas/cN_bM.txt, que es lo que el
manual incluye con \\VerbatimInput.

    python generar_salidas.py [--adam 2500]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time

SAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "salidas")


def captura(nombre, fn, *a, **k):
    """Ejecuta fn capturando stdout y lo guarda en salidas/<nombre>.txt."""
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        res = fn(*a, **k)
    txt = buf.getvalue()
    # recortar lineas larguisimas para que quepan en la caja del manual
    txt = "\n".join(l[:104] for l in txt.splitlines()) + "\n"
    os.makedirs(SAL, exist_ok=True)
    with open(os.path.join(SAL, nombre + ".txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"  [{nombre}] {time.time() - t0:6.1f}s  ->  salidas/{nombre}.txt",
          flush=True)
    return txt, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adam", type=int, default=2500)
    args = ap.parse_args()

    from comun import Cfg
    cfg = Cfg(n_adam=args.adam, lam_ic=10.0, lam_bc=10.0, seed=0)

    # ---------------- CASO 1 ----------------
    print("CASO 1", flush=True)
    import caso1_difusion as c1
    p1 = c1.Difusion()
    _ = captura("c1_b1", c1.bloque1, p1)
    captura("c1_b2", c1.bloque2, p1)
    captura("c1_b3", c1.bloque3, p1, cfg)
    captura("c1_b4", c1.bloque4, p1, cfg)
    captura("c1_b5", c1.bloque5, p1)

    # ---------------- CASO 2 ----------------
    print("CASO 2", flush=True)
    import caso2_trampa as c2
    p2 = c2.Trampa()
    captura("c2_b1", c2.bloque1, p2)
    captura("c2_b2", c2.bloque2, p2)
    captura("c2_b3", c2.bloque3, p2, cfg)
    captura("c2_b4", c2.bloque4, p2, cfg)
    captura("c2_b5", c2.bloque5, p2)

    # ---------------- CASO 3 ----------------
    print("CASO 3", flush=True)
    import caso3_cuantico as c3
    p3 = c3.Cuantico()
    captura("c3_b1", c3.bloque1, p3)
    _, (_, err_ad) = captura("c3_b2", c3.bloque2, p3)
    captura("c3_b3", c3.bloque3, p3, cfg)
    captura("c3_b4", c3.bloque4, p3, cfg, err_ad)

    print("LISTO", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
