# Embedding de Ermakov en una PINN — paquete del manual

Material de apoyo del curso. Complemento práctico del manual
`manual_embedding.pdf`.

## Qué hay aquí

| archivo | qué es |
|---|---|
| `comun.py` | Maquinaria compartida: tronco `MLP`, interpolante `HermiteC1`, derivadas por autograd, bucle de entrenamiento, métricas. Nada específico de ningún caso. |
| `caso1_difusion.py` | Caso 1: `u_t = D(t) u_xx`. `c = -1`, `omega = 0`. |
| `caso2_trampa.py` | Caso 2: `u_t = D(t) u_xx + gamma(t) (x u)_x`. `c = -1`, `omega != 0`. |
| `caso3_cuantico.py` | Caso 3: `i psi_t = -1/2 psi_xx + 1/2 omega^2(t) x^2 psi`. `c = +1`. |
| `generar_salidas.py` | Corre los tres casos y guarda la salida de cada bloque en `../salidas/`. Es lo que produjo las cajas de salida del manual. |

## Requisitos

```
pip install torch numpy scipy
```

Todo corre en CPU. No hace falta GPU.

## Cómo usarlo

Cada caso está partido en bloques numerados y cada bloque corre solo. La idea
es ejecutar el bloque, mirar la salida, y **sólo entonces** leer el apartado
correspondiente del manual.

```bash
python caso1_difusion.py --bloque 1    # el problema y la identidad de momentos
python caso1_difusion.py --bloque 2    # la reducción a Ornstein-Uhlenbeck
python caso1_difusion.py --bloque 3    # inspeccionar la función de pérdida
python caso1_difusion.py --bloque 4    # entrenar la escalera de ablación
python caso1_difusion.py --bloque 5    # el peso implícito rho^-3

python caso2_trampa.py --bloque 1      # integrar la EDO y verificar el interpolante
python caso3_cuantico.py --bloque 2    # la referencia y el control adiabático

python caso1_difusion.py               # todos los bloques
python caso2_trampa.py --adam 6000     # más presupuesto de entrenamiento
```

El bloque 4 de cada caso es el que entrena, y es el único que tarda
(unos minutos por caso con el presupuesto por omisión).

## Tres cosas que conviene saber antes de tocar nada

1. **`float64`, no `float32`.** Lo fija `comun.py` en las primeras líneas. En
   precisión simple los cuatro peldaños de la escalera dan lo mismo y uno
   concluye, equivocadamente, que el embedding no sirve.

2. **La verificación no negociable.** En el caso 2, bloque 1(c): la derivada
   que `autograd` obtiene al atravesar el interpolante tiene que coincidir con
   la derivada analítica. Si no coincide, el grafo está roto y la red entrena
   contra un residuo equivocado **sin dar ningún mensaje de error**.

3. **Sin el peldaño de control no hay resultado.** Un número del tipo "EB-Net
   es N veces mejor que un PINN" no significa nada por sí solo. Hay que mirar
   *dónde* está el salto en la escalera.

## Sobre los números

El presupuesto por omisión (2500 iteraciones de Adam, una semilla) está
elegido para que esto corra en minutos en una laptop. **No** reproduce los
números del artículo, que usa un presupuesto mayor y promedia sobre varias
semillas. Sirve para ver el *orden* de la escalera, que es lo que el manual
quiere enseñar a leer.
