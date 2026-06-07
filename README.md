# VInculacion

Prototipo web para tracking de manos y rostro orientado a una futura traduccion de lengua de senias ecuatoriana.

## Que hace

- Usa la camara del navegador en tiempo real.
- Detecta hasta dos manos con MediaPipe Gesture Recognizer.
- Detecta rostro y expresiones basicas con MediaPipe Face Landmarker.
- Traduce gestos base como mano abierta, puno cerrado, pulgar arriba e indice arriba.
- Permite registrar senias ecuatorianas personalizadas desde la interfaz.
- Tiene modo local para guardar muestras solo en el navegador de cada persona.
- Tiene modo administrador para guardar muestras globales en `data/global-data.json`.
- Permite capturar muestras estaticas y muestras con movimiento.
- Incluye modo abecedario LSEC con A, B, C, CH, D, E, F, G, H, I, J, K, L, LL, LLL, M, N, NN, O, P, Q, R, RR, S, T, U, V, W, X, Y, Z.

## Como correrlo en localhost

Desde esta carpeta:

```powershell
npm install
npm start
```

Luego abre:

```text
http://localhost:5173
```

## Modo administrador

El boton de inicio de sesion administrador esta arriba a la derecha.

Sin iniciar sesion, las muestras se guardan localmente en el navegador. Con sesion de administrador, las nuevas muestras se guardan globalmente para todas las personas que usen ese servidor.

Para este prototipo local, las credenciales se configuran en `server.mjs`. Para publicarlo en internet conviene reemplazarlo por autenticacion segura con base de datos y HTTPS.

## Nota importante

La lengua de senias ecuatoriana no se puede traducir correctamente solo con reglas genericas. Para una version academica o de produccion se necesita crear un dataset con senias locales, grabar multiples personas, angulos e iluminaciones, etiquetar las muestras y entrenar un modelo propio.

Este prototipo deja lista la base tecnica: captura de landmarks, reconocimiento en tiempo real, muestras locales y un dataset global administrado desde el servidor.
