# Autonomous development log

This document records the conceptual evolution of the project while the owner
is away. Stable constraints remain in `docs/context/DECISIONS.md`, current work
remains in `docs/context/STATUS.md` and detailed acceptance contracts remain in
`docs/context/missions/`.

## Mission 006 — Framework-independent sample contract

### Qué se hizo

Se añadió una frontera que compone una fila de la cohorte con su basename WFDB
oficial y permite cargar bajo demanda una muestra completa con identidad,
paciente, fold, split, señal y cinco targets.

### Por qué se hizo

Labels, cohorte y señales ya estaban validados por separado. Faltaba un único
punto reutilizable que demostrase que el modelo futuro recibirá la señal y los
targets del mismo `ecg_id` sin volver a implementar joins dentro de PyTorch.

### Decisiones técnicas

- El índice de muestras es transitorio y conserva el orden de la cohorte.
- La asociación con `filename_lr` es uno-a-uno mediante `ecg_id`.
- La carga es lazy: construir el índice no abre señales.
- La señal permanece en NumPy y con forma `(1000, 12)`.
- Los targets usan `float32`, forma `(5,)` y orden
  `NORM, MI, STTC, CD, HYP`.
- La conversión a `(channels, samples)` se reserva para la futura frontera con
  PyTorch.

### Alternativas consideradas

Persistir una nueva tabla con rutas habría duplicado un dato derivable y creado
otra fuente de verdad. Introducir ya un `torch.utils.data.Dataset` habría
mezclado el contrato básico con decisiones de framework y preprocessing. Se
eligió una composición pequeña con pandas y NumPy, dependencias ya presentes.

### Riesgos de leakage revisados

- Se valida la cohorte completa antes de que un consumidor seleccione un split.
- Se vuelve a comprobar la separación por paciente y la relación oficial entre
  `strat_fold` y `split`.
- No se calculan estadísticas de amplitud ni prevalencia.
- Validation y test solo participaron en smoke checks estructurales fijados por
  el contrato; no se tomó ninguna decisión usando su contenido.

### Archivos principales

- `src/ptbxl/data/samples.py`
- `src/ptbxl/data/signals.py`
- `tests/data/test_samples.py`
- `docs/context/missions/006_build_framework_independent_sample_contract.md`

### Tests añadidos

Los tests sintéticos cubren orden e identidad, carga lazy, señal y target
correctos, orden semántico de labels, filas fuera de cohorte, targets inválidos,
incoherencia fold/split, patient leakage y asociaciones ausentes o ambiguas.

### Resultados obtenidos

El índice real contiene las 21.388 filas esperadas y conserva su orden. Un smoke
check de un ECG de train, validation y test devolvió en los tres casos señal
`(1000, 12)`, targets `(5,)`, 100 Hz y las 12 derivaciones oficiales. Estos son
hechos estructurales, no resultados de un modelo.

La validación local está completa y el trabajo quedó asociado al issue `#16` y
a la pull request `#17`. Sus checks remotos aún estaban pendientes al redactar
esta entrada.

El sdist y el wheel también se construyeron correctamente, y la inspección del
wheel confirmó que `ptbxl/data/samples.py` forma parte del paquete distribuible.
Los artefactos de build permanecen ignorados por Git.

### Qué debería entender el propietario

`build_sample_index` demuestra la asociación entre fuentes sin cargar los ECG.
`load_sample` abre exactamente una señal y devuelve un contrato agnóstico de
framework. Este límite es el que la futura capa de preprocessing y el Dataset de
PyTorch deben reutilizar.

### Preguntas de entrevista relacionadas

- ¿Por qué conviene validar el join señal-target antes de introducir PyTorch?
- ¿Por qué el índice es transitorio en vez de persistirse?
- ¿Por qué se conserva `(samples, leads)` hasta la frontera del framework?
- ¿Cómo evita este diseño que cada Dataset vuelva a implementar labels y
  splits?
