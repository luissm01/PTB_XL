# Autonomous development log

This document records the conceptual evolution of the project while the owner
is away. Stable constraints remain in `docs/context/DECISIONS.md`, current work
remains in `docs/context/STATUS.md` and detailed acceptance contracts remain in
`docs/context/missions/`.

## Mission 008 — Living project guide

### Qué se hizo

Se creó `docs/PROJECT_GUIDE.md`, una guía en español de diez partes que conecta
los fundamentos de ECG y PTB-XL con las siete misiones implementadas y con el
camino de modelado aún pendiente. También se enlazó desde el README y se corrigió
el estado del preprocessing.

### Por qué se hizo

El repositorio ya contenía contratos, decisiones e informes precisos, pero la
explicación estaba distribuida entre archivos orientados al desarrollo. Faltaba
una narrativa accesible para estudiar el sistema, comprender por qué se tomaron
las decisiones y defenderlas en una entrevista sin confundir planes con hechos.

### Decisiones documentales

- La guía distingue `implementado y verificado`, `planificado` y `pendiente de
  resultados`.
- Las cifras del proyecto proceden de los informes JSON versionados.
- Las afirmaciones generales importantes enlazan fuentes primarias u oficiales.
- PyTorch, CNN, entrenamiento, métricas y thresholds se explican como diseño
  futuro y no como funcionalidad existente.
- La guía se actualizará por secciones después de misiones materiales, sin
  sustituir a `STATUS.md`, `DECISIONS.md` ni los contratos de misión.

### Riesgos revisados

- No se inventaron métricas ni resultados de modelo.
- No se utilizó test para añadir o justificar decisiones de modelado.
- Las auditorías estructurales de test se diferenciaron expresamente de la
  selección con test.
- Se revisaron counts, shapes, leads y parámetros del standardizer contra la
  evidencia versionada, no contra memoria o cifras de otra versión de PTB-XL.
- Las categorías diagnósticas se presentan como etiquetas del dataset y el
  documento advierte que el proyecto no es de uso clínico.

### Contenido principal

La guía cubre problema multilabel, dataset y cohortes, cuatro tipos de leakage,
forma y amplitud de las señales, estandarización train-only, teoría de Conv1D y
`BCEWithLogitsLoss`, métricas multilabel, arquitectura de módulos, MLOps,
resultados de integridad, limitaciones y 16 preguntas de entrevista explicadas.

### Validación local

- Todos los enlaces locales existen y las nueve fuentes externas responden.
- `git diff --check` pasó.
- Los 86 tests pasaron.
- Ruff lint y format check pasaron.
- El sdist y el wheel se construyeron correctamente.
- No se añadieron dependencias ni comportamiento de ejecución.

### Qué debería entender el propietario

`PROJECT_GUIDE.md` es un documento vivo: debe crecer con el sistema, pero solo
una vez que cada etapa esté realmente implementada y verificada. Las secciones
de modelo y evaluación sirven ahora para estudiar y fijar el próximo contrato;
los resultados seguirán vacíos hasta que exista un protocolo válido para
producirlos.

### Preguntas de entrevista relacionadas

- ¿Cómo distingues documentación de diseño de documentación de comportamiento?
- ¿Qué evidencia permite afirmar que un pipeline es reproducible?
- ¿Por qué una auditoría estructural de test no equivale a seleccionar con test?
- ¿Cómo mantienes una guía viva sin duplicar el status operativo del proyecto?

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

La validación local quedó completa y el trabajo se asoció al issue `#16` y a la
pull request `#17`. Python quality y GitGuardian pasaron en la PR, que se fusionó
mediante squash como `d494f54`; el workflow Quality posterior también pasó en
`main`. El issue quedó cerrado y las ramas de implementación se eliminaron.

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

## Mission 007 — Train-only global signal standardization

### Qué se hizo

Se implementó un preprocessor agnóstico de framework que estima una única media
y desviación estándar globales recorriendo secuencialmente solo las señales de
train. Los parámetros se congelan, se guardan en JSON determinista y se pueden
cargar para transformar cualquier señal compatible sin refit.

### Por qué se hizo

Una CNN suele entrenar de forma más estable con entradas de escala controlada,
pero una normalización mal diseñada puede introducir leakage o borrar amplitud
relevante. La regla elegida usa un único cambio afín global, de modo que conserva
la relación de amplitudes entre ECG y derivaciones y permite demostrar de forma
explícita el límite `fit(train) → transform(all)`.

### Decisiones técnicas

- Una media y una desviación poblacional (`ddof=0`) para todos los valores.
- Fit exclusivo sobre folds 1–8 de la cohorte ya validada.
- Acumulación streaming en `float64` mediante combinación de momentos.
- Transformación final a `float32`, conservando forma `(1000, 12)`.
- JSON con versión, método, configuración, leads, conteos y hashes de fuentes.
- Ninguna dependencia nueva; no se usa scikit-learn ni pickle.

### Alternativas consideradas

No normalizar era la opción más pequeña, pero dejaba implícito el contrato
numérico. Normalizar cada registro habría eliminado diferencias de amplitud
entre pacientes. Normalizar por derivación habría cambiado la relación entre
leads. Materializar todos los ECG para usar `StandardScaler` habría necesitado
memoria innecesaria. El acumulador streaming reproduce la regla estadística del
benchmark público de PTB-XL con una implementación más auditable.

### Riesgos de leakage revisados

- El índice completo se valida antes de seleccionar train.
- La función de fit rechaza explícitamente muestras validation o test.
- El iterador real abrió únicamente las 17.084 filas con split `train`.
- El mismo artefacto congelado deberá reutilizarse en todos los splits e
  inferencia.
- No se calcularon estadísticas de validation ni test.

### Archivos principales

- `src/ptbxl/preprocessing/standardization.py`
- `scripts/fit_global_standardizer.py`
- `tests/preprocessing/test_standardization.py`
- `reports/preprocessing/ptbxl_v1.0.3_train_global_standardizer.json`
- `docs/context/missions/007_fit_train_only_global_standardizer.md`

### Tests añadidos

Los tests verifican la equivalencia con NumPy, combinación streaming, rechazo
de splits no train, input vacío, varianza cero, no finitos, orden de leads,
forma y dtype de salida, no mutación, procedencia, esquema, conteos y bytes JSON
deterministas.

### Resultados obtenidos

Los dos fits completos procesaron exactamente 17.084 ECG y 205.008.000 valores.
Ambos produjeron media `-0.0008252533901116082`, desviación poblacional
`0.23222258117564917` y el mismo SHA-256 de artefacto:
`f791aeb9795c669a54a391d979f69806ccdca19f05128a9fde8f408ec36090bc`.
La suite local pasó con 86 tests, Ruff y build de paquete. Python quality y
GitGuardian pasaron en la PR `#20`, fusionada mediante squash como `12b945a`;
el workflow Quality posterior también pasó en `main`, el issue `#19` se cerró y
las ramas de implementación se eliminaron.

### Qué debería entender el propietario

El objeto ajustado no aprende targets ni observa validation/test. Su estado son
solo dos escalares, conteos y orden de derivaciones. La separación entre fit y
transform es una barrera metodológica, no solo una elección de API.

### Preguntas de entrevista relacionadas

- ¿Por qué cualquier estadística de normalización debe ajustarse solo con train?
- ¿Qué información perdería una normalización por registro?
- ¿Por qué usar `ddof=0` y cómo se relaciona con `StandardScaler`?
- ¿Cómo se combinan media y segundo momento sin cargar todo el dataset?
- ¿Por qué guardar JSON en lugar de pickle?
