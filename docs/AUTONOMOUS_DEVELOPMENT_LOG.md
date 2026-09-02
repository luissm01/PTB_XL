# Autonomous development log

This document records the conceptual evolution of the project while the owner
is away. Stable constraints remain in `docs/context/DECISIONS.md`, current work
remains in `docs/context/STATUS.md` and detailed acceptance contracts remain in
`docs/context/missions/`.

## Mission 012 — Validation-selected fit and safe checkpoints

### Qué se hizo

Se añadió `fit` multi-época, selección por mínima loss de validation y un
checkpoint atómico que restaura modelo y optimizer. Guarda historial completo y
procedencia; carga únicamente en modo `weights_only`.

### Decisiones y alternativas

El primer mínimo gana los empates. Se rechazaron early stopping y scheduler para
mantener aislada la semántica básica. Un JSON no puede contener tensores; se usa
el formato PyTorch seguro con estructuras simples y validación explícita.

### Riesgos de leakage revisados

Los loaders deben declarar `train` y `validation`; `test` falla antes del fit.
Solo se usaron datos sintéticos y no existen métricas o thresholds.

### Archivos y tests

- `src/ptbxl/training/fit.py`
- `tests/training/test_fit.py`
- `docs/context/missions/012_add_validation_fit_checkpoints.md`

Nueve tests cubren fit completo, historial, selección en empates, restauración,
round-trip exacto, roles de split, configuración, schema y procedencia.
La suite local completa pasó con 136 tests, Ruff y build del paquete.
El issue `#34` quedó cerrado; la PR `#35` se fusionó como `1bf5443` después de
pasar Python quality y GitGuardian. Quality volvió a pasar después del merge en
`main` y se eliminaron las ramas de implementación.

### Qué debería entender el propietario

El checkpoint no es solo pesos: identifica con qué dataset, cohorte,
preprocessing, modelo, seed y commit se produjo. Validation elige; test no puede
entrar en esta API.

### Preguntas de entrevista relacionadas

- ¿Por qué guardar estado del optimizer además del modelo?
- ¿Por qué seleccionar el primer mínimo en un empate?
- ¿Qué riesgos reducen escritura atómica y `weights_only=True`?

## Mission 011 — Reproducible epoch train/evaluate engine

### Qué se hizo

Se añadieron seeds unificadas, resolución explícita CPU/CUDA y dos fronteras:
`train_one_epoch` actualiza parámetros y `evaluate_loss` evalúa sin gradientes.
Ambas devuelven loss media ponderada por muestras y conteos.

### Por qué y decisiones técnicas

Separar train de evaluación evita optimizer steps accidentales. La ponderación
por muestras trata correctamente el batch final corto. Se validan mappings
`signal`/`targets`, shapes, loader no vacío y loss finita.

### Riesgos de leakage revisados

Solo se usaron datos sintéticos. No existen selección de checkpoint, métricas,
thresholds ni acceso a PTB-XL para entrenar.

### Archivos y tests

- `src/ptbxl/training/engine.py`
- `src/ptbxl/training/reproducibility.py`
- `tests/training/test_engine.py`

Nueve tests verifican streams reproducibles, devices, actualización exclusiva
en train, ausencia de cambios en evaluación, ponderación y errores explícitos.
La suite completa pasó con 127 tests, Ruff y build.

Issue `#31` y PR `#32` quedaron cerrados; la PR se fusionó como `79a64f8` con
Python quality, GitGuardian y Quality posterior al merge en verde.

### Qué debería entender el propietario

Una seed controla fuentes conocidas de azar, pero no promete identidad absoluta
entre cualquier GPU. Evaluación cambia a modo eval y no crea gradientes; train
es la única frontera que hace optimizer steps.

### Preguntas de entrevista relacionadas

- ¿Por qué promediar loss por muestras y no por batches?
- ¿Qué cambia entre `model.train()` y `model.eval()`?
- ¿Qué garantiza una seed y qué no garantiza en GPU?

## Mission 010 — Small 1D-CNN baseline contract

### Qué se hizo

Se implementó `SmallECGCNN`, el primer modelo del proyecto. Recibe batches
`float32` `(B, 12, 1000)`, extrae características con tres bloques Conv1D y
devuelve cinco logits crudos. `ECGCNNConfig` conserva de forma inmutable y
validada los canales, kernels y dropout de la arquitectura.

### Por qué se hizo

Dataset y batching ya tenían un contrato estable. Antes de construir un bucle de
entrenamiento era necesario demostrar por separado que un modelo pequeño acepta
exactamente ese contrato, representa correctamente una tarea multilabel y
propaga gradientes finitos.

### Decisiones técnicas

- Tres bloques `Conv1D -> BatchNorm1D -> ReLU -> MaxPool1D`.
- Canales por defecto `32, 64, 128` y kernels impares `7, 5, 3`.
- Padding que conserva longitud antes de cada pooling factor dos.
- Adaptive global average pooling, dropout `0.2` y Linear a cinco logits.
- Entrada exacta `float32 (B, 12, 1000)` con errores previos a las convoluciones.
- Ninguna sigmoid/softmax en el modelo; BCE recibe logits directamente.
- Configuración por defecto con 38.597 parámetros entrenables.

### Alternativas consideradas

Un transformer, ResNet grande o ensemble habría ocultado errores básicos y
añadido coste antes de disponer de training. Una MLP ignoraría la estructura
temporal. También era posible omitir BatchNorm o pooling, pero el bloque clásico
elegido ofrece un baseline compacto, reconocible y fácil de explicar.

### Riesgos de leakage revisados

- La arquitectura se eligió por simplicidad de ingeniería, sin observar
  resultados de validation ni test.
- Los tests usan exclusivamente tensores sintéticos.
- No se añadieron prevalencias, class weights, sampler, threshold ni métricas.
- El modelo no conoce splits y no abre datos.

### Archivos principales

- `src/ptbxl/models/cnn.py`
- `src/ptbxl/models/__init__.py`
- `tests/models/test_cnn.py`
- `docs/context/missions/010_implement_small_1d_cnn.md`

### Tests añadidos

Veintidós casos cubren configuración congelada, valores inválidos, varios
tamaños de batch, rango completo de errores de input, logits sin activación,
parameter count, loss BCE, gradientes de todos los parámetros y determinismo en
modo evaluación.

### Resultados obtenidos

El forward produce `(B, 5)`, la loss sintética es finita y todos los parámetros
entrenables reciben gradientes finitos. Estos son resultados de contrato de
software; el modelo no está entrenado y no existe resultado predictivo. La suite
completa pasó con 118 tests, Ruff y build, y el wheel contiene el paquete
`ptbxl.models`.

La implementación se asoció al issue `#28` y a la pull request `#29`. Python
quality y GitGuardian pasaron en la PR, fusionada mediante squash como
`a228b42`; el workflow Quality posterior también pasó en `main`. El issue quedó
cerrado y las ramas de implementación se eliminaron.

### Qué debería entender el propietario

Los logits son números sin limitar y no probabilidades. Durante entrenamiento
`BCEWithLogitsLoss` combina de forma estable cada logit con su decisión binaria.
Sigmoid solo será necesaria cuando evaluación o inferencia conviertan logits a
probabilidades.

### Preguntas de entrevista relacionadas

- ¿Por qué Conv1D usa 12 canales y recorre el eje de 1.000 muestras?
- ¿Por qué la cabeza devuelve cinco logits sin sigmoid?
- ¿Qué aporta el global average pooling y qué información puede perder?
- ¿Por qué fijar y probar el número de parámetros del baseline?
- ¿Qué significa un campo receptivo local de 0,3 segundos en esta arquitectura?

## Mission 009 — Thin PyTorch Dataset/DataLoader boundary

### Qué se hizo

Se añadió la primera frontera PyTorch del proyecto. `PTBXLDataset` recibe un
índice de muestras ya validado y de un único split, carga cada ECG bajo demanda,
aplica el standardizer global congelado y produce señal `float32` con forma
`(12, 1000)`, target `float32` `(5,)` y procedencia. Un constructor pequeño de
DataLoader hace explícita la seed cuando se activa shuffle.

### Por qué se hizo

La CNN futura necesita batches channel-first, pero las reglas de identidad,
labels, paths, splits y preprocessing ya estaban correctamente resueltas fuera
del framework. Una capa fina permite aprovechar PyTorch sin crear una segunda
implementación divergente del dataset.

### Decisiones técnicas

- PyTorch 2.13.0 se añadió con `uv add`; el lockfile fue generado por uv.
- Se mantiene el paquete estándar compatible con CPU/CUDA porque la RTX 3060 Ti
  local está disponible para futuros entrenamientos.
- Un Dataset contiene exactamente un split declarado y rechaza índices vacíos,
  mezclados o con columnas ausentes.
- `load_sample` sigue siendo la única composición señal-target.
- `GlobalStandardizer.transform` se reutiliza sin refit para cualquier split.
- La transposición se materializa como array contiguo antes de `torch.from_numpy`.
- Se usa la collation estándar; no se añadió un batch o sampler personalizado.

### Alternativas consideradas

Mover los joins y el preprocessing a `__getitem__` habría duplicado reglas ya
probadas. Persistir tensores preprocesados habría aumentado almacenamiento y
creado otra fuente derivada. Un DataModule o framework de entrenamiento completo
habría sido prematuro. También se consideró un entorno CPU-only más ligero, pero
se mantuvo soporte CUDA real sin añadir variantes de entorno todavía.

### Riesgos de leakage revisados

- El índice completo debe construirse y validarse antes de seleccionar un split.
- Cada Dataset rechaza una mezcla de train, validation y test.
- Ni Dataset ni DataLoader conocen una operación de fit.
- El smoke check de validation/test solo verificó forma, dtype, continuidad e
  identidad ya fijados; no produjo estadísticas ni decisiones de modelado.
- No se añadieron sampling weights, class weights, augmentations ni métricas.

### Archivos principales

- `src/ptbxl/data/pytorch.py`
- `tests/data/test_pytorch.py`
- `docs/context/missions/009_add_thin_pytorch_dataset.md`
- `pyproject.toml`
- `uv.lock`

### Tests añadidos

Diez tests sintéticos cubren carga lazy, copia defensiva del índice, los tres
splits, valores/forma/dtype/continuidad, procedencia, columnas ausentes, índice
vacío, split inválido o mezclado, standardizer inválido, índices fuera de rango,
batching y shuffle reproducible.

### Resultados obtenidos

Un smoke check real cargó dos ECG de cada split con el mismo artefacto de train.
En los tres casos obtuvo señales `(2, 12, 1000)`, targets `(2, 5)`, dtype
`torch.float32` y memoria contigua. Son comprobaciones estructurales, no
resultados de modelo. La suite completa pasó con 96 tests, Ruff y build; el
wheel contiene el adaptador y declara PyTorch como dependencia.

La implementación se asoció al issue `#25` y a la pull request `#26`. Python
quality y GitGuardian pasaron en la PR, fusionada mediante squash como
`9e6ab7e`; el workflow Quality posterior también pasó en `main`. El issue quedó
cerrado y las ramas de implementación se eliminaron.

### Qué debería entender el propietario

El Dataset no decide qué es una muestra: adapta una muestra que las capas
anteriores ya saben construir y validar. La separación `(1000, 12)` en NumPy y
`(12, 1000)` en PyTorch mantiene el dominio de datos independiente y concentra
la convención del framework en un único lugar.

### Preguntas de entrevista relacionadas

- ¿Por qué el Dataset exige un único split aunque cada fila ya incluya `split`?
- ¿Por qué el preprocessing no debe ajustarse dentro de `__getitem__`?
- ¿Por qué se fuerza memoria contigua después de transponer?
- ¿Qué parte de la reproducibilidad aporta el generator del DataLoader?

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

### Resultados obtenidos

La implementación se asoció al issue `#22` y a la pull request `#23`. Python
quality y GitGuardian pasaron en la PR, fusionada mediante squash como
`395c74b`; el workflow Quality posterior también pasó en `main`. El issue quedó
cerrado y las ramas de implementación se eliminaron.

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
