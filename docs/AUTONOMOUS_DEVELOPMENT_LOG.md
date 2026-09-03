# Autonomous development log

This document records the conceptual evolution of the project while the owner
is away. Stable constraints remain in `docs/context/DECISIONS.md`, current work
remains in `docs/context/STATUS.md` and detailed acceptance contracts remain in
`docs/context/missions/`.

## Mission 016 — One-time sealed final evaluation

### Qué se hizo

Se añadió un runner independiente que verifica por hashes exactos el baseline,
checkpoint, standardizer y thresholds congelados antes de construir un Dataset
exclusivo de test. Desde el commit limpio `056cdc4` se evaluaron una sola vez
los 2.158 ECG de fold 10 y se guardó el reporte agregado versionado.

### Por qué se hizo

Validation permitió construir y seleccionar el sistema, pero no proporciona una
estimación independiente de generalización. El evento sellado preserva esa
independencia y convierte fold 10 en evidencia final, no en otra fuente de
iteraciones.

### Decisiones técnicas y riesgos

El comando rechaza cualquier hash distinto y cualquier output preexistente
antes de leer señales. Solo acepta un loader declarado como `test`, restaura el
checkpoint sin optimizar y reutiliza las definiciones de ranking y punto
operativo. Las predicciones por ECG se guardan localmente como arrays NPZ sin
pickle; el reporte conserva tanto el hash del archivo como una huella canónica.
Tras observar el resultado quedan prohibidos tuning, calibración y selección
basados en test.

### Resultados obtenidos

Fold 10 produjo AUROC macro/micro `0,908895` / `0,922858`, AUPRC macro/micro
`0,785850` / `0,827715` y F1 macro/micro en thresholds congelados `0,725377` /
`0,756842`. `HYP` fue la clase más débil por AUPRC (`0,6249`) y F1 (`0,5932`),
una observación descriptiva que no autoriza cambiar el pipeline. El artefacto
local se recargó con 2.158 IDs y fingerprint idéntico. Antes del evento pasaron
187 tests, Ruff y build.

### Qué debería entender el propietario

El resultado final es interno a PTB-XL y no demuestra utilidad clínica ni
generalización externa. Su valor principal es que procede de un pipeline
completamente fijado y una sola observación del test. El siguiente desarrollo
puede explicar errores y facilitar inferencia, pero no mejorar este mismo
baseline usando fold 10.

### Preguntas de entrevista relacionadas

- ¿Por qué el test debe ejecutarse una sola vez?
- ¿Qué diferencia hay entre el hash NPZ y la huella canónica del contenido?
- ¿Por qué un resultado final interno no equivale a validación clínica?
- ¿Qué trabajo posterior puede hacerse sin convertir test en validation?

## Mission 015 — Frozen validation thresholds

### Qué se hizo

Se añadió una política reproducible que restaura el checkpoint del baseline,
recoge únicamente predicciones de validation, selecciona un threshold por clase
y guarda las decisiones junto con métricas, matrices de confusión y procedencia.

### Por qué se hizo

AUROC y AUPRC evalúan ranking pero no convierten probabilidades en decisiones.
La evaluación final y la inferencia necesitan un operating point fijo antes de
abrir fold 10.

### Decisiones técnicas y alternativas

Cada threshold maximiza F1 de su clase; un empate conserva el valor más alto y
la decisión usa `probability >= threshold`. Se descartó `0,5` global porque no
refleja las distintas distribuciones de scores. También se evitó una función de
coste clínica inventada: sin costes conocidos, F1 proporciona un equilibrio
explícito entre precision y sensibilidad.

### Riesgos de leakage revisados

La API de predicción sigue aceptando exclusivamente validation. El runner no
crea Dataset, loader, predicción ni métrica de test y no reentrena ni cambia el
checkpoint. Los resultados de punto operativo se etiquetan como optimistas
porque el mismo fold selecciona y mide los cutoffs.

### Archivos principales y tests

- `src/ptbxl/evaluation/thresholds.py`
- `src/ptbxl/experiments/thresholds.py`
- `configs/baseline_small_cnn_100hz_thresholds.toml`
- `scripts/select_validation_thresholds.py`
- `tests/evaluation/test_thresholds.py`
- `tests/experiments/test_threshold_selection.py`
- `reports/evaluation/baseline_small_cnn_100hz_thresholds.json`

Los tests cubren óptimo único, empates, conteos conocidos, cero predicciones,
inputs degenerados, fingerprint, schema estricto, coherencia algebraica,
vinculación del checkpoint, colisiones de output y un flujo WFDB sintético sin
reentrenar.

### Resultados obtenidos

En los 2.146 ECG de fold 9, los thresholds son `0,327765` para NORM, `0,511551`
para MI, `0,380263` para STTC, `0,387861` para CD y `0,145285` para HYP. El F1
macro es `0,737768` y el micro `0,767029`. El artefacto se volvió a cargar
exigiendo el SHA-256 exacto del checkpoint. La puerta local final pasó con 176
tests, Ruff lint, Ruff format y build del paquete. El issue `#41` y la PR única
`#42` contienen implementación, evidencia y cierre; GitHub conserva el estado
autoritativo de checks y merge.

### Qué debería entender el propietario

Un threshold no es una propiedad universal del diagnóstico: depende del modelo,
datos y objetivo. Por eso los cinco valores están vinculados a pesos y
preprocessing concretos. El F1 de validation sirve para fijarlos, pero solo test
podrá medir su rendimiento independiente.

### Preguntas de entrevista relacionadas

- ¿Por qué no usar `0,5` para todas las clases?
- ¿Por qué elegir un threshold por clase?
- ¿Qué optimismo introduce medir F1 en el mismo fold que selecciona el cutoff?
- ¿Cómo se evita mezclar thresholds y checkpoints incompatibles?

## Mission 014 — Reproducible real baseline

### Qué se hizo

Se conectaron datos, standardizer, CNN, entrenamiento, checkpoint y métricas en
un experimento configurado por TOML. La ejecución real entrenó con los 17.084
ECG de folds 1–8 y evaluó el checkpoint restaurado con los 2.146 ECG de fold 9.

### Por qué se hizo

Los componentes aislados ya estaban probados, pero faltaba demostrar que el
sistema completo podía producir un resultado real atribuible y repetible. Este
baseline neutral crea la referencia desde la que comparar mejoras posteriores.

### Decisiones técnicas y alternativas

Se fijaron diez épocas, seed 2026, batch 128, Adam con learning rate 0,001 y
`BCEWithLogitsLoss` sin pesos. La CNN conserva sus 38.597 parámetros originales.
Se eligió un TOML y un JSON determinista en vez de introducir MLflow para una
sola ejecución. El runner exige un commit limpio, algoritmos deterministas y
workers sembrados.

### Riesgos de leakage revisados

El comando construye únicamente Dataset y DataLoader de train y validation. El
standardizer sigue siendo el ajustado solo con train. No se calcularon
thresholds, no se cambió retrospectivamente la configuración y no se abrió
ninguna señal ni métrica de fold 10.

### Archivos principales y tests

- `configs/baseline_small_cnn_100hz.toml`
- `src/ptbxl/experiments/baseline.py`
- `scripts/run_baseline_experiment.py`
- `tests/experiments/test_baseline.py`
- `reports/experiments/baseline_small_cnn_100hz.json`
- `docs/context/missions/014_run_reproducible_real_baseline.md`

Los tests sintéticos cubren configuración estricta, procedencia Git, ejecución
end-to-end, checkpoint cargable, reporte y conteos. También se verificó el modo
determinista de PyTorch y se reutilizaron los contratos previos de entrenamiento
y evaluación.

### Resultados obtenidos

La época 9 fue el primer mínimo de loss de validation (`0,294930`). En fold 9,
el modelo obtuvo AUROC macro `0,915310`, AUPRC macro `0,785994`, AUROC micro
`0,926058` y AUPRC micro `0,832366`. Por clase, AUPRC fue `0,9137` en NORM,
`0,8039` en MI, `0,7620` en STTC, `0,8281` en CD y `0,6223` en HYP. El checkpoint
ignorado se volvió a cargar en CPU y coincidieron época, historial, loss,
procedencia y SHA-256. La puerta local final pasó con 158 tests, Ruff lint, Ruff
format y build del paquete. El issue `#39` y la PR única `#40` contienen la
implementación, evidencia y cierre; Python quality y GitGuardian pasaron.

### Qué debería entender el propietario

Ya existe una primera medida real del modelo, pero pertenece a validation. Sirve
para desarrollar y comparar; no debe presentarse como evaluación final ni como
evidencia clínica. HYP es la señal más débil por AUPRC y también la menos
frecuente, algo que convendrá analizar sin tocar nunca el test para decidir.

### Preguntas de entrevista relacionadas

- ¿Por qué exigir un árbol Git limpio antes de un experimento?
- ¿Qué aporta registrar hashes de configuración, fuentes y checkpoint?
- ¿Por qué un buen AUROC de validation no autoriza una afirmación clínica?
- ¿Por qué fold 10 sigue sellado después del primer entrenamiento real?

## Mission 013 — Split-safe multilabel validation evaluation

### Qué se hizo

Se añadió una frontera completa que recoge `ecg_id`, targets y probabilidades en
orden desde validation y calcula AUROC/AUPRC por clase, macro y micro. Los
resultados son estructuras congeladas y los arrays son copias de solo lectura.

### Por qué se hizo

La loss permite optimizar y seleccionar checkpoints, pero no describe bien la
capacidad de ranking de cada etiqueta. Estas métricas permiten comparar el
baseline sin elegir todavía un threshold ni abrir el test final.

### Decisiones técnicas y alternativas

Se incorporó scikit-learn 1.9.0 para no mantener algoritmos estadísticos propios.
AUPRC significa average precision no interpolada. Se rechazó calcular una media
ignorando clases degeneradas: cualquier clase sin positivos o negativos hace
fallar el resultado para no publicar un macro engañoso.

### Riesgos de leakage revisados

La API comprueba `dataset.split == "validation"` antes de ejecutar el modelo y
rechaza train/test. No calcula thresholds, no cambia la selección por loss y solo
se utilizaron batches sintéticos.

### Archivos principales y tests

- `src/ptbxl/evaluation/multilabel.py`
- `tests/evaluation/test_multilabel.py`
- `docs/context/missions/013_add_multilabel_validation_evaluation.md`

Dieciséis casos prueban identidad/orden, sigmoid, ausencia de gradientes,
métricas perfectas, empates, definición de average precision, splits inválidos,
formas, valores, duplicados y clases degeneradas.

### Resultados obtenidos

La frontera sintética está implementada; no representa rendimiento real del
modelo. Pasaron los 16 casos de evaluación y la suite completa de 152 tests,
Ruff y el build del paquete. El issue `#37` y la única PR `#38` reúnen código,
tests, documentación y cierre; GitHub conserva el estado final de CI y merge.

### Qué debería entender el propietario

AUROC y AUPRC miden ordenación sin fijar un punto de decisión. Macro da el mismo
peso a cada superclase; micro da el mismo peso a cada par ECG-etiqueta. AUPRC
tiene como referencia la prevalencia y aquí usa average precision.

### Preguntas de entrevista relacionadas

- ¿Por qué AUROC puede ser insuficiente con clases desbalanceadas?
- ¿Qué diferencia existe entre macro y micro en multilabel?
- ¿Por qué una clase con un único target hace inválida AUROC?
- ¿Por qué no se calculan todavía F1, sensibilidad o especificidad?

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
