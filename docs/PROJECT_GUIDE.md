# Guía del proyecto PTB-XL ML System

Esta guía explica qué problema resuelve el proyecto, qué se ha construido, por
qué se tomaron las decisiones actuales y cómo deberían construirse las etapas
que faltan. Está pensada para poder leerla sin experiencia previa en ECG o deep
learning y, al mismo tiempo, servir como material de preparación técnica.

> **Estado de esta edición:** las misiones 001–013 están implementadas y
> verificadas. La frontera PyTorch, la CNN y el motor por época existen; el
> fit multi-época, checkpoints y métricas de ranking para validation también
> existen; el entrenamiento real aún no está ejecutado. Este
> proyecto es experimental y no está destinado a uso clínico.

## Cómo leer el estado de cada sección

- **Implementado y verificado:** existe código y pruebas, y cuando corresponde
  también evidencia determinista obtenida con PTB-XL real.
- **Planificado:** explica el diseño previsto o la teoría necesaria, pero aún no
  constituye comportamiento del repositorio.
- **Pendiente de resultados:** reserva el lugar para resultados futuros sin
  inventarlos ni utilizar prematuramente el conjunto de test.

## Contenido

1. [El problema](#parte-1--el-problema)
2. [Los datos](#parte-2--los-datos)
3. [Leakage y separación experimental](#parte-3--leakage-y-separación-experimental)
4. [Señales y preprocessing](#parte-4--señales-y-preprocessing)
5. [Deep learning para este problema](#parte-5--deep-learning-para-este-problema)
6. [Evaluación](#parte-6--evaluación)
7. [Arquitectura de software](#parte-7--arquitectura-de-software)
8. [MLOps y reproducibilidad](#parte-8--mlops-y-reproducibilidad)
9. [Resultados y limitaciones](#parte-9--resultados-y-limitaciones)
10. [Preparación de entrevista](#parte-10--preparación-de-entrevista)

---

## Parte 1 — El problema

**Estado: problema definido y contrato del primer modelo implementado; capacidad
predictiva aún no evaluada.**

### 1.1 Qué representa un ECG

El corazón genera actividad eléctrica al despolarizarse y repolarizarse. Un
electrocardiograma, o ECG, registra cómo cambia esa actividad con el tiempo
desde diferentes puntos de vista del cuerpo.

Una **derivación** no es simplemente un electrodo. Es una vista o combinación
de diferencias de potencial. El ECG estándar de 12 derivaciones contiene:

- seis derivaciones de las extremidades: `I`, `II`, `III`, `aVR`, `aVL`, `aVF`;
- seis derivaciones precordiales: `V1`, `V2`, `V3`, `V4`, `V5`, `V6`.

Las vistas son redundantes de forma útil: observan el mismo fenómeno desde
ángulos distintos. Por eso el orden de las derivaciones y su relación de
amplitud forman parte del contrato de datos, no son un detalle cosmético. La
definición y colocación estándar están descritas en la
[declaración AHA/ACC/HRS sobre ECG de 12 derivaciones](https://doi.org/10.1016/j.jacc.2007.01.024).

### 1.2 Qué intenta predecir el sistema

El objetivo inicial es recibir diez segundos de un ECG de 12 derivaciones y
predecir la presencia de cinco superclases diagnósticas definidas por la
taxonomía oficial de PTB-XL:

| Etiqueta | Significado de la categoría en PTB-XL |
| --- | --- |
| `NORM` | ECG normal |
| `MI` | infarto de miocardio |
| `STTC` | cambios del segmento ST o de la onda T |
| `CD` | alteraciones de la conducción |
| `HYP` | hipertrofia |

Estas etiquetas describen las categorías del dataset. Una predicción de un
modelo experimental no equivale a un diagnóstico médico.

### 1.3 Por qué es clasificación multilabel

El problema es **multilabel**, no multiclase. En multiclase se elige exactamente
una opción, como “gato, perro o caballo”. En multilabel varias condiciones
pueden estar activas a la vez.

El target siempre usa el orden:

```text
[NORM, MI, STTC, CD, HYP]
```

Por ejemplo:

```text
[0, 1, 1, 0, 0]
```

significa que ese registro tiene `MI` y `STTC`. No hay una clase artificial
“MI+STTC”: cada salida se aprende de forma independiente, aunque el modelo pueda
aprender relaciones entre ellas.

De los 21.799 registros de metadatos, 5.144 tienen más de una superclase activa.
Esto confirma que forzar una única clase descartaría información real. La cifra
procede del [informe versionado de labels](../reports/labels/ptbxl_v1.0.3_superclass_summary.json).

---

## Parte 2 — Los datos

**Estado: implementado y verificado con PTB-XL v1.0.3.**

### 2.1 Dataset y versión

El proyecto fija su identidad en **PTB-XL v1.0.3**, no en una URL móvil del tipo
`latest`. PTB-XL es un dataset público de ECG clínicos de 12 derivaciones y diez
segundos. La descripción científica original está en el
[artículo de PTB-XL](https://doi.org/10.1038/s41597-020-0495-6) y los archivos de
esta versión en [PhysioNet PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/).

La versión local verificada contiene:

| Población de metadatos | Registros | Pacientes únicos |
| --- | ---: | ---: |
| PTB-XL v1.0.3 | 21.799 | 18.869 |

El proyecto no sube los datos clínicos a Git. En su lugar versiona el origen y
el SHA-256 de cada fuente pequeña en
[`data/ptbxl_metadata_manifest.json`](../data/ptbxl_metadata_manifest.json) y
[`data/ptbxl_scp_statements_manifest.json`](../data/ptbxl_scp_statements_manifest.json).
Así se puede comprobar qué datos generaron la evidencia sin duplicar el dataset.

### 2.2 Qué contienen los metadatos

`ptbxl_database.csv` contiene una fila por ECG. Para este proyecto, los campos
más importantes son:

- `ecg_id`: identidad única del registro;
- `patient_id`: identidad que permite separar pacientes;
- `strat_fold`: fold oficial del 1 al 10;
- `scp_codes`: códigos SCP anotados y sus valores de likelihood;
- `filename_lr` y `filename_hr`: referencias a las señales de 100 y 500 Hz.

Existen más atributos clínicos y técnicos, pero no se incorporan al target ni
al modelo de forma automática. Añadir uno requeriría una misión que justificase
su uso, su disponibilidad en inferencia y su riesgo de leakage.

### 2.3 Cómo se construyen las cinco etiquetas

El proceso no mantiene una lista manual de códigos. Hace lo siguiente:

1. valida identidad, folds y pacientes en los metadatos;
2. interpreta `scp_codes` con `ast.literal_eval`, nunca con `eval`;
3. carga el catálogo oficial `scp_statements.csv`;
4. conserva únicamente códigos marcados como diagnósticos y asignados a una de
   las cinco superclases objetivo;
5. activa cada columna binaria según la taxonomía oficial;
6. falla si aparece un código desconocido y contabiliza por separado los
   códigos oficiales que no pertenecen al target.

Un código presente activa su categoría sin imponer un umbral de likelihood.
PTB-XL puede utilizar cero para expresar certeza desconocida, por lo que un
filtro arbitrario habría cambiado las etiquetas sin una justificación estable.

Resultado auditable:

| Superclase | Registros etiquetados en los metadatos completos |
| --- | ---: |
| `NORM` | 9.514 |
| `MI` | 5.469 |
| `STTC` | 5.235 |
| `CD` | 4.898 |
| `HYP` | 2.649 |

Las columnas no suman 21.799 porque un registro puede tener varias etiquetas.

### 2.4 Cómo se define la cohorte inicial

El constructor de labels conserva las 21.799 filas, incluso las que no tienen
ninguna de las cinco superclases. Después, una decisión independiente define la
cohorte del primer problema:

```text
incluir si NORM + MI + STTC + CD + HYP >= 1
```

| Resultado | Registros |
| --- | ---: |
| Incluidos en la cohorte | 21.388 |
| Excluidos como `no_target_superclass` | 411 |

Los 411 registros excluidos no se reinterpretan como normales y no se borran de
la tabla maestra de labels. La decisión es reversible y su evidencia está en el
[informe de cohorte](../reports/cohort/ptbxl_v1.0.3_five_superclass_cohort_summary.json).

### 2.5 Splits oficiales

El proyecto respeta los folds recomendados por PTB-XL:

| Uso | Folds | Metadatos | Cohorte de cinco superclases |
| --- | --- | ---: | ---: |
| Entrenamiento | 1–8 | 17.418 | 17.084 |
| Validación | 9 | 2.183 | 2.146 |
| Test final | 10 | 2.198 | 2.158 |

- **Train** permite ajustar parámetros del modelo y estadísticas aprendidas.
- **Validation** permite escoger arquitectura, hiperparámetros, checkpoint y
  umbrales.
- **Test** queda sellado para una evaluación final, una vez tomadas las
  decisiones.

La implementación también verifica que cada paciente pertenezca a un solo fold
y que no exista solapamiento de pacientes entre splits. La política coincide
con el protocolo del
[benchmark público asociado a PTB-XL](https://github.com/helme/ecg_ptbxl_benchmarking).

---

## Parte 3 — Leakage y separación experimental

**Estado: salvaguardas de datos y preprocessing implementadas; las de selección
de modelo y umbrales deberán materializarse con el entrenamiento.**

### 3.1 Qué es leakage

Hay **data leakage** cuando el proceso de entrenamiento recibe información que
no debería tener en el momento de aprender. El resultado puede parecer mejor en
una evaluación, pero no representa cómo funcionaría el sistema con pacientes
nuevos.

Una analogía: train es el material de estudio, validation son simulacros para
mejorar el método y test es el examen final. Mirar las respuestas del examen
para decidir cómo estudiar invalida la nota.

### 3.2 Leakage de paciente

Un paciente puede tener más de un ECG. Si un registro suyo está en train y otro
en test, el modelo puede reconocer patrones propios de esa persona en lugar de
generalizar.

Protección actual:

- el split se hereda de `strat_fold`, no se vuelve a sortear por registro;
- se comprueba el solapamiento de `patient_id` entre cada pareja de splits;
- también se rechaza un paciente asignado a varios folds, aunque ambos folds
  acabasen dentro de train;
- la cohorte y el índice de muestras repiten estas comprobaciones críticas.

### 3.3 Leakage de preprocessing

Calcular la media o desviación usando validation o test deja que su distribución
influya en la representación de train. Aunque no se lean los targets, sigue
siendo información externa al entrenamiento.

Protección actual:

```text
fit del standardizer: solo train (folds 1–8)
transform: parámetros congelados para train, validation, test e inferencia
```

La API de ajuste rechaza cualquier muestra cuyo split no sea `train`, y el
script real selecciona train antes de abrir las señales.

### 3.4 Leakage de selección y umbrales

Una probabilidad multilabel necesita convertirse en una decisión binaria usando
un umbral. Elegir ese umbral mirando test es equivalente a entrenar una pequeña
parte del sistema con el examen final.

Política planificada:

- decidir primero qué métrica guía la selección;
- ajustar el checkpoint y, si procede, los umbrales usando validation;
- congelar modelo, preprocessing y umbrales;
- evaluar test una sola vez para el informe final acordado.

Repetir muchas evaluaciones de test y conservar la mejor también filtra
información, aunque nunca se ejecute backpropagation sobre test.

### 3.5 Auditar test no es seleccionar con test

El proyecto ya ha abierto señales de los tres splits para comprobar únicamente
contratos fijados de antemano: archivo presente, asociación exacta, forma,
frecuencia, derivaciones y valores finitos. Eso es una **auditoría de
integridad**, no selección de modelo.

No se han calculado estadísticas de amplitud de validation/test para cambiar el
preprocessing, ni métricas de modelo, ni umbrales. Esta distinción debe
mantenerse: verificar que el examen existe y tiene el formato correcto no es
consultar sus respuestas para diseñar la solución.

---

## Parte 4 — Señales y preprocessing

**Estado: carga a 100 Hz, contrato de muestra y estandarización global
implementados y verificados.**

### 4.1 Representación canónica

La primera frontera usa `filename_lr`, es decir, los registros oficiales a
100 Hz. Cada señal tiene:

```text
forma:              (1000, 12)
interpretación:     (muestras temporales, derivaciones)
duración:           1000 / 100 Hz = 10 segundos
orden de leads:     I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
```

WFDB devuelve los valores físicos del registro como un array NumPy. La frontera
actual valida forma, frecuencia, orden y finitud; no filtra ni reescala durante
la carga. La [documentación de entrada/salida de WFDB](https://wfdb.readthedocs.io/en/latest/io.html)
describe la lectura de registros y la conversión de muestras almacenadas a
señales físicas.

Se auditaron secuencialmente las 21.388 señales de la cohorte:

| Comprobación | Resultado |
| --- | ---: |
| Cargadas | 21.388 |
| Ausentes | 0 |
| Inválidas | 0 |
| Forma `(1000, 12)` | 21.388 |
| Frecuencia 100 Hz | 21.388 |
| Orden esperado | 21.388 |
| Valores no finitos | 0 |

La evidencia completa está en el
[informe de señales](../reports/signals/ptbxl_v1.0.3_lr_signal_audit.json).

### 4.2 Asociación entre señal y target

`build_sample_index` une la cohorte validada con `filename_lr` mediante
`ecg_id`, exigiendo una relación uno-a-uno. No copia esas rutas a una nueva tabla
persistente.

`load_sample` abre una señal bajo demanda y devuelve conjuntamente:

- `ecg_id`, `patient_id`, fold y split;
- señal NumPy `(1000, 12)`;
- frecuencia y orden de derivaciones;
- target `float32` con forma `(5,)` y orden fijo.

Construir el índice no abre señales. Esta carga **lazy** evita guardar toda la
cohorte en memoria y deja una única fuente de verdad para futuros consumidores.

### 4.3 Estandarización global entrenada solo con train

El primer preprocessing usa una media `mu` y desviación poblacional `sigma`
compartidas por todos los tiempos, ECG y derivaciones de train:

```text
x_standardized = (x - mu) / sigma
```

El cálculo recorre las señales una a una, combina momentos en `float64` y no
materializa un tensor gigante. Los parámetros reales son:

| Parámetro | Valor |
| --- | ---: |
| ECG de train observados | 17.084 |
| Valores escalares observados | 205.008.000 |
| Media global | -0,0008252533901116082 |
| Desviación estándar poblacional | 0,23222258117564917 |

El artefacto
[`ptbxl_v1.0.3_train_global_standardizer.json`](../reports/preprocessing/ptbxl_v1.0.3_train_global_standardizer.json)
incluye configuración, orden de leads, conteos y hashes de las fuentes. Dos
ajustes completos produjeron exactamente los mismos bytes y SHA-256:

```text
f791aeb9795c669a54a391d979f69806ccdca19f05128a9fde8f408ec36090bc
```

La transformación devuelve `float32` y conserva `(1000, 12)`.

### 4.4 Por qué no se normaliza cada ECG o cada derivación

- **Por registro:** obligar a que cada ECG tenga media cero y desviación uno
  eliminaría diferencias globales de amplitud entre personas y registros.
- **Por derivación:** cambiaría la escala relativa entre las doce vistas.
- **Global compartida:** controla la escala numérica conservando esas relaciones
  relativas. Es además comparable con la regla estadística del benchmark
  público de PTB-XL.

No significa que sea universalmente la mejor opción. Significa que es un
baseline simple, reproducible y comprobable. Comparar otras transformaciones
requeriría experimentos gobernados por train/validation.

### 4.5 Transformaciones deliberadamente ausentes

Por ahora no hay filtrado, corrección de baseline wander, denoising, resampling,
augmentación, normalización por registro ni normalización por lead. PTB-XL
ofrece señales ya preparadas técnicamente, pero eso no demuestra que ninguna de
esas técnicas pueda ayudar; simplemente evita añadir complejidad antes de tener
un baseline medible.

---

## Parte 5 — Deep learning para este problema

**Estado: frontera de datos PyTorch y baseline CNN implementados; entrenamiento
aún planificado.**

Esta parte separa los contratos ya implementados de las decisiones de
entrenamiento que todavía no existen. La CNN actual es el baseline inicial
elegido por simplicidad y facilidad de prueba, no por resultados de validation.

### 5.1 De NumPy a un batch para Conv1D

La representación canónica sigue siendo `(tiempo, leads) = (1000, 12)` porque
coincide con WFDB y es independiente del framework. `PTBXLDataset` realiza una
única conversión en la frontera con PyTorch:

```text
muestra NumPy:       (1000, 12)
tensor de muestra:   (12, 1000)
batch:               (batch_size, 12, 1000)
```

Así las 12 derivaciones se convierten en canales y una futura convolución podrá
recorrer el tiempo. El adaptador reutiliza `load_sample`, aplica el standardizer
congelado y devuelve almacenamiento contiguo `float32`; no reconstruye joins,
labels ni reglas de split.

Cada Dataset contiene exactamente un split declarado y conserva `ecg_id`,
`patient_id`, fold, split y `filename_lr`. `build_dataloader` usa la collation
estándar de PyTorch, exige una seed si hay shuffle y produce señales
`(B, 12, 1000)` y targets `(B, 5)`. La construcción sigue siendo lazy: el WFDB
solo se abre al solicitar un elemento.

### 5.2 Qué aprende una Conv1D

Una capa `Conv1D` desliza filtros pequeños a lo largo del eje temporal. Un
filtro temprano puede responder a pendientes, picos o patrones cortos; capas
posteriores combinan esos patrones en representaciones más amplias.

Conceptos esenciales:

- **canales de entrada:** las 12 derivaciones;
- **kernel:** ventana temporal que observa cada filtro;
- **stride:** salto entre posiciones;
- **padding:** tratamiento de los bordes;
- **canales de salida:** número de patrones aprendidos por la capa;
- **receptive field:** cantidad de señal original que puede influir en una
  activación profunda.

`SmallECGCNN` implementa tres bloques con canales `32, 64, 128` y kernels
`7, 5, 3`. Cada convolución conserva longitud y cada max pooling la divide por
dos:

```text
(B, 12, 1000)
  -> (B, 32, 500)
  -> (B, 64, 250)
  -> (B, 128, 125)
```

El campo receptivo local antes del pooling global es de 30 muestras, unos
0,3 segundos a 100 Hz. Puede representar morfología local, pero es una limitación
explícita para relaciones temporales largas. Modificarlo deberá ser un
experimento gobernado por validation, nunca por test. La API y formas de la
convolución temporal están documentadas en
[`torch.nn.Conv1d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html).

### 5.3 Pooling y representación del registro

El pooling reduce la dimensión temporal. En el baseline puede:

- disminuir coste y memoria;
- dar tolerancia a pequeños desplazamientos en el tiempo;
- resumir una secuencia completa antes de la cabeza de clasificación.

Después de los tres bloques, `AdaptiveAvgPool1d(1)` promedia las 125 posiciones
restantes y obtiene un vector de 128 características por ECG. Demasiado pooling
o un promedio global pueden diluir eventos breves; ese equilibrio deberá
evaluarse con validation.

### 5.4 Cinco logits, no una softmax

La última capa emite cinco **logits**, números reales sin limitar:

```text
logits.shape = (batch_size, 5)
```

Se aplicará una sigmoid independiente a cada logit para interpretar una
probabilidad por superclase:

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

No se usa softmax porque softmax fuerza a repartir toda la probabilidad entre
clases mutuamente excluyentes. Aquí `MI` y `STTC`, por ejemplo, pueden coexistir.

### 5.5 BCEWithLogitsLoss

La pérdida base prevista es `BCEWithLogitsLoss`: binary cross-entropy aplicada
a cada una de las cinco salidas y combinada con la sigmoid de forma
numéricamente estable. Recibe logits, no probabilidades ya transformadas. La
[documentación oficial de BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
explica esta combinación. La compatibilidad matemática y el backward ya se han
probado con tensores sintéticos, pero todavía no existe un training loop.

Los pesos de clase o `pos_weight` no se añadirán automáticamente. Pueden mejorar
algún objetivo bajo desbalance, pero también cambian el compromiso entre falsos
positivos y falsos negativos; necesitarán hipótesis, experimento y validación.

### 5.6 Contrato implementado de la primera CNN

Los tests sintéticos demuestran:

- tensor de entrada `float32` `(B, 12, 1000)` y salida `(B, 5)`;
- forward, loss y backward finitos;
- gradiente para cada parámetro entrenable;
- ausencia de sigmoid y softmax dentro del modelo;
- configuración inmutable y validada;
- 38.597 parámetros entrenables en la configuración por defecto;
- forward determinista en modo evaluación para estado e input fijos.

Arquitecturas más complejas —ResNet, transformers o ensembles— no deben preceder
a entrenar y evaluar correctamente este baseline pequeño.

---

## Parte 6 — Evaluación

**Estado: predicciones y métricas de ranking implementadas para validation;
thresholds y métricas de punto operativo todavía planificados.**

### 6.1 De probabilidades a decisiones

La sigmoid produce cinco valores entre cero y uno. Para calcular métricas
binarias se necesita un umbral `t`:

```text
predicción positiva si probability >= t
```

`0,5` es un punto de partida, no una verdad universal. Puede usarse un umbral
global o uno por clase, pero la regla debe elegirse y congelarse con validation.

### 6.2 Matriz de confusión por etiqueta

Para cada superclase:

- **TP:** el target era positivo y el sistema predijo positivo;
- **TN:** el target era negativo y predijo negativo;
- **FP:** predijo una condición ausente;
- **FN:** no detectó una condición presente.

A partir de ahí:

```text
sensibilidad (recall) = TP / (TP + FN)
especificidad          = TN / (TN + FP)
precision              = TP / (TP + FP)
F1                     = 2 * precision * recall / (precision + recall)
```

No existe una métrica perfecta. Sensibilidad alta reduce falsos negativos;
especificidad alta reduce falsos positivos. El uso previsto determina cuál es
más costoso, y este proyecto no debe hacer afirmaciones clínicas sin una
validación apropiada.

### 6.3 AUROC y AUPRC

- **AUROC** resume cómo ordena el modelo positivos frente a negativos mientras
  varía el umbral. Usa la relación entre sensibilidad y tasa de falsos
  positivos.
- **AUPRC** resume precision frente a recall. Suele mostrar con más claridad el
  rendimiento de la clase positiva cuando esta es poco frecuente y su baseline
  depende de la prevalencia.

Ambas evitan escoger primero un umbral, pero no sustituyen las métricas del punto
operativo finalmente elegido. La implementación usa las funciones probadas de
scikit-learn 1.9.0 y llama AUPRC a su *average precision* no interpolada. No usa
la integral trapezoidal de la curva PR, que puede producir un valor diferente.
Si una clase no contiene al menos un positivo y un negativo, el cálculo falla de
forma explícita porque AUROC no está definida.

### 6.4 Macro frente a micro

- **Macro:** calcula la métrica por clase y después promedia. Cada superclase
  pesa igual, por lo que una categoría menos frecuente como `HYP` no queda
  escondida por `NORM`.
- **Micro:** agrega todos los pares target-predicción antes de calcular la
  métrica. Las decisiones de las clases frecuentes pesan más.

La frontera implementada reporta ambas perspectivas y también valores por
clase, siempre en orden `NORM`, `MI`, `STTC`, `CD`, `HYP`. Macro es la media no
ponderada de las cinco clases; micro aplana todos los pares etiqueta-predicción.
Un único promedio podría ocultar un fallo importante.

### 6.5 Protocolo planificado de selección

1. Entrenar exclusivamente con folds 1–8.
2. Evaluar checkpoints en fold 9 con una métrica primaria declarada.
3. Seleccionar checkpoint e hiperparámetros con validation.
4. Ajustar umbral global o por clase únicamente con validation, si forma parte
   del experimento.
5. Congelar código, configuración, standardizer, pesos y umbrales.
6. Ejecutar fold 10 una sola vez para la evaluación final autorizada.
7. Reportar métricas por clase, macro y micro junto con intervalos o variabilidad
   cuando el diseño experimental lo permita.

El catálogo de métricas multilabel de
[scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics)
servirá como referencia de API si esa dependencia se justifica más adelante;
actualmente no forma parte del proyecto.

---

## Parte 7 — Arquitectura de software

**Estado: arquitectura de datos, preprocessing, entrada a PyTorch y baseline CNN
implementados; entrenamiento planificado.**

### 7.1 Flujo actual

```text
PTB-XL v1.0.3 + manifiestos SHA-256
                  |
                  v
       validar metadatos y folds
                  |
                  v
  derivar 5 labels desde scp_statements
                  |
                  v
 definir cohorte: >= 1 target activo
                  |
                  v
 asociar ecg_id <-> filename_lr (índice transitorio)
                  |
                  v
 cargar una muestra lazy: identidad + señal + targets
                  |
                  v
 fit global solo con train -> artefacto JSON congelado
                  |
                  v
       transform de una señal float32
                  |
                  v
 adapter PyTorch: tensor (12, 1000) + target (5,)
                  |
                  v
       CNN pequeña -> cinco logits
                  |
                  v
 train/validation -> checkpoint por loss
                  |
                  v
 validation -> sigmoid -> AUROC/AUPRC
```

Cada frontera valida su entrada antes de producir la siguiente. Así un fallo de
identidad o leakage aparece cerca de su causa, no durante el entrenamiento.

### 7.2 Responsabilidad de los módulos actuales

| Ruta | Responsabilidad |
| --- | --- |
| [`data/metadata.py`](../src/ptbxl/data/metadata.py) | validar IDs y folds, asignar splits y detectar solapamientos de pacientes |
| [`data/reporting.py`](../src/ptbxl/data/reporting.py) | hashes, verificación de fuentes e informes JSON deterministas |
| [`data/labels.py`](../src/ptbxl/data/labels.py) | interpretar SCP y derivar las cinco columnas binarias oficiales |
| [`data/cohort.py`](../src/ptbxl/data/cohort.py) | incluir registros con al menos un target y auditar exclusiones |
| [`data/signals.py`](../src/ptbxl/data/signals.py) | asociar y validar registros WFDB a 100 Hz |
| [`data/samples.py`](../src/ptbxl/data/samples.py) | componer y cargar una muestra completa, lazy y agnóstica de framework |
| [`data/pytorch.py`](../src/ptbxl/data/pytorch.py) | adaptar un split a tensores channel-first y batches reproducibles |
| [`preprocessing/standardization.py`](../src/ptbxl/preprocessing/standardization.py) | ajustar, transformar, guardar y cargar la estandarización global |
| [`models/cnn.py`](../src/ptbxl/models/cnn.py) | validar el batch y producir cinco logits con la CNN configurable |
| [`training/engine.py`](../src/ptbxl/training/engine.py) | ejecutar una época de optimización o evaluación de loss con semánticas separadas |
| [`training/fit.py`](../src/ptbxl/training/fit.py) | orquestar épocas y restaurar el primer mínimo de validation desde un checkpoint seguro |
| [`evaluation/multilabel.py`](../src/ptbxl/evaluation/multilabel.py) | recoger predicciones de validation y calcular AUROC/AUPRC por clase, macro y micro |

Los scripts bajo [`scripts/`](../scripts/) son puntos de entrada reproducibles
que coordinan estas funciones. La lógica reutilizable permanece bajo `src/` y
los notebooks, cuando existan, serán solo para exploración.

### 7.3 Por qué se separan estas responsabilidades

- El loader de señales no decide labels.
- La construcción de labels no decide la cohorte.
- El índice de muestras no persiste otra copia de las rutas.
- El standardizer no conoce el modelo ni los targets.
- El Dataset de PyTorch delega la composición y el preprocessing en esas capas
  y solo cambia representación y batching.

Esta separación reduce duplicación y permite probar las transformaciones
importantes con datos sintéticos pequeños.

### 7.4 Arquitectura planificada

Las siguientes fronteras mínimas quedan por añadir:

1. comando/configuración de experimento reproducible y tracking local simple.
2. baseline real y comparación controlada usando exclusivamente validation.
3. selección y congelación de umbrales.
4. evaluación final sellada, error analysis, interpretación e inferencia.

---

## Parte 8 — MLOps y reproducibilidad

**Estado: entorno, pruebas, CI, seeds, fit, checkpoints y métricas de ranking
implementados; tracking comparativo planificado.**

### 8.1 Entorno reproducible

- Python 3.11 es la versión del proyecto.
- `pyproject.toml` declara dependencias directas.
- `uv.lock` fija la resolución exacta y no se edita manualmente.
- `uv sync --locked` impide que CI resuelva versiones diferentes en silencio.
- PyTorch 2.13.0 está fijado actualmente; el mismo paquete funciona en CPU y
  detecta la GPU NVIDIA local para futuros entrenamientos.
- scikit-learn 1.9.0 proporciona AUROC y average precision probadas en lugar de
  mantener implementaciones estadísticas propias.
- La arquitectura CNN vive en una dataclass congelada; canales, kernels y
  dropout pueden registrarse sin depender de valores ocultos en un script.
- Los datos raw y productos grandes permanecen ignorados.

### 8.2 Pruebas y CI

La suite usa datos sintéticos para comprobar casos normales y fallos: IDs
duplicados, leakage, códigos inválidos, asociaciones ambiguas, señales con forma
incorrecta, splits no permitidos y artefactos incoherentes.

GitHub Actions ejecuta en cada pull request:

```bash
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Los datos reales no se descargan en CI. Esto mantiene el workflow rápido y
estable, mientras informes deterministas versionados conservan la evidencia de
las validaciones completas ejecutadas localmente.

### 8.3 Manifiestos, informes y procedencia

Hay tres tipos de archivos con propósitos distintos:

- **manifiesto:** identifica una fuente por versión, URL y checksum;
- **informe:** conserva hechos agregados y deterministas de una auditoría;
- **artefacto aprendido:** contiene estado que debe reutilizarse, como la media
  y desviación del standardizer, además de su procedencia.

No se incluyen timestamps ni rutas absolutas en esos JSON. Dos ejecuciones con
las mismas fuentes deben producir los mismos bytes cuando el contrato promete
determinismo.

### 8.4 Flujo de desarrollo

Cada cambio material se ejecuta como una misión pequeña:

```text
issue -> rama -> implementación y tests -> PR -> CI -> squash merge -> cierre
```

Las decisiones estables viven en
[`DECISIONS.md`](context/DECISIONS.md), el punto actual de reanudación en
[`STATUS.md`](context/STATUS.md) y el contrato detallado de cada etapa en
[`docs/context/missions/`](context/missions/). Esto permite retomar el trabajo
sin reconstruir la historia desde una conversación.

### 8.5 Lo que deberá añadirse con entrenamiento

El núcleo ya separa `train_one_epoch` y `evaluate_loss`. La primera activa
gradientes y optimizer steps; la segunda usa modo eval e inference. Ambas
ponderan la loss por muestras, para que un último batch corto no pese igual que
uno completo. Una seed reinicia Python, NumPy, PyTorch y CUDA, aunque hardware y
kernels GPU todavía pueden limitar la reproducibilidad absoluta.

`fit` añade un número fijo de épocas y selecciona el primer mínimo de loss de
validation. La API comprueba los roles declarados por los Dataset y rechaza test
como fuente de selección. El mejor estado de modelo y optimizer se escribe
atómicamente junto con el historial completo y procedencia; la carga usa
`weights_only=True` y restaura ese estado antes de devolver el resultado.

Antes de considerar reproducible un experimento de modelo harán falta, como
mínimo:

- configuración versionada de datos, modelo, optimizer y scheduler;
- seeds y registro de los límites de determinismo del hardware;
- identidad del commit y del artefacto de preprocessing;
- historial de métricas de train y validation;
- criterio de selección de checkpoint declarado de antemano;
- pesos, configuración y umbrales vinculados entre sí;
- comando de evaluación que no pueda refit ni seleccionar usando test.

Un sistema de tracking se justificará cuando existan suficientes experimentos
para que archivos simples dejen de ser claros. Introducirlo antes añadiría una
abstracción sin un problema real que resolver.

---

## Parte 9 — Resultados y limitaciones

**Estado: existen resultados de integridad y preprocessing; resultados de modelo
pendientes.**

### 9.1 Evidencia conseguida hasta ahora

| Etapa | Resultado verificable |
| --- | --- |
| Identidad de datos | PTB-XL v1.0.3 fijado con manifiestos SHA-256 |
| Metadatos | 21.799 ECG, 18.869 pacientes, folds válidos y sin solapamiento de pacientes |
| Labels | cinco superclases derivadas de la taxonomía oficial en orden fijo |
| Cohorte | 21.388 incluidos y 411 exclusiones explícitas |
| Señales | 21.388/21.388 válidas a 100 Hz, forma `(1000, 12)`, sin no-finitos |
| Muestras | identidad, señal, targets y split compuestos de forma lazy |
| Preprocessing | 17.084 ECG de train, parámetros globales deterministas y congelados |
| PyTorch | Dataset lazy por split y batches `(B, 12, 1000)` / `(B, 5)` verificados con datos sintéticos y un smoke check real |
| Modelo | CNN de 38.597 parámetros, entrada/salida, BCE y gradientes verificados sintéticamente |
| Entrenamiento | train/evaluate por época separados, loss ponderada y seeds verificados sintéticamente |
| Checkpoint | primer mínimo de validation, historial/procedencia y round-trip exacto verificados sintéticamente |
| Evaluación | identidades ordenadas y AUROC/AUPRC por clase, macro y micro verificados sintéticamente |
| Calidad de software | 152 tests, Ruff y build superados al cerrar la misión 013 |

Estos son resultados de **ingeniería e integridad de datos**, no rendimiento
predictivo.

### 9.2 Resultados que todavía no existen

No se dispone aún de:

- loss de entrenamiento o validación;
- AUROC, AUPRC, F1, sensibilidad o especificidad de un modelo entrenado real;
- checkpoint seleccionado;
- umbrales;
- comparación de arquitecturas;
- resultado final en fold 10.

Cuando existan, esta sección deberá registrar configuración, criterio de
selección, métricas por clase/macro/micro y limitaciones. Una tabla vacía no debe
rellenarse con estimaciones o cifras copiadas de otro proyecto.

### 9.3 Limitaciones actuales

- Todavía no se ha demostrado capacidad predictiva.
- Solo se ha elegido la señal oficial de 100 Hz; 500 Hz queda como posible
  experimento posterior.
- Las etiquetas son anotaciones del dataset, no una verdad clínica perfecta.
- Se excluyen registros sin ninguna superclase objetivo en esta primera tarea.
- El preprocessing inicial es intencionadamente mínimo y no compara filtros o
  augmentations.
- PTB-XL representa una población y contexto de adquisición concretos; una
  evaluación interna no demuestra generalización a otro hospital o dispositivo.
- El proyecto no es un producto sanitario ni ha sido validado para decisiones
  clínicas.

---

## Parte 10 — Preparación de entrevista

### 10.1 ¿Por qué el problema es multilabel y no multiclase?

Porque un ECG puede tener varias superclases simultáneas. El target contiene
cinco bits independientes y la salida prevista son cinco logits con sigmoid. Una
softmax impondría exclusividad y representaría mal los datos.

### 10.2 ¿Por qué separar pacientes y no solo registros?

Varios ECG del mismo paciente comparten características biológicas y de
adquisición. Repartirlos entre train y test permitiría memorizar información de
la persona. La unidad de aislamiento es por ello `patient_id`, aunque la unidad
de predicción sea un ECG.

### 10.3 ¿Por qué se usan los folds oficiales?

Permiten comparabilidad con el protocolo publicado y evitan diseñar un split a
partir de distribuciones observadas. Folds 1–8 entrenan, 9 valida y 10 permanece
como test final.

### 10.4 ¿Por qué construir labels desde `scp_statements.csv`?

Porque es la taxonomía versionada del dataset. Una lista copiada manualmente
podría quedar incompleta, mezclar códigos no diagnósticos o perder trazabilidad.
El código además falla ante códigos desconocidos en vez de ignorarlos.

### 10.5 ¿Por qué los registros sin target no se convierten en `NORM`?

Ausencia de una de las cinco superclases no demuestra normalidad. `NORM` es una
etiqueta oficial positiva. Los 411 all-zero se conservan en labels y se excluyen
con una razón explícita de la cohorte inicial.

### 10.6 ¿Por qué validar el join antes de introducir PyTorch?

Un modelo puede entrenar aunque señales y targets estén mal asociados. Probar la
relación uno-a-uno por `ecg_id` en una capa pequeña aísla ese riesgo antes de que
quede oculto dentro de workers, batches y transforms.

### 10.7 ¿Por qué el índice de muestras es transitorio?

`filename_lr` ya existe en los metadatos oficiales y puede unirse de forma
determinista. Persistir otra tabla duplicaría información derivada y crearía una
fuente adicional que podría quedar desactualizada.

### 10.8 ¿Por qué conservar `(samples, leads)` hasta PyTorch?

Es la forma natural que entrega WFDB y mantiene la capa de datos independiente
del framework. La transposición a `(channels, samples)` pertenece a un único
adaptador, donde `Conv1D` la necesita. Hacer una copia contigua en ese punto
evita que el resto del pipeline tenga que conocer strides de una vista NumPy
transpuesta.

### 10.9 ¿Cómo se ajusta la media sin cargar todo el dataset?

Cada ECG aporta conteo, media y suma de desviaciones cuadráticas. Esos momentos
se combinan secuencialmente corrigiendo la diferencia entre medias de bloques.
Así se obtiene la misma media y varianza poblacional en `float64` usando memoria
acotada.

### 10.10 ¿Por qué el standardizer solo usa train?

Sus parámetros dependen de la distribución observada, por lo que también son
parámetros aprendidos. Incluir validation o test transferiría información de
esos conjuntos a la entrada del modelo. Se ajusta una vez con train y después se
congela.

### 10.11 ¿Por qué una estadística global y no por ECG?

La normalización por ECG borra diferencias de media y escala entre registros.
Una transformación global mantiene esas relaciones y proporciona un baseline
simple compatible con el benchmark publicado. Su idoneidad final debe comprobarse
experimentalmente, no asumirse.

### 10.12 ¿Por qué `BCEWithLogitsLoss` en lugar de cross-entropy multiclase?

Cada etiqueta es una decisión binaria y pueden coexistir varias positivas.
`BCEWithLogitsLoss` trata las cinco salidas como binarias y combina logits con
sigmoid de forma estable. La cross-entropy multiclase normalmente presupone una
única clase correcta.

### 10.13 ¿Qué diferencia hay entre macro y micro?

Macro da el mismo peso final a cada clase al promediar métricas por separado.
Micro junta todas las decisiones, por lo que dominan las clases con más
ejemplos. Reportar ambas evita confundir rendimiento global con equilibrio entre
superclases.

### 10.14 ¿Por qué no basta AUROC?

AUROC mide ranking a través de muchos umbrales, pero puede ocultar un rendimiento
práctico pobre en la clase positiva cuando hay desbalance. AUPRC y métricas en un
umbral congelado aportan perspectivas distintas; además deben verse resultados
por clase.

### 10.15 ¿Cómo se protege el test durante el desarrollo?

Las decisiones se toman con train y validation. El test solo puede pasar por
comprobaciones estructurales predefinidas hasta que modelo, checkpoint,
preprocessing y umbrales estén congelados. Una evaluación final no debe
convertirse después en otra ronda de tuning.

### 10.16 ¿Qué harías a continuación y por qué?

Construiría un comando de experimento reproducible que una datos, modelo, fit,
checkpoint y las métricas ya probadas en un registro local estructurado. Después
ejecutaría el baseline real usando validation, manteniendo thresholds y test
final fuera hasta congelar el protocolo.

### 10.17 ¿Qué limitación temporal tiene la primera CNN?

Sus tres convoluciones y poolings producen un campo receptivo local de unas 30
muestras antes del promedio global. Detecta patrones locales distribuidos por
los diez segundos, pero el promedio no modela explícitamente el orden entre
eventos alejados. Es una limitación medible que podría motivar un experimento
posterior, no una razón para complicar el baseline antes de entrenarlo.

---

## Referencias y evidencia local

Fuentes externas principales:

- [PTB-XL v1.0.3 en PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)
- [PTB-XL: a large publicly available electrocardiography dataset](https://doi.org/10.1038/s41597-020-0495-6)
- [Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL](https://doi.org/10.1109/JBHI.2020.3022989)
- [Código del benchmark PTB-XL](https://github.com/helme/ecg_ptbxl_benchmarking)
- [AHA/ACC/HRS: estandarización del ECG, parte I](https://doi.org/10.1016/j.jacc.2007.01.024)

Evidencia versionada del proyecto:

- [metadatos](../reports/metadata/ptbxl_v1.0.3_summary.json)
- [labels](../reports/labels/ptbxl_v1.0.3_superclass_summary.json)
- [cohorte](../reports/cohort/ptbxl_v1.0.3_five_superclass_cohort_summary.json)
- [auditoría de señales](../reports/signals/ptbxl_v1.0.3_lr_signal_audit.json)
- [standardizer de train](../reports/preprocessing/ptbxl_v1.0.3_train_global_standardizer.json)

## Cómo mantener esta guía

Este archivo sí debe evolucionar con el proyecto. Después de cada misión
material:

1. actualizar solo las secciones afectadas;
2. mover una etapa de “planificada” a “implementada” únicamente cuando existan
   código, checks y evidencia suficiente;
3. enlazar el nuevo artefacto, informe o contrato en vez de copiar todos sus
   detalles;
4. registrar resultados de modelo con su configuración y protocolo;
5. conservar las limitaciones y no reescribir decisiones después de mirar test;
6. añadir preguntas de entrevista cuando una nueva decisión aporte una lección
   generalizable.

`PROJECT_GUIDE.md` explica el sistema. `STATUS.md` indica dónde reanudarlo,
`DECISIONS.md` conserva reglas estables y los archivos de misión documentan la
aceptación detallada. Mantener esos papeles separados evita que la guía se
convierta en un diario difícil de estudiar.
