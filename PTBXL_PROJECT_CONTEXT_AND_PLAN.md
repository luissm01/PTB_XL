# PTB-XL Machine Learning Engineering Project

## 1. Visión general

Este proyecto personal tiene como objetivo desarrollar, desde cero, un sistema completo de Machine Learning aplicado a señales ECG utilizando el dataset **PTB-XL**.

El proyecto no se plantea únicamente como un experimento académico ni como un notebook de clasificación. Su propósito es cubrir de forma integrada:

- comprensión del dominio clínico;
- exploración y preprocesamiento de señales ECG;
- diseño experimental;
- entrenamiento y evaluación de modelos;
- interpretabilidad;
- calibración e incertidumbre;
- buenas prácticas de ingeniería de software;
- seguimiento de experimentos;
- testing;
- integración continua;
- empaquetado;
- despliegue;
- monitorización.

El resultado final debe ser un repositorio público, reproducible y defendible en entrevistas para puestos de:

- Machine Learning Engineer;
- AI Engineer;
- Research Engineer;
- Biomedical AI Engineer;
- Signal Processing Engineer;
- Healthcare Machine Learning Engineer.

El proyecto se desarrollará durante aproximadamente **8–12 semanas**, principalmente en tardes y fines de semana, utilizando herramientas gratuitas y evitando servicios cloud de pago.

---

## 2. Contexto del desarrollador

El desarrollador del proyecto es graduado en Ingeniería de la Salud y está terminando un máster en Inteligencia Artificial.

Cuenta con experiencia profesional en:

- Java;
- jBPM;
- FHIR;
- SQL;
- Docker;
- software sanitario.

También tiene experiencia académica o personal con:

- Python;
- PyTorch;
- TensorFlow;
- Scikit-learn;
- clasificación;
- segmentación médica;
- CNN;
- U-Net;
- MLflow;
- DVC;
- GitHub Actions;
- despliegues básicos en AWS;
- investigación en imagen médica;
- publicación científica sobre clasificación de tumores cerebrales en resonancia magnética.

La principal carencia que este proyecto pretende cubrir es la aplicación práctica de ingeniería de Machine Learning:

- diseño de sistemas;
- APIs;
- empaquetado;
- CI/CD;
- testing;
- seguimiento de modelos;
- monitorización;
- despliegue;
- mantenibilidad;
- reproducibilidad.

---

## 3. Objetivo técnico

Construir un sistema reproducible que reciba un ECG de:

- 12 derivaciones;
- 10 segundos de duración;
- frecuencia de muestreo inicial de 100 Hz;

y prediga una o varias de las cinco superclases diagnósticas de PTB-XL:

- `NORM`: ECG normal;
- `MI`: infarto de miocardio;
- `STTC`: alteraciones ST/T;
- `CD`: trastornos de conducción;
- `HYP`: hipertrofia.

La tarea será de **clasificación multietiqueta**, ya que un mismo ECG puede contener más de una alteración.

El sistema final deberá incluir:

1. carga y validación del dataset;
2. preprocesamiento reproducible;
3. generación segura de los splits;
4. baseline clásico;
5. baseline de deep learning;
6. modelo principal;
7. evaluación multietiqueta;
8. calibración;
9. selección de umbrales;
10. interpretabilidad temporal y por derivación;
11. inferencia reproducible;
12. API;
13. tests;
14. CI;
15. Docker;
16. seguimiento de experimentos;
17. registro de modelos;
18. monitorización básica;
19. demo gratuita.

---

## 4. Pregunta principal de investigación

> ¿Puede una CNN 1D ligera clasificar las cinco superclases diagnósticas de PTB-XL con probabilidades calibradas y explicaciones temporales y por derivación suficientemente consistentes para integrarla en un servicio de inferencia mantenible?

Esta pregunta permite estudiar:

- diferencias entre modelos clásicos y redes profundas;
- efecto del preprocesamiento;
- efecto del desbalance;
- selección de umbrales por clase;
- calibración;
- incertidumbre;
- robustez al ruido;
- interpretabilidad;
- abstención ante entradas dudosas;
- coste y latencia de inferencia.

No se pretende crear una nueva arquitectura de investigación de frontera. El valor principal estará en ejecutar rigurosamente todo el ciclo de vida de Machine Learning.

---

## 5. Alcance inicial

### Incluido

- PTB-XL a 100 Hz.
- Cinco superclases diagnósticas.
- Splits oficiales separados por paciente.
- Clasificación multietiqueta.
- Baseline clásico.
- CNN 1D sencilla.
- ResNet 1D o arquitectura equivalente.
- Métricas por clase y agregadas.
- Calibración.
- Umbrales por clase.
- Interpretabilidad.
- API de inferencia.
- Docker.
- MLflow.
- GitHub Actions.
- Tests.
- Demo gratuita.

### Fuera del alcance inicial

- Clasificación de los 71 códigos SCP.
- Detección latido a latido.
- Segmentación automática de complejos P-QRS-T.
- Entrenamiento distribuido.
- Kubernetes.
- Kubeflow.
- Airflow.
- Spark.
- Terraform.
- servicios cloud de pago;
- uso clínico real;
- validación clínica;
- integración con dispositivos médicos reales;
- despliegue hospitalario.

---

## 6. Decisiones iniciales sobre los datos

### Dataset

Se utilizará **PTB-XL**, disponible en PhysioNet.

Archivos principales:

- `ptbxl_database.csv`;
- `scp_statements.csv`;
- `records100/`;
- `records500/`.

### Frecuencia inicial

Se utilizarán los registros de **100 Hz**.

Cada entrada tendrá aproximadamente la forma:

```text
12 derivaciones × 1000 muestras
```

Esto reduce el coste computacional y permite entrenar modelos en local o Google Colab gratuito.

La versión de 500 Hz puede estudiarse como ampliación futura.

### Splits

Se respetará el campo `strat_fold`.

Distribución inicial:

- folds 1–8: entrenamiento;
- fold 9: validación;
- fold 10: test final.

El conjunto de test no se utilizará para:

- elegir arquitectura;
- seleccionar hiperparámetros;
- seleccionar umbrales;
- decidir preprocesamiento;
- calibrar el modelo.

Se implementarán tests automáticos para verificar que no existen pacientes compartidos entre entrenamiento, validación y test.

---

## 7. Evaluación

### Métricas principales

- Macro ROC-AUC.
- Macro PR-AUC.
- Macro F1.
- F1 por clase.
- Sensibilidad por clase.
- Especificidad por clase.

### Métricas posteriores

- Brier Score.
- Expected Calibration Error.
- Curvas de calibración.
- Cobertura frente a rendimiento.
- Porcentaje de abstención.
- Latencia de inferencia.
- Robustez ante ruido.

La accuracy no se utilizará como métrica principal debido al carácter multietiqueta y al desbalance.

---

## 8. Modelos previstos

### Baseline 1: modelo clásico

Se estudiará un modelo clásico sobre características sencillas o representaciones resumidas.

Posibles opciones:

- regresión logística;
- random forest;
- gradient boosting.

Su objetivo no es obtener el mejor resultado, sino disponer de una referencia fácil de interpretar.

### Baseline 2: CNN 1D sencilla

Primera red profunda:

- convoluciones temporales;
- normalización;
- activaciones;
- pooling;
- dropout;
- capa de salida multietiqueta.

Debe ser pequeña, clara y fácil de depurar.

### Modelo principal: ResNet 1D

Se utilizará una arquitectura residual 1D o una alternativa similar.

El modelo principal se elegirá atendiendo a:

- rendimiento;
- estabilidad;
- calibración;
- coste computacional;
- latencia;
- interpretabilidad;
- facilidad de despliegue.

No se utilizarán Transformers salvo que exista una hipótesis clara que justifique su inclusión.

---

## 9. Preprocesamiento

El proyecto comenzará con el mínimo preprocesamiento posible.

Posibles líneas experimentales:

- señal original a 100 Hz;
- normalización global;
- normalización por ECG;
- normalización por derivación;
- filtrado pasa banda;
- eliminación de deriva de línea base;
- clipping de valores extremos;
- augmentations con ruido;
- escalado de amplitud;
- pérdida temporal de segmentos;
- ocultación de derivaciones.

Cada transformación deberá:

- estar implementada fuera de notebooks;
- ser reproducible;
- estar configurada;
- tener tests cuando corresponda;
- evitar usar información del conjunto de test;
- poder desactivarse.

No se aplicará filtrado complejo por defecto sin comprobar que aporta valor.

---

## 10. Interpretabilidad

La interpretabilidad será una parte central del proyecto, no un elemento decorativo.

Se estudiarán técnicas como:

- Integrated Gradients;
- saliency maps;
- occlusion;
- feature ablation;
- ocultación por derivación;
- ocultación por ventanas temporales.

Objetivos:

- identificar qué derivaciones influyen más;
- localizar los segmentos temporales relevantes;
- comparar explicaciones entre aciertos y errores;
- detectar dependencia de artefactos;
- estudiar estabilidad de explicaciones;
- analizar falsos positivos y falsos negativos.

Se utilizará principalmente **Captum** al trabajar con PyTorch.

---

## 11. Calibración e incertidumbre

El sistema deberá evitar interpretar directamente la salida sigmoid como una probabilidad clínica fiable.

Se estudiarán:

- curvas de calibración;
- Brier Score;
- Expected Calibration Error;
- temperature scaling u otros métodos adecuados;
- umbrales específicos por clase;
- abstención ante baja confianza;
- cobertura frente a rendimiento.

La calibración se realizará usando validación, nunca test.

---

## 12. Entorno de desarrollo

### Entorno principal

El proyecto se desarrollará principalmente en:

- Windows;
- WSL2;
- Ubuntu;
- Visual Studio Code.

VS Code será el centro de trabajo para:

- código;
- terminal;
- notebooks;
- Git;
- tests;
- debugging;
- Docker;
- Codex;
- revisión de cambios.

### Google Colab

Google Colab gratuito se utilizará únicamente como recurso de cómputo adicional para entrenamientos con GPU.

Colab no será el entorno principal del proyecto.

El notebook de Colab deberá limitarse a:

1. clonar el repositorio;
2. instalar dependencias;
3. acceder a los datos;
4. ejecutar los mismos scripts del repositorio;
5. guardar o descargar artefactos.

No se mantendrá una implementación separada del entrenamiento dentro del notebook.

---

## 13. Herramientas

### Desarrollo y sistema

- Visual Studio Code.
- WSL2.
- Ubuntu.
- Git.
- GitHub.
- Codex para VS Code.
- terminal Linux.
- Docker Desktop.

### Lenguaje y entorno

- Python 3.11.
- `uv`.
- `pyproject.toml`.
- `uv.lock`.

### Machine Learning

- PyTorch.
- NumPy.
- Pandas.
- SciPy.
- Scikit-learn.
- WFDB.
- Matplotlib.
- Captum.

### Notebooks

- Jupyter.
- Extensión Jupyter de VS Code.
- Google Colab.

### Configuración

- YAML.
- PyYAML.

Hydra solo se estudiará si las configuraciones se vuelven difíciles de gestionar.

### Seguimiento de experimentos

- MLflow Tracking.
- MLflow Model Registry.
- SQLite como backend local.
- almacenamiento local de artefactos.

### Optimización de hiperparámetros

- Optuna, únicamente en una fase posterior;
- búsquedas pequeñas y justificadas;
- integración con MLflow.

### Calidad de código

- Pytest.
- Ruff.
- Pyright o Mypy.
- pre-commit.
- type hints.

### Integración continua

- GitHub Actions.

### API

- FastAPI.
- Pydantic.
- Uvicorn.

### Empaquetado

- Docker.
- Docker Compose cuando exista más de un servicio.

### Demo y publicación

- Hugging Face Hub.
- Hugging Face Spaces.
- Gradio.

### Monitorización

Primera versión:

- logging estructurado;
- métricas de latencia;
- errores;
- distribución de predicciones;
- confianza;
- tasa de abstención;
- estadísticas de entrada;
- almacenamiento local o SQLite.

Posible ampliación:

- Evidently.

Prometheus y Grafana se considerarán únicamente si el sistema básico está terminado y su inclusión aporta aprendizaje real.

---

## 14. Herramientas que no se introducirán inicialmente

No se utilizarán por defecto:

- AWS;
- Azure;
- Google Cloud de pago;
- Kubernetes;
- Kubeflow;
- Terraform;
- Airflow;
- Prefect;
- Spark;
- W&B;
- ClearML;
- DVC desde el primer día;
- Prometheus;
- Grafana.

Estas herramientas solo se incorporarían si resuelven un problema concreto del proyecto.

---

## 15. Arquitectura lógica

El sistema tendrá varias capas:

```text
PTB-XL
   ↓
Validación y carga
   ↓
Construcción de etiquetas
   ↓
Splits por paciente
   ↓
Preprocesamiento
   ↓
Dataset y DataLoader
   ↓
Entrenamiento
   ↓
Evaluación
   ↓
MLflow
   ↓
Selección y registro del modelo
   ↓
Pipeline de inferencia
   ↓
FastAPI
   ↓
Docker
   ↓
Demo en Hugging Face Spaces
   ↓
Monitorización
```

La lógica de entrenamiento, evaluación e inferencia deberá compartir componentes para evitar inconsistencias.

---

## 16. Estructura prevista del repositorio

```text
ptbxl-ml-system/
├── .github/
│   └── workflows/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── scripts/
├── src/
│   └── ptbxl/
│       ├── data/
│       ├── preprocessing/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── interpretability/
│       ├── inference/
│       ├── monitoring/
│       └── api/
├── tests/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── evaluation/
│   ├── inference/
│   └── api/
├── artifacts/
├── outputs/
├── .gitignore
├── .pre-commit-config.yaml
├── AGENTS.md
├── LICENSE
├── README.md
├── pyproject.toml
└── uv.lock
```

### Reglas

- Los notebooks se utilizarán para exploración.
- La lógica definitiva irá en `src/`.
- Los scripts serán puntos de entrada.
- Los datos no se subirán a Git.
- Los checkpoints no se subirán directamente al historial de Git.
- Los experimentos se definirán mediante configuración.
- Cada transformación crítica tendrá tests.

---

## 17. Uso de Git y GitHub

### Repositorio

Se creará un repositorio público llamado provisionalmente:

```text
ptbxl-ml-system
```

### Ramas

- `main`: versión funcional;
- ramas cortas por tarea.

Ejemplos:

```text
data/load-metadata
data/official-splits
model/baseline-cnn
evaluation/multilabel-metrics
api/predict-endpoint
```

### Issues

Cada unidad de trabajo deberá formularse como una issue concreta.

Ejemplos:

- `[DATA] Descargar PTB-XL de forma reproducible`.
- `[DATA] Construir las cinco superclases`.
- `[TEST] Verificar separación de pacientes`.
- `[MODEL] Implementar baseline CNN`.
- `[EVAL] Añadir métricas multietiqueta`.
- `[API] Crear endpoint de predicción`.

### Pull requests

Se utilizarán pull requests para cambios importantes, aunque el proyecto tenga un solo desarrollador.

Cada pull request deberá indicar:

- objetivo;
- cambios realizados;
- decisiones tomadas;
- tests ejecutados;
- riesgos;
- limitaciones.

---

## 18. Uso de MLflow

MLflow se utilizará para registrar:

- nombre del experimento;
- arquitectura;
- hiperparámetros;
- configuración;
- semilla;
- commit de Git;
- versión del dataset;
- preprocesamiento;
- métricas de entrenamiento;
- métricas de validación;
- métricas por clase;
- curvas;
- figuras;
- umbrales;
- calibración;
- checkpoints;
- modelo final.

Configuración inicial:

- servidor local;
- SQLite;
- artefactos en disco;
- interfaz web local.

El Model Registry se utilizará cuando existan modelos candidatos reales.

La API final deberá cargar una versión aprobada del modelo, no un archivo elegido manualmente sin trazabilidad.

---

## 19. Testing

Se utilizará Pytest.

### Tests de datos

- parsing de `scp_codes`;
- construcción de etiquetas;
- forma de las señales;
- orden de derivaciones;
- ausencia de NaN;
- ausencia de pacientes compartidos;
- correspondencia entre registros y metadatos.

### Tests de preprocesamiento

- conservación de forma;
- tipo `float32`;
- comportamiento determinista;
- manejo de entradas inválidas;
- ausencia de leakage.

### Tests de modelos

- forward pass;
- forma de salida;
- gradientes;
- guardado y carga;
- inferencia en CPU.

### Tests de evaluación

- métricas conocidas sobre casos sintéticos;
- selección de umbrales;
- calibración;
- tratamiento de clases sin positivos.

### Tests de API

- endpoint de salud;
- entrada válida;
- entrada inválida;
- número incorrecto de derivaciones;
- longitud incorrecta;
- modelo no disponible;
- formato de salida.

### Tests de integración

- señal de entrada;
- preprocesamiento;
- modelo;
- postprocesamiento;
- respuesta final.

La CI no entrenará el modelo completo.

---

## 20. Integración continua

GitHub Actions ejecutará:

1. instalación con `uv`;
2. linting con Ruff;
3. comprobación de formato;
4. type checking;
5. Pytest;
6. construcción de la imagen Docker en fases posteriores.

Se usarán datos sintéticos o fixtures pequeños.

La CI deberá ser rápida y no depender del dataset completo.

---

## 21. Despliegue

### API

FastAPI expondrá inicialmente:

- `GET /health`;
- `GET /model-info`;
- `POST /predict`.

Posibles endpoints posteriores:

- `POST /explain`;
- `GET /metrics`.

La API deberá:

- validar las 12 derivaciones;
- validar la longitud;
- validar tipos;
- aplicar el mismo preprocesamiento que entrenamiento;
- devolver probabilidades;
- devolver etiquetas;
- devolver umbrales;
- indicar posibles abstenciones;
- registrar latencia y errores.

### Docker

La API se empaquetará en una imagen Docker reproducible.

Objetivos:

- ejecutar en local;
- ejecutar en cualquier máquina compatible;
- reutilizar el mismo entorno en CI;
- facilitar despliegue.

### Demo

La demo se publicará gratuitamente usando:

- Hugging Face Spaces;
- Gradio;
- Hugging Face Hub para el modelo.

La demo mostrará claramente que el sistema es experimental y no está destinado a uso clínico.

---

## 22. Monitorización

Se simulará un entorno productivo.

### Métricas técnicas

- latencia;
- errores;
- tiempo de carga;
- número de peticiones;
- tamaño de entradas;
- uso de memoria si es viable.

### Métricas del modelo

- distribución de probabilidades;
- clases predichas;
- tasa de abstención;
- amplitud de señal;
- porcentaje de señales fuera de rango;
- calidad de entrada;
- drift respecto al entrenamiento.

Como no habrá etiquetas reales en tiempo real, la monitorización de rendimiento se simulará cuando sea necesario.

---

## 23. Roadmap de 8–12 semanas

### Fase 0 — Entorno y repositorio

Objetivos:

- instalar herramientas;
- configurar WSL2;
- instalar Python y `uv`;
- crear repositorio;
- crear estructura;
- configurar Git;
- conectar GitHub;
- crear `pyproject.toml`;
- configurar Ruff y Pytest;
- crear CI mínima;
- redactar README y AGENTS.

Criterio de finalización:

- proyecto instalable;
- tests vacíos o iniciales funcionando;
- lint funcionando;
- repositorio conectado a GitHub;
- CI en verde.

### Fase 1 — Dataset confiable

Objetivos:

- descargar PTB-XL;
- estudiar metadatos;
- cargar señales;
- construir etiquetas;
- implementar splits;
- validar pacientes;
- hacer EDA.

Criterio de finalización:

- carga reproducible;
- etiquetas correctas;
- forma `12 × 1000`;
- splits sin leakage;
- tests de datos;
- prevalencias documentadas.

### Fase 2 — Baselines

Objetivos:

- baseline clásico;
- CNN 1D sencilla;
- pipeline de entrenamiento;
- métricas;
- configuración;
- checkpoints.

Criterio de finalización:

- entrenamientos reproducibles;
- baseline profundo supera al trivial;
- métricas por clase;
- resultados guardados;
- tests básicos.

### Fase 3 — MLflow y experimentación

Objetivos:

- integrar MLflow;
- registrar configuraciones;
- registrar métricas;
- registrar artefactos;
- comparar experimentos.

Criterio de finalización:

- todos los entrenamientos generan runs;
- se puede reproducir una ejecución;
- cada run incluye commit y configuración.

### Fase 4 — Modelo principal

Objetivos:

- implementar ResNet 1D;
- estudiar preprocesamiento;
- estudiar pérdidas;
- ajustar hiperparámetros;
- analizar errores.

Criterio de finalización:

- modelo candidato seleccionado;
- comparación documentada;
- test todavía sin utilizar para decisiones.

### Fase 5 — Calibración e interpretabilidad

Objetivos:

- ajustar umbrales;
- calibrar;
- Integrated Gradients;
- occlusion;
- análisis por derivación;
- análisis de errores.

Criterio de finalización:

- probabilidades calibradas;
- visualizaciones reproducibles;
- limitaciones identificadas;
- explicación coherente en casos seleccionados.

### Fase 6 — Inferencia y API

Objetivos:

- crear pipeline de inferencia;
- validar entradas;
- FastAPI;
- tests de API;
- modelo registrado.

Criterio de finalización:

- predicción reproducible;
- API funcional;
- tests en verde;
- documentación de endpoints.

### Fase 7 — Docker y CI/CD

Objetivos:

- Dockerfile;
- build reproducible;
- GitHub Actions;
- pruebas de imagen;
- versionado de releases.

Criterio de finalización:

- API ejecutable con Docker;
- CI valida código y build;
- release candidata disponible.

### Fase 8 — Monitorización y demo

Objetivos:

- logging;
- métricas;
- simulación de drift;
- Gradio;
- Hugging Face Spaces;
- model card.

Criterio de finalización:

- demo pública;
- monitorización básica;
- documentación de limitaciones;
- proyecto reproducible.

### Fase 9 — Cierre

Objetivos:

- evaluación final en test;
- análisis de errores;
- README final;
- diagrama de arquitectura;
- resultados;
- CV;
- LinkedIn;
- preparación de entrevista.

Criterio de finalización:

- test utilizado una única vez para cierre;
- resultados finales documentados;
- repositorio entendible por terceros;
- sistema desplegado;
- decisiones defendibles.

---

## 24. Riesgos

### Riesgo: exceso de alcance

Recorte:

- mantener solo CNN y ResNet;
- no añadir Transformers;
- no añadir Prometheus/Grafana;
- no añadir DVC;
- mantener monitorización simple.

### Riesgo: Colab inestable

Recorte:

- entrenamientos pequeños;
- checkpoints frecuentes;
- trabajo a 100 Hz;
- modelos ligeros;
- ejecución local para depuración.

### Riesgo: resultados modestos

Respuesta:

- comparar con baselines;
- analizar métricas por clase;
- priorizar calidad metodológica;
- estudiar calibración;
- documentar limitaciones.

### Riesgo: leakage

Prevención:

- folds oficiales;
- tests por paciente;
- transformaciones ajustadas solo con train;
- calibración y umbrales solo con validación;
- test aislado.

### Riesgo: dependencia excesiva de Codex

Prevención:

- tareas pequeñas;
- revisar diffs;
- exigir explicación;
- no aceptar código no comprendido;
- escribir personalmente decisiones críticas;
- mantener documentación de decisiones.

---

## 25. Uso de ChatGPT y Codex

### ChatGPT

Se utilizará como:

- mentor técnico;
- profesor de ECG;
- apoyo teórico;
- revisor metodológico;
- arquitecto;
- revisor de experimentos;
- detector de leakage;
- preparador de entrevistas;
- apoyo para documentación.

### Codex

Se utilizará como:

- asistente dentro del repositorio;
- implementador de tareas pequeñas;
- generador de tests;
- revisor de código;
- ayuda para refactorizar;
- apoyo para configuración;
- ayuda para Docker;
- ayuda para GitHub Actions.

### Regla principal

No se aceptará código que el desarrollador no pueda explicar respondiendo:

- qué hace;
- por qué existe;
- qué entradas recibe;
- qué devuelve;
- cómo puede fallar;
- cómo se prueba;
- qué alternativa más simple existe.

---

## 26. Entregable final

El entregable final será un repositorio que incluya:

- código fuente;
- scripts reproducibles;
- configuraciones;
- tests;
- CI;
- Docker;
- API;
- MLflow;
- modelo registrado;
- inferencia;
- interpretabilidad;
- calibración;
- monitorización;
- demo;
- model card;
- documentación técnica;
- resultados finales;
- limitaciones;
- roadmap futuro.

El proyecto deberá poder explicarse en una entrevista como un caso completo de investigación y Machine Learning Engineering aplicado a salud.

---

## 27. Mensaje profesional del proyecto

Una posible descripción final sería:

> Diseñé e implementé un sistema end-to-end para clasificación multietiqueta de ECG de 12 derivaciones con PTB-XL. Construí un pipeline reproducible de datos, entrené y comparé modelos 1D, implementé calibración, umbrales por clase e interpretabilidad, registré experimentos y modelos con MLflow, añadí tests y CI con GitHub Actions, empaqueté la inferencia con FastAPI y Docker y desplegué una demo monitorizada utilizando servicios gratuitos.

---

## 28. Principios del proyecto

1. El aprendizaje importa más que añadir herramientas.
2. La reproducibilidad importa más que una métrica aislada.
3. El test no se utiliza para tomar decisiones.
4. Las particiones se realizan por paciente.
5. Los notebooks no contienen la lógica definitiva.
6. Toda transformación crítica debe poder probarse.
7. Cada dependencia debe estar justificada.
8. La solución más simple válida tiene prioridad.
9. Todo código generado debe comprenderse.
10. El proyecto debe poder explicarse en una entrevista.
