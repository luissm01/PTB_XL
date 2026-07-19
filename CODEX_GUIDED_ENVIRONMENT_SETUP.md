# Guía interactiva para crear el entorno del proyecto PTB-XL con Codex

## Propósito de este documento

Este documento debe utilizarse como guía de trabajo dentro de un chat de Codex en Visual Studio Code.

El objetivo no es que Codex cree automáticamente todo el proyecto de una sola vez. Debe actuar como acompañante técnico mientras el desarrollador:

- entiende cada herramienta;
- ejecuta personalmente los comandos importantes;
- revisa cada archivo creado;
- toma las decisiones;
- aprende Git, GitHub, `uv`, Python y la estructura del repositorio;
- puede explicar después todo el proceso en una entrevista.

Codex debe avanzar únicamente una etapa cada vez y esperar confirmación antes de pasar a la siguiente.

---

# Instrucciones generales para Codex

Actúa como mentor técnico y compañero de programación para configurar un proyecto de Machine Learning Engineering basado en PTB-XL.

No ejecutes toda la configuración de una vez.

En cada etapa:

1. Explica brevemente el objetivo.
2. Explica por qué se hace.
3. Indica qué debe ejecutar o crear el desarrollador personalmente.
4. Proporciona solamente los comandos necesarios para esa etapa.
5. Espera a que el desarrollador comparta el resultado.
6. Revisa el resultado antes de continuar.
7. Señala cualquier error o diferencia relevante.
8. No avances hasta que el desarrollador confirme que la etapa funciona.

Antes de modificar archivos:

- inspecciona el estado actual del repositorio;
- explica qué archivo se va a modificar;
- muestra el contenido propuesto;
- pide al desarrollador que lo revise;
- realiza cambios pequeños;
- enseña el diff después del cambio.

No instales dependencias ni herramientas sin explicar:

- qué problema resuelven;
- por qué se necesitan ahora;
- qué alternativa más simple existe;
- si son dependencias de producción o desarrollo.

No generes todavía:

- código de descarga de PTB-XL;
- modelos;
- pipelines de entrenamiento;
- Docker;
- MLflow;
- FastAPI;
- GitHub Actions complejas;
- estructura final completa.

Primero se debe terminar una configuración mínima, limpia y comprobada.

---

# Contexto del proyecto

El proyecto será un sistema completo de Machine Learning Engineering aplicado a señales ECG usando PTB-XL.

Objetivos futuros:

- procesamiento de ECG de 12 derivaciones;
- clasificación multietiqueta de cinco superclases;
- modelos CNN 1D y ResNet 1D;
- seguimiento de experimentos con MLflow;
- interpretabilidad con Captum;
- API con FastAPI;
- tests;
- integración continua;
- Docker;
- monitorización;
- demo gratuita.

La primera etapa se limita a preparar correctamente:

- carpeta del proyecto;
- entorno Linux;
- Git;
- GitHub;
- Python;
- `uv`;
- paquete Python mínimo;
- calidad de código mínima;
- primer commit.

Nombre provisional del repositorio:

```text
ptbxl-ml-system
```

---

# Resultado esperado de esta sesión

Al finalizar esta guía deben existir:

```text
ptbxl-ml-system/
├── src/
│   └── ptbxl/
│       └── __init__.py
├── tests/
│   └── test_import.py
├── .gitignore
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
└── .python-version
```

Además:

- el entorno virtual debe estar creado;
- el paquete debe poder importarse;
- `pytest` debe funcionar;
- `ruff` debe funcionar;
- Git debe estar inicializado;
- el repositorio remoto de GitHub debe estar conectado;
- debe existir un primer commit;
- la rama principal debe estar publicada en GitHub.

No se deben crear aún carpetas vacías para todas las fases futuras salvo que exista una razón práctica.

---

# Etapa 0 — Comprobar el entorno actual

## Objetivo

Conocer el estado real del sistema antes de instalar o modificar nada.

## Codex debe pedir al desarrollador que ejecute

Desde una terminal WSL:

```bash
pwd
ls -la
uname -a
python3 --version
git --version
uv --version
code --version
```

Algunos comandos pueden fallar si la herramienta no está instalada. Eso no es un problema.

## Codex debe revisar

- que la terminal sea WSL/Linux;
- versión de Ubuntu;
- versión de Python;
- disponibilidad de Git;
- disponibilidad de `uv`;
- disponibilidad del comando `code`;
- carpeta desde la que se está trabajando.

## Preguntas que Codex debe plantear

- ¿El proyecto se guardará dentro del sistema de archivos de WSL?
- ¿Está configurado el nombre y correo de Git?
- ¿Existe ya un repositorio de GitHub vacío?
- ¿Está instalada GitHub CLI (`gh`) o se usará la web?

## Criterio de finalización

El desarrollador y Codex conocen el estado del sistema y han decidido qué herramientas falta instalar.

Codex debe detenerse aquí hasta recibir los resultados.

---

# Etapa 1 — Elegir la ubicación del proyecto

## Objetivo

Crear el proyecto dentro del sistema de archivos Linux, evitando problemas de rendimiento y permisos.

## Recomendación

Trabajar en una ruta similar a:

```text
~/projects/ptbxl-ml-system
```

No se recomienda trabajar directamente en:

```text
/mnt/c/...
```

porque el acceso desde WSL puede ser más lento y puede producir diferencias de permisos.

## El desarrollador debe ejecutar personalmente

```bash
mkdir -p ~/projects
cd ~/projects
mkdir ptbxl-ml-system
cd ptbxl-ml-system
pwd
```

## Codex debe explicar

- qué representa `~`;
- qué hace `mkdir -p`;
- por qué se usa `cd`;
- por qué se verifica con `pwd`.

## Comprobación

```bash
ls -la
```

La carpeta debe estar vacía.

## Criterio de finalización

La carpeta existe, está dentro de WSL y se abre correctamente con:

```bash
code .
```

Codex debe esperar confirmación.

---

# Etapa 2 — Inicializar Git

## Objetivo

Empezar a versionar el proyecto desde su creación.

## Antes de ejecutar

Codex debe explicar:

- qué es un repositorio Git;
- diferencia entre Git y GitHub;
- qué representa `.git`;
- qué es la rama principal;
- qué significa un commit.

## El desarrollador debe comprobar su identidad

```bash
git config --global user.name
git config --global user.email
```

Si no existe, Codex debe ayudar a configurarla, pero el desarrollador debe elegir los valores.

## Inicialización

```bash
git init -b main
git status
```

## Comprobación

```bash
ls -la
```

Debe aparecer `.git`.

## Criterio de finalización

- Git está inicializado.
- La rama actual es `main`.
- `git status` funciona.
- El desarrollador comprende que todavía no existe un repositorio remoto.

Codex debe detenerse.

---

# Etapa 3 — Instalar o verificar `uv`

## Objetivo

Utilizar `uv` para gestionar:

- versión de Python;
- entorno virtual;
- dependencias;
- lockfile;
- ejecución de herramientas.

## Codex debe explicar

- diferencia entre Python del sistema y entorno virtual;
- qué es `pyproject.toml`;
- qué es `uv.lock`;
- por qué se bloquean versiones;
- diferencia entre dependencia de producción y desarrollo.

## Comprobación

```bash
uv --version
```

Si `uv` no está instalado, Codex debe proporcionar la instalación oficial adecuada al entorno y pedir al desarrollador que cierre o recargue la terminal si es necesario.

Después:

```bash
uv --version
```

## Criterio de finalización

`uv` responde correctamente.

Codex no debe crear todavía el proyecto Python hasta que esto se confirme.

---

# Etapa 4 — Inicializar el proyecto Python

## Objetivo

Crear una base mínima de paquete Python usando `uv`.

## Antes de ejecutar

Codex debe explicar las opciones disponibles:

- aplicación;
- paquete;
- script;
- librería.

Para este proyecto se recomienda un paquete con estructura `src/`, porque:

- evita imports accidentales desde la raíz;
- se parece a proyectos profesionales;
- facilita tests e instalación;
- separa código fuente del resto del repositorio.

## Codex debe proponer el comando adecuado

Por ejemplo:

```bash
uv init --package --python 3.11
```

Antes de ejecutarlo, Codex debe revisar en la versión instalada de `uv` si las opciones son válidas.

El desarrollador debe ejecutar el comando.

## Después

```bash
find . -maxdepth 3 -type f | sort
```

Codex debe revisar los archivos creados.

## Codex no debe aceptar automáticamente todo lo generado

Debe explicar:

- `pyproject.toml`;
- `.python-version`;
- `README.md`;
- estructura `src`;
- archivo `__init__.py`.

Si `uv` crea archivos de ejemplo innecesarios, Codex debe proponer eliminarlos de uno en uno.

## Criterio de finalización

Existe un paquete Python mínimo y el desarrollador entiende cada archivo.

Codex debe detenerse.

---

# Etapa 5 — Crear el entorno virtual

## Objetivo

Crear un entorno aislado para el proyecto.

## El desarrollador debe ejecutar

```bash
uv sync
```

Después:

```bash
ls -la
```

Debe aparecer `.venv`.

## Codex debe explicar

- qué contiene `.venv`;
- por qué no se sube a Git;
- cómo `uv run` evita activar manualmente el entorno;
- diferencia entre:

```bash
python
uv run python
```

## Prueba

```bash
uv run python --version
uv run python -c "import sys; print(sys.executable)"
```

## Criterio de finalización

Python se ejecuta desde `.venv` y usa la versión esperada.

Codex debe esperar.

---

# Etapa 6 — Revisar y simplificar `pyproject.toml`

## Objetivo

Comprender y dejar una configuración mínima.

## Codex debe abrir primero el archivo

Debe pedir al desarrollador que lo lea o mostrarlo antes de modificarlo.

## Configuración mínima esperada

Debe contener, como mínimo:

- nombre del proyecto;
- versión;
- descripción;
- archivo README;
- versión mínima de Python;
- dependencias de producción;
- configuración del sistema de build.

No se deben añadir todavía librerías de Machine Learning.

## Dependencias iniciales

Las dependencias de producción pueden estar vacías.

Dependencias de desarrollo iniciales:

- `pytest`;
- `ruff`.

Codex debe añadirlas usando comandos `uv`, no editando manualmente el lockfile.

Ejemplo:

```bash
uv add --dev pytest ruff
```

## Codex debe explicar

- por qué `pytest` y `ruff` son dependencias de desarrollo;
- cómo cambia `pyproject.toml`;
- cómo cambia `uv.lock`;
- por qué nunca se edita `uv.lock` manualmente.

## Comprobación

```bash
git diff
```

## Criterio de finalización

El desarrollador entiende las secciones principales y las dependencias están instaladas.

---

# Etapa 7 — Crear un test mínimo

## Objetivo

Verificar que el paquete se instala e importa correctamente.

## Archivo

```text
tests/test_import.py
```

## Codex debe pedir primero al desarrollador que intente escribirlo

Comportamiento esperado:

- importar el paquete `ptbxl`;
- verificar que el import no falla.

Codex puede dar una pista, pero no debe crear el archivo directamente salvo que el desarrollador lo pida.

## Ejecución

```bash
uv run pytest
```

## Codex debe explicar

- cómo descubre Pytest los tests;
- por qué los archivos comienzan con `test_`;
- qué significa una prueba pasada;
- por qué este test es pequeño pero útil.

## Criterio de finalización

`pytest` termina correctamente.

---

# Etapa 8 — Configurar Ruff

## Objetivo

Añadir comprobación básica de formato y calidad.

## Codex debe explicar

- qué es linting;
- diferencia entre linting y testing;
- diferencia entre comprobar y modificar;
- por qué se comienza con una configuración pequeña.

## Comandos iniciales

```bash
uv run ruff check .
uv run ruff format --check .
```

Si existen errores, Codex debe explicarlos antes de corregirlos.

## Configuración

Solo si es necesario, se añadirá una sección mínima a `pyproject.toml`.

No se deben activar decenas de reglas de golpe.

## Criterio de finalización

Ambos comandos terminan correctamente.

---

# Etapa 9 — Crear `.gitignore`

## Objetivo

Evitar versionar archivos generados, privados o pesados.

## Codex debe pedir al desarrollador que proponga primero qué ignorar

Elementos iniciales:

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/
.vscode/
.env
```

Elementos previstos para el proyecto:

```gitignore
data/raw/
data/interim/
data/processed/
artifacts/
outputs/
mlruns/
mlflow.db
*.pt
*.pth
*.onnx
```

Codex debe explicar que ignorar una carpeta no la elimina y que `.gitignore` no afecta automáticamente a archivos que ya estén versionados.

## Comprobación

```bash
git status
```

`.venv` no debe aparecer.

## Criterio de finalización

Solo aparecen archivos que realmente deben versionarse.

---

# Etapa 10 — Crear README mínimo

## Objetivo

Documentar el estado actual sin fingir que el proyecto ya está terminado.

## Contenido inicial

- nombre;
- objetivo breve;
- estado del proyecto;
- requisitos;
- instalación;
- ejecución de tests;
- linting;
- aviso de que está en desarrollo;
- indicación de que no es un sistema clínico.

Codex debe pedir al desarrollador que redacte primero un borrador corto.

Después puede revisarlo y proponer mejoras.

No debe generar todavía:

- resultados inventados;
- badges inexistentes;
- arquitectura futura detallada;
- instrucciones que aún no funcionan.

## Criterio de finalización

Una persona puede clonar el repositorio, instalarlo y ejecutar tests siguiendo el README.

---

# Etapa 11 — Crear `AGENTS.md`

## Objetivo

Dar instrucciones permanentes a Codex dentro del repositorio.

## Contenido recomendado

```markdown
# Project purpose

Production-oriented ECG classification system using PTB-XL.

# Development principles

- Prefer simple, testable implementations.
- Prevent patient-level data leakage.
- Use the official PTB-XL folds.
- Keep notebooks for exploration only.
- Keep reusable logic under `src/`.
- Do not add dependencies without justification.
- Do not modify generated lock files manually.
- Important data transformations require tests.
- Do not use the final test fold for model selection.
- Avoid unnecessary abstractions.

# Workflow

Before changing code:

1. Inspect relevant files.
2. Explain the proposed change.
3. Identify risks.
4. Implement the smallest useful change.
5. Run the relevant checks.
6. Show and explain the diff.

# Commands

- Install: `uv sync`
- Tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Format check: `uv run ruff format --check .`
```

Codex debe revisar este contenido con el desarrollador antes de crearlo.

## Criterio de finalización

Las instrucciones reflejan la forma de trabajo acordada.

---

# Etapa 12 — Primer commit local

## Objetivo

Crear un punto estable y comprensible.

## Antes del commit

Ejecutar:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
git status
git diff
```

Codex debe revisar el diff con el desarrollador.

## Añadir archivos

El desarrollador debe ejecutar:

```bash
git add .
git status
```

Codex debe explicar la diferencia entre:

- working tree;
- staging area;
- commit.

## Commit

Ejemplo:

```bash
git commit -m "chore: initialize Python project with uv"
```

Codex debe explicar por qué el mensaje usa `chore`.

## Criterio de finalización

`git status` indica que el árbol está limpio.

---

# Etapa 13 — Crear el repositorio en GitHub

## Objetivo

Crear el repositorio remoto sin sobrescribir el trabajo local.

## Opción A — GitHub web

El desarrollador crea un repositorio vacío:

```text
ptbxl-ml-system
```

Debe evitar seleccionar opciones que creen archivos remotos si ya existen localmente:

- no README automático;
- no `.gitignore` automático;
- no licencia automática, salvo que se haya decidido antes.

## Opción B — GitHub CLI

Si `gh` está instalado y autenticado:

```bash
gh auth status
```

Después Codex puede proponer un comando de creación, pero debe mostrarlo y explicarlo antes de que el desarrollador lo ejecute.

## Codex debe explicar

- repositorio local;
- repositorio remoto;
- `origin`;
- URL HTTPS frente a SSH;
- repositorio público frente a privado.

## Criterio de finalización

El repositorio remoto existe y está vacío.

---

# Etapa 14 — Conectar Git con GitHub

## Objetivo

Relacionar el repositorio local con el remoto.

## Comando

La URL debe obtenerse del repositorio real del desarrollador.

Ejemplo:

```bash
git remote add origin <URL_REAL_DEL_REPOSITORIO>
```

Después:

```bash
git remote -v
```

## Publicar

```bash
git push -u origin main
```

## Codex debe explicar

- qué hace `remote add`;
- qué significa `origin`;
- qué hace `push`;
- qué significa `-u`;
- diferencia entre commit y push.

## Criterio de finalización

- la rama `main` aparece en GitHub;
- los archivos son visibles;
- `git status` está limpio;
- `git remote -v` apunta al repositorio correcto.

---

# Etapa 15 — Comprobación desde cero

## Objetivo

Comprobar que el repositorio es reproducible y no depende accidentalmente del entorno actual.

## Opción recomendada

Crear un clon temporal en otra carpeta:

```bash
cd ~/projects
git clone <URL_REAL_DEL_REPOSITORIO> ptbxl-ml-system-check
cd ptbxl-ml-system-check
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Después se puede borrar el clon temporal.

## Codex debe explicar

Esta comprobación detecta:

- archivos no versionados necesarios;
- instrucciones incompletas;
- dependencias ausentes;
- imports accidentales;
- configuración local no reproducible.

## Criterio de finalización

El proyecto funciona correctamente después de clonarlo desde cero.

---

# Etapa 16 — Cierre de la configuración inicial

## Codex debe pedir al desarrollador que explique

Sin consultar notas:

1. Qué diferencia hay entre Git y GitHub.
2. Qué contiene `.venv`.
3. Para qué sirve `pyproject.toml`.
4. Para qué sirve `uv.lock`.
5. Por qué se usa una estructura `src/`.
6. Diferencia entre dependencia de producción y desarrollo.
7. Diferencia entre `pytest` y Ruff.
8. Diferencia entre commit y push.
9. Qué representa `origin`.
10. Cómo reconstruir el entorno desde cero.

Si alguna respuesta no está clara, Codex debe corregirla antes de cerrar la fase.

---

# Primer mensaje recomendado para Codex

Copiar este mensaje en un chat nuevo de Codex desde la carpeta del proyecto:

```text
Quiero configurar contigo, paso a paso, el entorno inicial de un proyecto de Machine Learning Engineering con Python, uv, Git y GitHub.

Lee el archivo de instrucciones que te voy a proporcionar y sigue estrictamente su flujo interactivo.

No ejecutes ni generes toda la configuración de una vez. Empezaremos únicamente por la Etapa 0.

En cada etapa:
- explica qué vamos a hacer y por qué;
- dime qué comandos debo ejecutar yo;
- espera a que te copie el resultado;
- revisa el resultado;
- no avances sin mi confirmación.

No modifiques archivos sin enseñarme primero el cambio propuesto.
No añadas todavía dependencias de Machine Learning.
No crees aún Docker, MLflow, FastAPI ni GitHub Actions.

Comienza por la Etapa 0: comprobar el entorno actual.
```

---

# Normas de aprendizaje

Durante esta configuración:

- el desarrollador ejecutará personalmente los comandos principales;
- Codex no ocultará comandos detrás de automatizaciones;
- todo error se analizará antes de corregirlo;
- los cambios se revisarán con `git diff`;
- no se copiarán configuraciones sin comprenderlas;
- cada herramienta se añadirá cuando exista una necesidad;
- se priorizará una base pequeña y reproducible.

El objetivo no es terminar rápido. El objetivo es que el desarrollador pueda volver a crear y explicar el entorno sin depender de Codex.
