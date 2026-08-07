

# Reconocimiento de Entidades Nombradas Anidadas como Etiquetado de Secuencia en un Solo Paso

[![CI](https://github.com/amunozo/nner_as_sl/actions/workflows/ci.yml/badge.svg)](https://github.com/amunozo/nner_as_sl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/ACL-Anthology-red.svg)](https://aclanthology.org/2025.findings-emnlp.530/)

Código de investigación para **[Nested Named Entity Recognition as Single-Pass Sequence Labeling](https://aclanthology.org/2025.findings-emnlp.530/)**, de Alberto Muñoz-Ortiz, David Vilares, Caio Corro y Carlos Gómez-Rodríguez, publicado en Findings of EMNLP 2025.

El proyecto representa las entidades anidadas como árboles similares a los de sintaxis constitucional, lineariza dichos árboles con CoDeLin y entrena un etiquetador de secuencias multitarea con MaChAmp. Incluye las codificaciones ABS, REL, JUX, DYN y 4EC utilizadas en los experimentos. Se trata de un artefacto de investigación, no de un paquete de NER de propósito general.

## Estructura del repositorio

- `src/data/`: Análisis de datos NNER y conversión de árboles/etiquetas.
- `src/evaluation/`: métricas de coincidencia exacta generales y por etiqueta, longitud de span y profundidad de anidación.
- `src/machamp/`: generación de configuraciones de MaChAmp por semilla.
- `scripts/train.py`: codificación de datos y entrenamiento con múltiples semillas.
- `scripts/evaluate.py`: predicción, decodificación, medición de tiempo y agregación de métricas.
- `scripts/entities_per_depth.py`: estadísticas del conjunto de datos.
- `scripts/label_coverage.py`: cobertura de ida y vuelta de codificación/decodificación.
- `parameter_configs/`: plantilla de parámetros de MaChAmp.
- `CoDeLin/` y `machamp/`: submódulos de Git fijos.
- `tests/`: pruebas rápidas de conversión, métricas y construcción de comandos.

## Instalación

Clona los submódulos y crea un entorno Python aislado:

```bash
git clone --recurse-submodules https://github.com/amunozo/nner_as_sl.git
cd nner_as_sl
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si el repositorio se clonó sin `--recurse-submodules`, ejecuta:

```bash
git submodule update --init --recursive
```

La revisión de MaChAmp es la originalmente utilizada por este proyecto. El objeto CoDeLin original registrado por el repositorio ya no se sirve desde su remoto original; por lo tanto, el submódulo está fijado a una revisión disponible que contiene las correcciones para NER anidado y 4EC, y expone la misma interfaz de línea de comandos utilizada aquí.

## Formato de los datos

Los conjuntos de datos no se redistribuyen. Coloca cada conjunto en `data/<dataset>/` con `train.data`, `dev.data` y `test.data`. Cada ejemplo contiene una oración tokenizada seguida de cero o más entidades separadas por el carácter `|`. Los desplazamientos finales son inclusivos.

```text
IL-2 gene expression and NF-kappa B activation through CD28 requires reactive oxygen production .
0,1 G#DNA|4,5 G#protein|8,8 G#protein
```

Los spans anidados deben estar correctamente anidados. Los spans cruzados no pueden representarse mediante la conversión a árbol de sintaxis constitucional y se rechazan con un error explícito.

## Entrenamiento

El comando de entrenamiento crea archivos de árbol y etiquetas cuando sea necesario, escribe una configuración por semilla e invoca el punto de entrada de MaChAmp fijo sin usar una shell:

```bash
python scripts/train.py \
  --dataset genia \
  --encoder bert-base-uncased \
  --encoding REL \
  --n-seeds 3 \
  --num-epochs 30 \
  --device 0 \
  --time
```

Las semillas completadas con un `model.pt` se omiten. Si un directorio de semilla contiene solo salida incompleta, MaChAmp se lanza nuevamente en ese directorio y la situación se informa de manera explícita. Usa `--force-encode` para regenerar los archivos de etiquetas después de cambiar los datos de origen.

Todas las rutas raíz importantes pueden anularse mediante `--data-dir`, `--logs-dir`, `--template-dir`, `--machamp-dir` y `--codelin-dir`. Ejecuta `--help` para ver la interfaz completa.

## Evaluación

```bash
python scripts/evaluate.py \
  --dataset genia \
  --encoder bert-base-uncased \
  --encoding REL \
  --device 0
```

Usa `--no-predict` para decodificar y evaluar un archivo `output.labels` existente, o repite `--seed` para seleccionar semillas específicas. Los archivos `results.json` por semilla y un archivo `avg_results.json` se almacenan junto a los modelos.

Las métricas utilizan coincidencias exactas sobre tuplas `(etiqueta, inicio, fin_inclusivo)`:

- los grupos por longitud de span usan `end - start + 1`;
- los informes por etiqueta y longitud incluyen grupos de solo predicción, por lo que los falsos positivos no quedan ocultos;
- los archivos con diferentes cantidades de oraciones fallan en lugar de truncarse silenciosamente;
- el recall por profundidad agrupa las predicciones correctas según la profundidad de referencia (gold) de la entidad, mientras que la precisión por profundidad las agrupa según la profundidad predicha. Se informan conteos correctos por separado porque una entidad recuperada correctamente puede cambiar de profundidad si se pasa por alto una entidad circundante.

Los diagnósticos del conjunto de datos y la codificación están disponibles mediante CLIs explícitos:

```bash
python scripts/entities_per_depth.py genia ace2005
python scripts/label_coverage.py genia --encodings ABS REL DYN 4EC
```

## Verificaciones de desarrollo

```bash
python -m pip install -r requirements-dev.txt nltk
python -m ruff check .
python -m pytest
```

La CI valida el código determinista de conversión y evaluación en Python 3.10 y 3.12. No ejecuta el entrenamiento de transformadores, ya que esto requiere los conjuntos de datos, los pesos de modelo descargados y capacidad de cómputo adecuada.

## Citación

```bibtex
@inproceedings{munoz-ortiz-etal-2025-nested,
    title = {Nested Named Entity Recognition as Single-Pass Sequence Labeling},
    author = {Mu\~noz-Ortiz, Alberto and Vilares, David and Corro, Caio and G\'omez-Rodr\'iguez, Carlos},
    booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
    year = {2025},
    address = {Suzhou, China},
    publisher = {Association for Computational Linguistics},
    doi = {10.18653/v1/2025.findings-emnlp.530},
    pages = {9993--10002},
}
```

## Contacto

Alberto Muñoz-Ortiz: [alberto.munoz.ortiz@udc.es](mailto:alberto.munoz.ortiz@udc.es)

## Agradecimientos

Este trabajo recibió apoyo de SCANNER-UDC (PID2020-113230RB-C21), Xunta de Galicia (ED431C 2024/02), GAP (PID2022-139308OA-I00), PRE2021-097001, LATCHING (PID2023-147129OB-C21), TSI-100925-2023-1, CITIC, el Centro de Supercomputación de Galicia y los proyectos que apoyan a Caio Corro. Consulta el artículo para la declaración completa de financiamiento.
