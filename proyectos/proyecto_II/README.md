# 🏥 Proyecto II — MLOps en Kubernetes: readmisión hospitalaria de pacientes diabéticos (Grupo 5)

Sistema MLOps completo desplegado en un **clúster local de Kubernetes** (K3s dentro de Rancher)
sobre una máquina virtual:

- **Airflow** carga el conjunto *Diabetes 130-US hospitals* en lotes de hasta 15.000 registros.
  Cada lote se guarda **crudo**, luego **limpio**, y con él se **entrenan varios modelos**.
- **MLflow** (con PostgreSQL y MinIO) registra todos los modelos y **promueve automáticamente**
  al mejor con el alias `@champion`.
- **Un API de FastAPI** sirve el modelo productivo leyéndolo desde MLflow, **sin cambios de código
  cuando cambia el modelo**, y registra cada inferencia en la base de datos.
- **Streamlit** es la interfaz de usuario y **Locust** genera la carga.
- **Prometheus** y **Grafana** muestran el comportamiento del servicio.

> 🎥 **Video de sustentación:** _(enlace de YouTube)_

---

## ⭐ Puntos clave

- **Todo corre en Kubernetes:** 13 contenedores propios o de terceros. **Todos** tienen
  `requests`/`limits` y probes. La configuración vive en ConfigMaps, las credenciales en Secrets
  y los datos en PVCs. Las imágenes propias están publicadas en Docker Hub.
- **Carga incremental idempotente:** 7 corridas cargaron los 101.766 registros. Una octava corrida
  detectó que no quedaba nada y **saltó** todo lo demás sin duplicar datos. Los reintentos
  reutilizan el lote ya registrado.
- **Tres capas de datos separadas** en PostgreSQL:
  - `raw`: el texto tal cual llega, con lote, hash y estado por fila.
  - `clean`: las variables tipadas y la partición.
  - `inference`: una fila por cada predicción del API.
- **Objetivo con sentido clínico:** se predice la readmisión en **menos de 30 días** (11,4% de
  positivos).
  - Accuracy sería engañosa: no predecir nunca da 88,7%. Por eso **los modelos se eligen por
    PR-AUC**.
  - El **umbral de decisión maximiza F2** (pesa más no dejar pasar a un paciente que vuelve),
    limitado a marcar como máximo el 30% de los pacientes.
- **Sin fuga de información:** la partición entrenamiento/validación/prueba (70/15/15) se hace
  **por paciente** (hash de `patient_nbr`). Hay pacientes con hasta 40 encuentros.
- **Promoción campeón contra retador:**
  - Cada corrida registra una versión nueva.
  - El alias `@champion` solo se mueve si el retador **supera** al campeón en el **mismo**
    conjunto de prueba.
  - Historial: v1, v2, v3, v4 y v7 fueron promovidas; v5 y v6 fueron rechazadas.
- **El API sigue al campeón por sí solo.** Cada 60 s pregunta a MLflow a qué versión apunta
  `@champion` y descarga el modelo solo si cambió. `POST /reload` fuerza la revisión. El umbral
  viaja con la versión como etiqueta.
- **Las pruebas de carga encontraron un defecto real, que se corrigió y se verificó:** bajo
  saturación, los probes de salud hacían fila detrás de las predicciones y Kubernetes sacaba el
  pod del Service. También se midió el límite operativo:
  - unas **8 req/s por réplica**;
  - **~15 req/s con 3 réplicas**, limitado por las 4 vCPU de la VM.

---

## 📑 Contenido

1. [Cumplimiento del enunciado](#1-cumplimiento-del-enunciado)
2. [Arquitectura](#2-arquitectura)
3. [Datos: análisis y limpieza](#3-datos-análisis-y-limpieza)
4. [Capas de datos en PostgreSQL](#4-capas-de-datos-en-postgresql)
5. [El DAG de Airflow](#5-el-dag-de-airflow)
6. [Entrenamiento, métrica y umbral](#6-entrenamiento-métrica-y-umbral)
7. [Registro y promoción en MLflow](#7-registro-y-promoción-en-mlflow)
8. [API de inferencia](#8-api-de-inferencia)
9. [Interfaz gráfica (Streamlit)](#9-interfaz-gráfica-streamlit)
10. [Observabilidad](#10-observabilidad)
11. [Pruebas de carga y límite operativo](#11-pruebas-de-carga-y-límite-operativo)
12. [Despliegue paso a paso](#12-despliegue-paso-a-paso)
13. [Imágenes publicadas](#13-imágenes-publicadas)
14. [Problemas encontrados y cómo se resolvieron](#14-problemas-encontrados-y-cómo-se-resolvieron)
15. [Limitaciones](#15-limitaciones)
16. [Estructura del repositorio](#16-estructura-del-repositorio)
17. [Evidencias](#17-evidencias)

---

## 1. Cumplimiento del enunciado

| Requisito | Dónde se cumple |
|---|---|
| Todo en Kubernetes: Deployment/StatefulSet, Service, ConfigMap, Secret, PVC, requests/limits, probes | `k8s/base/*`, sección 2 |
| Imágenes propias en Docker Hub, sin imágenes locales | Sección 13 |
| Airflow dentro del clúster | Helm chart oficial 1.22.0, `airflow/values.yaml` |
| DAG con validación de la fuente, carga por lotes (≤15.000), calidad, preprocesamiento, tabla limpia, partición, entrenamiento, registro, comparación y promoción | `airflow/dags/diabetes_pipeline.py`, sección 5 |
| Idempotencia, control de duplicados, errores visibles | Sección 5.3 |
| Capas RAW / CLEAN / INFERENCE LOGS separadas | Esquemas `raw`, `clean`, `inference`, sección 4 |
| `raw` con lote, fecha de carga, origen, estado e id | `batch_id`, `load_timestamp`, `source_file`, `status`, `row_hash`, `encounter_id` |
| MLflow con metadatos en PostgreSQL (no SQLite) y artefactos en MinIO | `k8s/base/mlflow`, sección 7 |
| Métrica principal justificada (no accuracy) | PR-AUC + umbral F2 con tope, sección 6 |
| Promoción automática del mejor modelo | Tareas `compare_with_champion` y `promote`, alias `@champion` |
| API con `/health`, `/predict`, `/model-info` y `/metrics`, modelo dinámico desde MLflow | `ml/src/api/main.py`, sección 8 |
| `/predict` devuelve predicción, probabilidad, modelo, versión/alias y tiempo | Sección 8.3 |
| Registro de cada inferencia (fecha, entrada, predicción, probabilidad, modelo, versión, tiempo, id) | `inference.predictions`, sección 4 |
| Streamlit: ingresar valores, **cargar ejemplos**, enviar, ver predicción y versión, ver errores de validación; solo habla con el API | `ui/app.py`, sección 9 |
| Locust contra el API en Kubernetes | `k8s/base/locust`, sección 11 |
| Dashboard: total de solicitudes, req/s, latencia promedio y percentiles, errores, tasa de error, CPU/memoria del pod | `k8s/base/grafana`, sección 10 |
| Límite operativo explicado | Sección 11 |

## 2. Arquitectura

```
                 ┌─────────────────────────── Kubernetes (K3s en Rancher) ───────────────────────────┐
                 │  namespace airflow                     namespace mlops                            │
 Diabetes.csv    │  ┌────────────────┐  KubernetesPod  ┌──────────────────┐                          │
 (bucket MinIO)──┼─▶│ Airflow        │────Operator────▶│ pods trainer     │──┐                       │
                 │  │ scheduler,     │                 │ (preprocess,     │  │ runs, modelos         │
                 │  │ api-server,    │──SQL──┐         │ train, compare,  │  ▼                       │
                 │  │ dag-processor  │       │         │ promote)         │ ┌────────┐   ┌───────┐   │
                 │  └────────────────┘       ▼         └────────┬─────────┘ │ MLflow │──▶│ MinIO │   │
                 │                    ┌──────────────┐          │           └───┬────┘   └───────┘   │
                 │                    │ PostgreSQL   │◀─────────┘   metadatos   │                    │
                 │                    │ diabetes:    │◀──────────────────────────┘ (base mlflow)      │
                 │                    │ raw · clean ·│                                               │
                 │                    │ inference    │◀──registro de inferencias──┐                  │
                 │                    └──────────────┘                            │                  │
                 │   ┌──────────┐ HTTP  ┌────────────────┐  @champion (cada 60 s) │                  │
 navegador ──────┼──▶│Streamlit │──────▶│ API (FastAPI)  │────────────────────────┘                  │
                 │   └──────────┘       │ /predict ...   │◀────── Locust                            │
                 │                      └───────┬────────┘                                          │
                 │        Prometheus ◀── /metrics + cAdvisor (CPU/memoria) ──▶ Grafana              │
                 └──────────────────────────────────────────────────────────────────────────────────┘
```

**Por qué esta forma:**
- **Un solo PostgreSQL con bases separadas:**
  - `diabetes`, con los esquemas `raw`, `clean` e `inference`;
  - `mlflow`;
  - `airflow`.

  Cada una tiene su propio rol. El enunciado permite esquemas dentro de una misma instancia, y
  separar bases y roles mantiene el aislamiento sin pagar tres servidores en una VM de 15 GiB.
- **StatefulSet** para lo que guarda datos (PostgreSQL, MinIO, Prometheus) y **Deployment** para lo
  que no guarda estado (MLflow, API, UI, Grafana, Locust).
- **Los pasos pesados del DAG corren como pods aparte** (KubernetesPodOperator), con la misma
  imagen que el API. Así tienen sus propios recursos (un OOM mata el pod, no al scheduler), no hay
  conflictos de dependencias con Airflow, y **se entrena y se sirve con exactamente las mismas
  versiones de librerías**.
- **Manifiestos YAML con Kustomize** (`kubectl apply -k k8s/base`). Se eligió así para controlar
  cada recurso y probe, y para no depender de charts de terceros. Airflow es la excepción: se
  instala con su chart oficial, como recomienda la guía del curso.

### Componentes y recursos

| Componente | Tipo | Imagen | Requests → Limits | Puerto VM |
|---|---|---|---|---|
| PostgreSQL | StatefulSet + PVC 5Gi | `postgres:16-alpine` | 100m/256Mi → 1/1Gi | interno |
| MinIO | StatefulSet + PVC 5Gi | `cgr.dev/chainguard/minio@sha256:…` | 100m/256Mi → 500m/512Mi | 8012 (consola) |
| MLflow | Deployment | `sebas10acosta/mlops-mlflow:3.16.1` | 200m/640Mi → 1/1Gi | 8011 |
| Airflow scheduler | StatefulSet (chart) | `sebas10acosta/mlops-airflow:0.1.0` | 250m/768Mi → 1/1536Mi | — |
| Airflow api-server | Deployment (chart) | ídem | 200m/512Mi → 1/1Gi | 8010 |
| Airflow dag-processor | Deployment (chart) | ídem | 100m/384Mi → 500m/768Mi | — |
| Pods trainer (por tarea) | Pod (KubernetesPodOperator) | `sebas10acosta/mlops-diabetes-ml:0.1.0` | 500m/768Mi → 2/1536Mi (train: 2Gi) | — |
| API de inferencia | Deployment | `sebas10acosta/mlops-diabetes-ml:0.2.2` | 250m/512Mi → 1/1Gi | 8013 |
| Streamlit | Deployment | `sebas10acosta/mlops-diabetes-ui:0.1.2` | 50m/128Mi → 500m/384Mi | 8014 |
| Grafana | Deployment | `grafana/grafana:12.1.1` | 50m/128Mi → 500m/384Mi | 8015 |
| Prometheus | StatefulSet + PVC 2Gi | `prom/prometheus:v3.5.0` | 100m/256Mi → 500m/768Mi | 8016 |
| Locust | Deployment | `locustio/locust:2.46.7` | 100m/128Mi → 1/512Mi | 8017 |

Hay también un `LimitRange` en `mlops` que da límites por defecto (10m/32Mi → 200m/128Mi) a
cualquier contenedor que no los declare. Lo necesita el *sidecar* de XCom que KubernetesPodOperator
añade sin límites.

**Exposición:** la VM solo permite los puertos 8000–8025. Como K3s vive *dentro* del contenedor de
Rancher, ese contenedor publica `8010-8020 → 30010-30020`, y cada interfaz es un NodePort en ese
rango (tabla en `k8s/README.md`). Así no se necesita `kubectl port-forward`, que además quedaría
en medio de las pruebas de carga.

## 3. Datos: análisis y limpieza

**El conjunto de datos** tiene 101.766 encuentros, 50 columnas y 71.518 pacientes, sin filas
duplicadas. La variable `readmitted` toma los valores `NO` (53,9%), `>30` (34,9%) y `<30` (11,2%).

**Hallazgos que guiaron la limpieza:**
- **Pacientes repetidos:** 16.773 pacientes tienen 2 o más encuentros, hasta 40. Partir por fila
  filtraría al mismo paciente entre entrenamiento y prueba, y las métricas saldrían optimistas.
  → **La partición se hace por paciente.**
- **`"None"` en `max_glu_serum` (95%) y `A1Cresult` (83%) significa "prueba no realizada"**: es una
  categoría, no un faltante. pandas convierte la palabra "None" en NaN por defecto, así que se lee
  con `keep_default_na=False`.
- **Faltantes marcados con `"?"`:** weight (97%), payer_code (40%), medical_specialty (49%),
  race (2%) y diag (≤1,4%).
- **Columnas constantes:** examide y citoglipton. Otros 10 medicamentos aparecen en menos de 100
  encuentros.
- **1.652 encuentros terminaron en fallecimiento** (códigos de alta 11, 19, 20 y 21) **y 771 en
  hospicio** (13 y 14): esos pacientes no pueden ser readmitidos.
- **Hay señal:** con 0 hospitalizaciones previas, el 8,4% vuelve en menos de 30 días; con 5 o más,
  el 36,4%.

**Decisiones de limpieza** (código: `ml/src/diabetes_features/cleaning.py`, 34 pruebas unitarias):

| Decisión | Motivo |
|---|---|
| Objetivo binario: `<30` contra el resto | La readmisión a 30 días es el indicador clínico estándar |
| Excluir fallecidos y hospicio, y 3 filas con género Unknown/Invalid (101.766 → **99.340**) | No pertenecen a la población a la que se le aplicaría el modelo |
| Eliminar weight y payer_code | 97% y 40% faltantes. payer_code no aporta (medido: −0,001 de PR-AUC) |
| Eliminar examide, citoglipton y los 10 medicamentos con menos de 100 usos | Constantes o casi constantes |
| `?` → `Unknown` en race y medical_specialty | La ausencia de dato es informativa; se conserva como categoría |
| diag_1/2/3 (717–790 códigos ICD-9) → **9 grupos clínicos** | Agrupación de Strack et al. (2014). Medido: 0,239 de PR-AUC frente a 0,235 con los 3 primeros dígitos |
| Edad `[70-80)` → 75 (punto medio) | Conserva el orden de los rangos |
| Columnas `*_id` como categóricas | Son códigos, no cantidades |
| Sin variables derivadas | Se probaron y no mejoraron el resultado (0,237) |
| Codificación: `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50)` | Categorías nuevas o raras en producción no rompen el API |

Las mismas funciones de limpieza las usan los pods de entrenamiento **y** el API. Es un solo
paquete (`diabetes_features`) dentro de una sola imagen, así que no puede haber diferencias entre
cómo se entrena y cómo se sirve.

**Comparación con el estudio que publicó el dataset (Strack et al., 2014):**
- **Coincidimos en:** excluir fallecidos y hospicio, eliminar weight y payer_code, tratar los
  faltantes como categoría, los 9 grupos ICD-9 y el objetivo `<30`.
- **Ellos se quedan con un encuentro por paciente** (69.984 filas), porque su objetivo es inferencia
  estadística y necesitan observaciones independientes. Nuestro objetivo es predecir, así que se
  conservan todos los encuentros y se evita la fuga con la partición por paciente.

## 4. Capas de datos en PostgreSQL

Las tablas se crean con scripts idempotentes (`airflow/dags/sql/`), que el DAG ejecuta al inicio
de cada corrida.

| Capa | Tabla | Contenido |
|---|---|---|
| RAW | `raw.diabetes` | Las 50 columnas **como texto, sin transformar**, más `batch_id`, `load_timestamp`, `source_file`, `row_hash` y `status` (`loaded` → `valid`/`rejected` → `processed`/`excluded`) con `status_reason`. Clave primaria: `encounter_id` |
| RAW | `raw.batch_log` | Un registro por lote: qué corrida (`dag_run_id` único) cargó qué filas |
| CLEAN | `clean.diabetes` | Las 33 variables tipadas, el objetivo `readmitted_30d`, `split` (train/validation/test) y `batch_id` (trazable hasta `raw`) |
| INFERENCE LOGS | `inference.predictions` | `request_id` (UUID), `created_at`, `input` (JSONB, tal como llegó), `prediction`, `probability`, `threshold`, `model_name`, `model_version`, `model_alias`, `latency_ms` |

## 5. El DAG de Airflow

`diabetes_pipeline` (Airflow 3.2.2, LocalExecutor):

```
create_tables → validate_source → load_raw_batch → quality_checks → preprocess (pod)
   → assign_split → train (pod) → compare_with_champion (pod) → promote (pod)
```

### 5.1 Tareas

| Tarea | Dónde | Qué hace |
|---|---|---|
| `create_tables` | Airflow | Aplica los scripts SQL (idempotentes) |
| `validate_source` | Airflow | Verifica que `s3://datasets/diabetes/Diabetes.csv` exista en MinIO y que su encabezado coincida con las columnas de `raw` |
| `load_raw_batch` | Airflow | Toma el siguiente tramo de **≤15.000 filas** y lo inserta como texto en `raw.diabetes`. Si ya no queda nada, **salta** todo lo que sigue |
| `quality_checks` | Airflow | Valida formatos y el objetivo de cada fila (`valid`/`rejected` con motivo). **Falla la corrida si más del 5% de las filas son malas** |
| `preprocess` | Pod | Filas `valid` → `clean_features` → `clean.diabetes`. Marca las filas `processed`/`excluded`, todo en una transacción |
| `assign_split` | Airflow (SQL) | Hash md5 de `patient_nbr` → 70/15/15, solo para filas nuevas: un paciente nunca cambia de partición |
| `train` | Pod | Entrena 6 candidatos y registra todo en MLflow (sección 6) |
| `compare_with_champion` | Pod | Retador contra `@champion` en el conjunto de prueba actual |
| `promote` | Pod | Registra la versión con sus etiquetas y mueve `@champion` si ganó |

Los resultados de los pods vuelven a Airflow por XCom (`/airflow/xcom/return.json`) y pasan al
siguiente paso como JSON en la variable `STEP_INPUT`.

### 5.2 Periodicidad

Por defecto el DAG es **manual**, lo que facilita la demostración. Para automatizarlo basta
definir `DIABETES_PIPELINE_SCHEDULE` (por ejemplo `*/10 * * * *`) en los valores de Helm.
`max_active_runs=1`: dos corridas nunca cargan a la vez.

### 5.3 Idempotencia y fallos

- **Duplicados:**
  - `encounter_id` es la clave primaria y los inserts usan `ON CONFLICT DO NOTHING`.
  - `raw.batch_log` asocia corrida ↔ lote: **un reintento reutiliza su propio lote** en vez de
    tomar el siguiente.
  - Cuando no quedan lotes, la corrida termina en *skipped*. Una octava corrida lo demostró.
- **Fallos:**
  - Cada tarea tiene 1 reintento (30 s).
  - Cada escritura es una transacción, así que no quedan lotes a medias.
  - Un fallo de calidad usa `AirflowFailException` (visible y sin reintento inútil).
  - Un pod que falla deja sus logs en Airflow.
- **Tiempos medidos:** de 1 min 50 s (lote 1) a 3 min 36 s (lote 7). El entrenamiento crece con
  los datos.

## 6. Entrenamiento, métrica y umbral

**Candidatos** (`ml/src/trainer/train.py`): 6 por corrida, todos con `class_weight="balanced"` y
el mismo preprocesamiento (`OneHotEncoder` + `StandardScaler`):
- Regresión logística con C = 0,1 y C = 1.
- `HistGradientBoosting` con learning rate 0,05 y 0,1, combinados con 15 y 31 hojas.

**Métrica de selección: PR-AUC en validación.**
- Con 11,4% de positivos, la accuracy premia no predecir nunca (88,7%).
- La ROC-AUC se ve optimista cuando hay tantos negativos.
- La PR-AUC mide directamente qué tan bien se encuentran los positivos. El azar da 0,114.

**Umbral de decisión: el F2 máximo en validación, marcando como máximo el 30% de los pacientes.**
- Un falso negativo (un paciente que vuelve sin haber sido marcado) cuesta más que un falso
  positivo, y F2 pesa más el recall.
- **Hallazgo:** sin tope, el F2 marcaba al ~54% de los pacientes (recall 0,76, precisión 0,17).
  Con un modelo modesto, F2 empuja a marcar a casi todos, y eso no sirve operativamente.
- Con el tope de 30% (`MAX_FLAGGED_RATE`): umbral ≈ 0,52, **recall ≈ 0,51, precisión ≈ 0,19**, es
  decir, 1,7 veces la tasa base.
- En producción, Grafana muestra que el API marca al **29,8%**. El tope se cumple también con datos
  reales.

**Registrado en MLflow por cada candidato:**
- parámetros;
- métricas de validación (PR-AUC, ROC-AUC, precisión, recall, F2, % marcado);
- el umbral;
- la curva precisión-recall y la matriz de confusión (PNG);
- el modelo serializado con su firma y ejemplo de entrada.

Los candidatos quedan anidados bajo una corrida padre `pipeline_<dag_run_id>`.

**Resultado:** ROC-AUC ≈ 0,67–0,68, típica para este dataset. El problema se puede aprender, pero
es difícil.

## 7. Registro y promoción en MLflow

MLflow 3.16.1 guarda los metadatos en PostgreSQL (base `mlflow`) y los artefactos en MinIO
(bucket `mlflow`). Los artefactos pasan *a través* del servidor (proxied artifacts), así que **solo
el pod de MLflow tiene credenciales de MinIO**.

**Regla de promoción:**
1. Cada corrida registra su mejor candidato como **una versión nueva** de `diabetes_readmission`.
   Así queda el historial completo.
2. `compare_with_champion` evalúa al retador y al campeón actual **sobre las mismas filas**: el
   conjunto de prueba actual, que ninguno de los dos usó para entrenar ni ajustar.
3. El retador gana solo si su PR-AUC de prueba es **estrictamente mayor**. Si no hay campeón, gana
   por defecto.
4. Si gana, `@champion` pasa a apuntar a esa versión.

**Etiquetas de cada versión:**
- `decision_threshold`
- métricas de validación y prueba
- `dag_run_id`
- `promoted`
- `comparison`, el texto de la comparación

**Historial real:**

| Versión | Modelo | PR-AUC prueba | Comparación | ¿Promovida? |
|---|---|---|---|---|
| v1 | LogReg C=0,1 | 0,2184 | sin campeón | ✅ |
| v2 | HistGB | 0,2483 | vs v1 | ✅ |
| v3 | HistGB | 0,2494 | vs v2 | ✅ |
| v4 | HistGB | 0,2400 | vs v3 (reevaluado en el nuevo conjunto de prueba) | ✅ |
| v5 | HistGB | 0,2334 | vs v4 | ❌ |
| v6 | HistGB | 0,2283 | vs v4 | ❌ |
| **v7** | **HistGB** | **0,2351** | vs v4 (0,2330) | ✅ **@champion** |

> **Nota:** v4 aparece con una PR-AUC menor que la de v3 y aun así fue promovida. No es un error.
> El conjunto de prueba crece con cada lote, y la comparación reevalúa **ambos** modelos sobre el
> conjunto de ese momento. Las etiquetas de versiones distintas se midieron sobre conjuntos
> distintos y no son comparables entre sí; solo es válida la comparación hecha dentro de la misma
> corrida.

**Modelo productivo (v7):** umbral 0,525, recall de validación 0,507, precisión de validación
0,192, PR-AUC de prueba 0,235.

## 8. API de inferencia

`ml/src/api/main.py`. FastAPI con un worker de uvicorn por pod; se escala con réplicas.

### 8.1 Cómo carga el modelo (sin cambios de código)

**Al arrancar** resuelve `models:/diabetes_readmission@champion` a un número de versión y carga
**esa** versión. Así, si el alias se mueve durante la descarga, el modelo y su información siguen
siendo coherentes.

**Recarga:**
- **Un hilo en segundo plano revisa cada 60 s** (`RELOAD_INTERVAL_SECONDS`) a qué versión apunta el
  alias. Es una sola llamada de metadatos, y **solo descarga si la versión cambió**.
- **`POST /reload`** fuerza esa misma revisión sin esperar.

**Protecciones:**
- El modelo y su información se reemplazan **en una sola asignación**: una petición nunca mezcla
  versiones.
- Un candado evita dos cargas simultáneas.
- Si la revisión falla (MLflow caído, sin campeón, versiones de librerías distintas), **se conserva
  el modelo actual**.
- El umbral se lee de la etiqueta `decision_threshold` de la versión: cada modelo trae el umbral con
  el que se ajustó.
- Se verifica que scikit-learn, numpy y skops coincidan con las versiones con que se entrenó el
  modelo.

Verificado: al promover una versión nueva en MLflow, el API la tomó en el siguiente ciclo, con su
umbral nuevo.

### 8.2 Endpoints

| Método | Ruta | Respuesta |
|---|---|---|
| GET | `/health` | 200 si hay modelo cargado, 503 si no. Es el readiness probe: un pod sin modelo no recibe tráfico |
| POST | `/predict` | Predicción para un encuentro (abajo) |
| GET | `/model-info` | Nombre, alias, versión, run, tipo, umbral, métricas, motivo de promoción, fecha de carga y última revisión |
| POST | `/reload` | Revisa `@champion` ahora |
| GET | `/metrics` | Métricas para Prometheus |
| GET | `/docs` | Swagger, con un ejemplo listo |

### 8.3 `/predict`

**Entrada:** el encuentro en el **formato crudo del CSV**, validado con Pydantic:
- rangos de edad como `[70-80)`;
- códigos ICD-9;
- `?` para desconocido;
- valores cerrados (`Literal`) para edad, medicamentos, glucosa/A1C, cambio y género.

Si la entrada es inválida, el API responde **422** con el detalle.

```json
{
  "request_id": "6eb1bcb8-9323-4122-975b-33e0da56ace7",
  "prediction": 1,
  "label": "readmission <30 days: HIGH risk",
  "probability": 0.6641,
  "decision_threshold": 0.525309,
  "model_name": "diabetes_readmission",
  "model_version": "7",
  "model_alias": "champion",
  "latency_ms": 115.23
}
```

**Registro de inferencias:** cada predicción se inserta en `inference.predictions` **después de
enviar la respuesta** (`BackgroundTasks`), así la base de datos no suma latencia. Si la escritura
falla, se cuenta en `api_inference_log_errors_total` en vez de devolverle un error a un cliente
que ya tiene su predicción.

### 8.4 Métricas expuestas

| Métrica | Qué mide |
|---|---|
| `api_predict_requests_total{status}` | Respuestas de `/predict` por **código HTTP**, incluidos 422 (validación) y 500 (error inesperado). Se cuentan en un middleware |
| `api_predict_latency_seconds` | Histograma del tiempo de `/predict` (limpieza + modelo), con buckets hasta 60 s |
| `api_predictions_total{prediction, model_version}` | Predicciones por clase y versión |
| `api_predicted_probability` | Distribución de las probabilidades |
| `api_model_version` | Versión servida (0 = ninguna) |
| `api_model_reloads_total{trigger}` | Cambios de campeón detectados (startup/poll/manual) |
| `api_inference_log_errors_total` | Inferencias no registradas |

Los endpoints livianos (`/health`, `/metrics`, `/model-info`) son `async`: corren en el event loop
y **no hacen fila detrás de las predicciones**. Esto salió de las pruebas de carga (sección 11).

## 9. Interfaz gráfica (Streamlit)

`ui/app.py`, en su propia imagen pequeña. **Solo habla con el API**, nunca con MLflow ni con la
base de datos.
- **Barra lateral:** el modelo en producción (versión, tipo, umbral, métricas y motivo de
  promoción), tomado de `/model-info`.
- **"Load example values":** carga un ejemplo hecho a mano o uno de 7 encuentros reales del dataset
  en el formulario.
- **Formulario:** paciente, admisión, estancia e historial, diagnósticos y pruebas, y medicamentos.
- **Resultado:** riesgo alto o bajo, probabilidad, umbral, versión del modelo, latencia e id de la
  petición registrada.
- **Errores:** si el API rechaza la entrada (422), se muestra su detalle de validación.

## 10. Observabilidad

**Prometheus** (`k8s/base/prometheus`, retención de 7 días o 1,5 GB) tiene tres fuentes:
1. **Pods del API:** se descubren con `kubernetes_sd` por la anotación `prometheus.io/scrape`.
   Cada réplica es su propio target, y el dashboard suma.
2. **CPU y memoria de cada contenedor:** se leen del **cAdvisor del kubelet a través del proxy del
   API server**, con un ClusterRole de solo lectura. El K3s de Rancher no trae metrics-server, y
   así no hace falta instalar node-exporter ni kube-prometheus-stack (~1,5 GB de RAM). Solo se
   conservan las 7 métricas que usa el dashboard, para `mlops` y `airflow`.
3. Prometheus mismo.

**Grafana** (`k8s/base/grafana`) no tiene estado propio: la fuente de datos y el dashboard
*Diabetes readmission API* vienen de ConfigMaps versionados en git, y el dashboard abre como página
de inicio.

| Fila | Paneles |
|---|---|
| Overview | Versión servida, réplicas arriba, req/s, latencia p95, % fallidas, errores de registro de inferencias |
| | **Total de solicitudes**, **latencia promedio**, **número de errores** (en el rango de tiempo) y **tasa de error** |
| Traffic and latency | Req/s por pod, **respuestas por código HTTP**, latencia p50/p95/p99 |
| API resources | CPU del API vs su límite (1 core), % de *throttling*, memoria vs 1 GiB |
| Model output | Predicciones/s por clase, % marcado como alto riesgo |
| Whole system | CPU y memoria de cada pod de `mlops` y `airflow` |

## 11. Pruebas de carga y límite operativo

**Escenario** (`k8s/base/locust/locustfile.py`, Locust 2.46.7 **dentro del clúster**, contra
`http://api:8000`):
- **Cada usuario simulado envía un encuentro real al azar** de `payloads.json`: 200 encuentros
  elegibles del CSV con semilla fija, todos validados contra el modelo Pydantic del API.
- **Pausa de 1 a 2,5 s** entre peticiones, es decir, ~0,57 req/s por usuario.
- **Una petición cuenta como exitosa** solo si devuelve 200, una predicción en {0, 1} y una
  probabilidad en [0, 1].

### 11.1 Primera prueba: un defecto encontrado

Con 1 réplica, a 100 y 200 usuarios:
- el throughput se quedó en **~8 req/s**;
- Locust marcó **~1 fallo/s**, mientras el API contaba **0% de errores**;
- había **huecos** en Grafana y el panel "API replicas up" parpadeaba.

**Causa:**
1. `/health` y `/metrics` eran funciones síncronas y hacían fila en el mismo pool de hilos que las
   predicciones.
2. El readiness probe (timeout de 1 s) fallaba, así que **Kubernetes sacaba al pod del Service**.
3. Las conexiones fallaban antes de llegar al API, por eso el API no las contaba.
4. Los scrapes de Prometheus también vencían, y de ahí los huecos.

**Corrección** (imagen 0.2.1):
- endpoints livianos `async`;
- timeout de readiness de 3 s;
- buckets de latencia hasta 60 s (antes todo lo saturado se veía como "5 s").

**Prueba A/B local** (1 CPU, 100 usuarios): `/health` bajo carga pasó de **89–2116 ms a 33–770 ms**,
con el mismo throughput.

### 11.2 Resultados después de la corrección

| Usuarios | Réplicas | RPS | Latencia Locust | Fallos |
|---|---|---|---|---|
| 10 | 1 | ~5 | p95 ~1 s | 0% |
| 50 | 1 | **~7–8 (techo)** | p50 ~5 s, p95 6–10 s | 0% |
| 200 | 3 | ~13,5 (picos de 19) | p50 ~12 s, p95 20–25 s | 0% |
| 50 | 3 | **~15–17** | **p50 ~1,5 s**, p95 ~3–5 s | 0% |

### 11.3 Análisis

- **Límite operativo de una réplica: ~8 req/s, equivalente a unos 14 usuarios con este patrón.**
  Cada predicción cuesta ~120 ms de CPU en la VM: construir el DataFrame, limpiar con pandas y
  aplicar el pipeline del modelo. 10 usuarios piden 5,7 req/s y se atienden; 50 piden ~29 req/s y
  el API se satura.
- **La latencia es fila de espera, no código lento.** La ley de Little lo predice:
  latencia = usuarios / throughput − pausa.
  - 200 usuarios, 1 réplica: 200/8 − 1,75 ≈ 23 s. Locust midió un p50 de ~20–25 s.
  - 200 usuarios, 3 réplicas: 200/13,5 − 1,75 ≈ 13 s. Locust midió ~12 s.
  - 50 usuarios, 3 réplicas: 50/15 − 1,75 ≈ 1,6 s. Locust midió ~1,5 s.
- **El cuello de botella de una réplica es un solo proceso de Python (el GIL), no el límite de
  Kubernetes.** El pod usa ~0,85 core con solo ~3% de *throttling*. Subir el límite de CPU no
  ayudaría; las réplicas sí.
- **Con 3 réplicas el throughput se duplica (8 → ~15 req/s), pero no se triplica:** cada pod usa
  solo ~0,5 core, con 0% de *throttling*. El límite pasa a ser **la VM**: `vmstat` muestra un 75–84%
  de CPU de usuario, 3–13% libre y **13–33 procesos esperando CPU con 4 vCPU**. Las 4 vCPU se
  reparten entre las 3 réplicas, Rancher/K3s, Locust, Prometheus y Airflow.
- **La memoria no es un factor:** ~230–250 MiB por réplica, estable, frente a 1 GiB de límite.
- **Mejoras posibles:**
  - **Más throughput por core:** predecir sin pandas por fila, o agrupar peticiones en lotes.
  - **Más capacidad total:** más nodos; las réplicas solo escalan mientras quede CPU en el nodo.

## 12. Despliegue paso a paso

### 12.1 VM (Rocky Linux 9, 4 vCPU, 15 GiB)

1. **Puertos 8000–8025/tcp** abiertos en firewalld.
2. **Docker con sus datos en `/docker-data`:** `/var` no tenía espacio.
   - `data-root` en `/etc/docker/daemon.json`;
   - `root` en `/etc/containerd/config.toml`, porque Docker usa el *containerd image store*;
   - etiquetas de SELinux con `semanage fcontext -e` + `restorecon`.
3. **Módulos del kernel** para Rancher: `ip_tables nf_conntrack iptable_filter xt_state`.
4. **Rancher:**
   ```bash
   sudo docker run -d --name rancher --privileged --restart=unless-stopped \
     -p 8002:80 -p 8001:443 -p 8010-8020:30010-30020 \
     -v rancher_data:/var/lib/rancher rancher/rancher:v2.15.2
   ```
5. **kubectl v1.36.2** con el kubeconfig de Rancher, y **Helm v3.22.1**.
6. **StorageClass:** el K3s de Rancher no trae ninguna, así que se instala local-path-provisioner:
   ```bash
   kubectl apply -f k8s/infra/local-path-storage.yaml
   ```

### 12.2 Secretos

```bash
cp k8s/secrets.example.env k8s/secrets.env      # editar: solo letras y dígitos
kubectl create namespace mlops
kubectl -n mlops create secret generic mlops-secrets --from-env-file=k8s/secrets.env
```

`secrets.env` está en `.gitignore`. Las contraseñas no pueden tener símbolos porque van dentro de
URLs (`postgresql://usuario:clave@host`).

### 12.3 Infraestructura, API, UI y observabilidad

```bash
kubectl apply -k k8s/base
kubectl -n mlops get pods,pvc
```

### 12.4 Airflow

```bash
bash k8s/airflow/create-secrets.sh
set -a; source k8s/secrets.env; set +a
helm repo add apache-airflow https://airflow.apache.org && helm repo update
helm upgrade --install airflow apache-airflow/airflow --version 1.22.0 \
  --namespace airflow -f airflow/values.yaml \
  --set createUserJob.defaultUser.password="$AIRFLOW_ADMIN_PASSWORD" \
  --wait --timeout 20m
```

### 12.5 Datos y ejecución

1. Subir `Diabetes.csv` al bucket `datasets`, en la ruta `diabetes/Diabetes.csv`, desde la consola
   de MinIO (puerto 8012).
2. En Airflow (puerto 8010), activar y disparar `diabetes_pipeline` 7 veces, una por lote.
3. Abrir MLflow (8011), probar el API en `http://<VM>:8013/docs` y la UI en el puerto 8014.
4. Abrir Grafana (8015) y lanzar la carga desde Locust (8017).

**Escalar el API** para las pruebas:
```bash
kubectl -n mlops scale deploy/api --replicas=3
```

### 12.6 Qué requiere reconstruir

| Cambio | Acción |
|---|---|
| Código del API o del entrenamiento | Nueva imagen `mlops-diabetes-ml:<tag>` → actualizar el tag en `k8s/base/api/deployment.yaml` (API) o `ML_IMAGE` en `airflow/values.yaml` (trainer) |
| DAG | Nueva imagen `mlops-airflow:<tag>` → `helm upgrade` |
| Dashboard, `prometheus.yml` o locustfile | Solo `kubectl apply -k k8s/base`: el `configMapGenerator` renombra el ConfigMap y reinicia el pod |
| **Un modelo nuevo** | **Nada:** el API lo toma solo desde MLflow |

## 13. Imágenes publicadas

Todas en Docker Hub, usuario [`sebas10acosta`](https://hub.docker.com/u/sebas10acosta), con
versiones fijas (nunca `latest`):

| Imagen | Contenido | Tamaño |
|---|---|---|
| `mlops-mlflow:3.16.1` | Servidor MLflow (`--workers` configurable) | — |
| `mlops-airflow:0.1.0` | `apache/airflow:slim-3.2.2-python3.12` + providers postgres, amazon, cncf-kubernetes y fab + DAGs | 1,41 GB (la imagen oficial completa pesa 3,12 GB) |
| `mlops-diabetes-ml:0.1.0` / `0.2.2` | Python 3.11 + uv: `diabetes_features`, trainer y API. Versiones fijadas con `uv.lock`: scikit-learn 1.9.1, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, skops 0.16.0, mlflow-skinny 3.16.1 | 832 MB |
| `mlops-diabetes-ui:0.1.2` | Streamlit 1.50.0 + requests | ~190 MB comprimida |

## 14. Problemas encontrados y cómo se resolvieron

| Problema | Solución |
|---|---|
| `/var` lleno, Rancher no cabía | Datos de Docker y containerd movidos a `/docker-data`, con etiquetas SELinux |
| Rancher solo publica sus puertos 80/443, fuera del rango permitido | Mapeo a 8001/8002, y 8010–8020 → NodePorts 30010–30020 |
| El K3s de Rancher no trae StorageClass y los PVCs quedaban *Pending* | local-path-provisioner v0.0.37 |
| La red universitaria corta `raw.githubusercontent.com` | Archivos descargados en el PC y copiados con `scp` |
| MLflow usaba 1,9 GiB incluso con 2 workers | 8 procesos *huey* de MLflow 3 (~1,7 GB) que no se usan → `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` → **624 MiB** |
| El script de inicio de PostgreSQL se ejecutaba con `source` bajo `set -eu` | Montado con `defaultMode: 0555` |
| Primera instalación de Airflow: `context deadline exceeded`, pods en `Init:CrashLoopBackOff` | Los Jobs de migración eran *Helm hooks* y se bloqueaban mutuamente con `--wait` → `useHelmHooks: false` |
| El chart de Airflow pide 100Gi para logs (`workers.celery.persistence`) | 2Gi |
| El chart necesita el provider FAB | Añadido a la imagen |
| El sidecar de XCom de KubernetesPodOperator no declara límites | `LimitRange` en `mlops` |
| skops se niega a cargar árboles de HistGB | `skops_trusted_types` explícitos por familia de modelo |
| pandas convierte "None" en NaN | `keep_default_na=False` |
| El umbral F2 marcaba al 54% de los pacientes | Tope del 30% de marcados |
| Bajo carga, el pod salía del Service y había huecos en las métricas | Endpoints livianos `async`, timeout de readiness de 3 s (sección 11) |
| El contenedor de Rancher se reinicia en ráfagas (código 1, no OOM) | Los PVCs y el *self-healing* de Kubernetes conservaron todos los datos. Se limpiaron los `index.lock` huérfanos de sus catálogos. Antes de demostrar algo, revisar `kubectl get pods -A` |

## 15. Limitaciones

- **Clúster de un solo nodo dentro de un contenedor:** si Rancher se reinicia, se reinician todos
  los pods. No hay alta disponibilidad real.
- **El modelo es modesto** (ROC-AUC ~0,68). Es lo típico para este dataset con estas variables; el
  foco del proyecto es el sistema.
- **El orden de los lotes es el del archivo.** No hay fecha, así que no se simula una deriva
  temporal real.
- **Las réplicas escalan solo hasta agotar las 4 vCPU de la VM.**
- **Las credenciales se crean a mano** desde `secrets.env`. No hay un gestor de secretos.

## 16. Estructura del repositorio

```
proyectos/proyecto_II/
├── airflow/
│   ├── Dockerfile               # imagen slim + providers + DAGs
│   ├── values.yaml              # valores del chart oficial 1.22.0
│   └── dags/
│       ├── diabetes_pipeline.py
│       └── sql/                 # 01_raw, 02_clean, 03_inference, assign_split
├── ml/                          # una imagen: trainer + API
│   ├── Dockerfile, pyproject.toml, uv.lock
│   ├── src/diabetes_features/   # limpieza compartida (entrenamiento = inferencia)
│   ├── src/trainer/             # preprocess, train, compare, promote
│   ├── src/api/main.py          # FastAPI
│   └── tests/                   # 34 pruebas de la limpieza
├── ui/                          # Streamlit: app.py, examples.json, Dockerfile
├── mlflow/                      # imagen del servidor MLflow
├── k8s/
│   ├── README.md                # puertos y comandos de despliegue
│   ├── secrets.example.env
│   ├── infra/local-path-storage.yaml
│   ├── airflow/create-secrets.sh
│   └── base/                    # Kustomize: postgres, minio, mlflow, api, ui,
│                                #   prometheus, grafana (dashboard JSON), locust
└── images/                      # evidencias
```

## 17. Evidencias

| # | Archivo | Qué muestra |
|---|---|---|
| 01–02 | `01-rancher-local`, `02-prueba-pvc` | Clúster local en Rancher y prueba de PVC |
| 03–06 | `03-mlflow-ui`, `04-minio-bucket`, `05-pods-mlops`, `06-dataset-minio` | Infraestructura base y CSV en MinIO |
| 07–08 | `07-airflow-dag-run`, `08-airflow-8-runs` | DAG completo; 7 lotes y la 8.ª corrida saltada |
| 09–11 | `09-mlflow-logged-models`, `10/11-mlflow-registry…` | 42 modelos, 7 versiones, `@champion` = v7 y etiquetas de comparación |
| 12–14 | `12-api-docs`, `13-streamlit`, `14-inference-log` | API, UI e inferencias registradas |
| 15 | `15-prometheus-targets` | cadvisor, pods y prometheus **UP** |
| 16–20 | `…-1rep` | Primera prueba: techo de ~8 req/s y el defecto de los probes |
| 21–26 | `21-locust-levels-fixed` … `26-grafana-all-pods-3rep` | Después de la corrección: niveles, 3 réplicas, salida del modelo |
| 27–30 | `27-vmstat-3rep` … `30-locust-latency-users-all-levels` | CPU de la VM saturada; 50 usuarios con 3 réplicas |

### Referencias

- Strack, B. et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis
  of 70,000 Clinical Database Patient Records.* BioMed Research International. PMC3996476.
- Dataset: *Diabetes 130-US hospitals for years 1999–2008*, UCI Machine Learning Repository.
