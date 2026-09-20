# Taller Airflow

Pipeline completo de MLOps orquestado con Airflow: dos bases de datos
separadas (MySQL para los datos, Postgres para los metadatos de Airflow),
un DAG con 4 tareas encadenadas (borrar → cargar → preprocesar → entrenar),
y un API de inferencia con selección de modelo — todo en un solo
`docker-compose.yml`.

## Arquitectura

```
┌──────────────┐     ┌──────────────┐
│  MySQL       │     │  Postgres    │
│  (datos)     │     │  (metadatos  │
│              │     │   de Airflow)│
└──────┬───────┘     └──────┬───────┘
       │                    │
       │             ┌──────┴───────────────────┐
       │             │  airflow-init (una vez)   │
       │             │  airflow-scheduler         │
       │◄────────────┤  airflow-webserver         │
       │  DAG lee/    └──────┬────────────────────┘
       │  escribe            │ escribe modelos
       │                     ▼
       │              ┌──────────────┐
       │              │  ./models    │  (volumen compartido)
       │              └──────┬───────┘
       │                     │ lee
       │              ┌──────▼───────┐
       └─────────────►│  api         │  (inferencia)
                       └──────────────┘
```

## Estructura de carpetas

```
Taller_Airflow/
├── docker-compose.yml
├── .env.example        # plantilla — copiar a .env
├── airflow/             # Dockerfile común a init/scheduler/webserver
├── api/                 # Dockerfile + main.py del API de inferencia
├── dags/
│   └── penguins_dag.py  # el DAG: delete → load → preprocess → train
├── data/
│   └── penguins.csv
├── models/               # volumen compartido: aquí caen los .pkl entrenados
└── logs/                 # logs de Airflow
```

## Requisitos (Linux)

- Docker Engine + plugin de Docker Compose (`docker compose`, con espacio).
  Si no los tienes:
  ```bash
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker $USER
  ```
  Cierra sesión y vuelve a entrar (o `newgrp docker`) para no necesitar
  `sudo` en cada comando.

> **Nota sobre el `docker-compose.yaml` oficial de Airflow:** si buscas en
> la documentación de Airflow vas a encontrar un `docker-compose.yaml` de
> referencia (`curl -LfO '.../docker-compose.yaml'`) con Postgres +
> Redis + Celery + varios workers. **No lo necesitas aquí** — este
> proyecto ya tiene su propio `docker-compose.yml`, construido a medida
> con `LocalExecutor` (más simple, suficiente para 4 tareas secuenciales)
> y las dos bases de datos que pide el enunciado. No lo descargues ni lo
> mezcles con el de este repo.

## Setup en la VM (Linux)

### 1. Clonar y ubicarte en la carpeta

```bash

cd MLOps/Nivel_2/Taller_Airflow
```

### 2. Crear tu `.env`

```bash
cp .env.example .env
```

### 3. Ajustar `AIRFLOW_UID` a tu usuario real

La documentación oficial de Airflow sugiere:
```bash
echo -e "AIRFLOW_UID=$(id -u)" > .env
```
**No uses ese comando tal cual aquí** — sobrescribiría todo el archivo y
perderías las credenciales de Postgres/MySQL que también viven en
`.env`. En su lugar, reemplaza solo esa línea:
```bash
sed -i "s/^AIRFLOW_UID=.*/AIRFLOW_UID=$(id -u)/" .env
```
Por qué importa: si `AIRFLOW_UID` no coincide con tu usuario de Linux,
los archivos que Airflow escribe en `./dags`, `./logs` y `./models`
quedan con otro dueño, y vas a necesitar `sudo` hasta para borrarlos.

### 4. Carpetas de volúmenes

`dags/`, `data/`, `models/` y `logs/` ya existen en el repo (los dos
primeros con contenido real, los otros dos vacíos vía `.gitkeep`), así
que no hace falta el `mkdir -p` que sugiere la guía oficial de Airflow.
Si por algún motivo no existieran:
```bash
mkdir -p ./dags ./logs ./models
```

### 5. Construir las imágenes

```bash
docker compose build
```

### 6. Levantar todo

```bash
docker compose up -d
docker compose ps
```

Espera a que `postgres-airflow` y `mysql-data` muestren `healthy`, y que
`airflow-init` termine en `Exited (0)` — ese es su comportamiento
correcto, no un error (ver README de `Taller_contenedores` si quieres
el repaso completo de por qué).

### 7. Activar y disparar el DAG

Por defecto, todo DAG nuevo arranca **pausado**. Actívalo y dispáralo:

```bash
docker compose exec airflow-scheduler airflow dags unpause penguins_pipeline
docker compose exec airflow-scheduler airflow dags trigger penguins_pipeline
```

O hazlo desde la UI: `http://<ip-de-la-vm>:8010` (usuario `admin`,
contraseña `admin`) — activa el toggle junto al DAG y usa el botón ▶.

Revisa el progreso:
```bash
docker compose exec airflow-scheduler airflow dags list-runs -d penguins_pipeline
```

### 8. Confirmar que se entrenaron los modelos

```bash
ls -la models/
cat models/metrics.json
```

Deberías ver los 4 `.pkl`, `encoders.pkl` y `metrics.json` con la
accuracy de cada modelo.

### 9. Probar el API de inferencia

```bash
curl http://localhost:8025/
curl http://localhost:8025/models
curl -X POST http://localhost:8025/predict \
  -H "Content-Type: application/json" \
  -d '{"island":"Torgersen","bill_length_mm":39.1,"bill_depth_mm":18.7,"flipper_length_mm":181,"body_mass_g":3750,"sex":"male"}'
```

Para elegir el modelo con el menú desplegable (no a mano por string):
abre `http://<ip-de-la-vm>:8025/docs`, despliega `POST /select-model`,
"Try it out" — vas a ver un `<select>` con las 4 opciones, no una caja
de texto libre.

### 10. Apagar

```bash
docker compose down
```

Los datos de las bases (volúmenes con nombre) y los modelos entrenados
(`./models`, bind mount) sobreviven a esto — es apagar, no borrar.

## Reiniciar desde cero de verdad

Si quieres repetir la demo completa desde un estado limpio (útil para
verificar que "Etapa 1: borrar contenido" realmente funciona en la
primerísima corrida):

```bash
docker compose down -v          # -v también borra los volúmenes de Postgres/MySQL
rm -f models/*.pkl models/*.json
rm -rf logs/*
docker compose up -d
```

## Endpoints del API

| Método | Ruta             | Qué hace                                              |
|--------|------------------|----------------------------------------------------------|
| GET    | `/`              | Info general y modelo actualmente seleccionado.           |
| GET    | `/models`        | Lista los modelos disponibles y su accuracy.                |
| POST   | `/select-model`  | Cambia el modelo activo — `?model_name=...` (dropdown en `/docs`). |
| POST   | `/predict`       | Predice la especie con el modelo activo.                     |

## El DAG (`dags/penguins_dag.py`)

4 tareas encadenadas con `>>`:

1. **`delete_content`** — `DROP TABLE IF EXISTS` sobre ambas tablas. Corre
   primero siempre, y es seguro correrlo aunque las tablas no existan
   (primera corrida) o ya estén vacías (reintentos).
2. **`load_data`** — carga `penguins.csv` a `penguins_raw` tal cual,
   sin limpiar nada (nulos incluidos).
3. **`preprocess`** — limpia nulos, codifica `island`/`sex`, guarda en
   `penguins_processed` y persiste los encoders en `models/encoders.pkl`.
4. **`train_model`** — entrena 4 modelos candidatos sobre los datos
   preprocesados, guarda cada uno y un `metrics.json` con la accuracy de
   todos y cuál es el mejor.

Todas las tareas usan `MySqlHook` con la conexión `mysql_penguins`,
definida vía la variable de entorno `AIRFLOW_CONN_MYSQL_PENGUINS` en
`docker-compose.yml` (no hay que crearla a mano en la UI).

## Glosario de comandos nuevos

(Para `docker compose up -d`, `curl -X POST`, etc. ver el glosario ya
armado en el README de `Taller_contenedores` — no se repite aquí.)

- **`id -u`** — imprime el UID numérico de tu usuario actual en Linux.
  Es lo que usamos para que los contenedores de Airflow escriban
  archivos con el dueño correcto en vez de otro usuario/root.
- **`sed -i "s/^AIRFLOW_UID=.*/AIRFLOW_UID=$(id -u)/" .env`** — edita el
  archivo *en el lugar* (`-i`), reemplazando solo la línea que empieza
  con `AIRFLOW_UID=` por una nueva con tu UID real. El resto del
  archivo (credenciales de las bases) queda intacto.
- **`airflow dags unpause <dag_id>`** — un DAG nuevo arranca pausado por
  defecto (protección para no disparar algo por accidente); este
  comando lo activa para que pueda correr.
- **`airflow dags trigger <dag_id>`** — dispara una corrida manual del
  DAG ahora mismo, sin esperar a un schedule (que en este proyecto ni
  siquiera existe — `schedule=None`).
- **`airflow dags list-runs -d <dag_id>`** — lista las corridas de un
  DAG y su estado (`running`, `success`, `failed`), útil para revisar
  progreso desde la terminal sin abrir la UI.
- **`docker compose down -v`** — igual que `down`, pero además borra los
  volúmenes con nombre (`postgres_airflow_data`, `mysql_data`) — un
  reinicio total de las bases de datos, no solo de los contenedores.
