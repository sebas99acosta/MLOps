# Taller Desarrollo en Contenedores

Entorno de desarrollo con **Docker Compose**: un servicio de **JupyterLab**
(instalado con `uv`) para entrenar modelos, y un **API con FastAPI** para
servir inferencia. Ambos comparten un volumen (`models/`) — cuando el
notebook guarda un modelo nuevo, el API lo detecta automáticamente (con
`watchdog`) y lo deja disponible para predecir, **sin reiniciar nada**.

## Arquitectura

```
┌─────────────────────────┐        ┌─────────────────────────┐
│   jupyter (puerto 8888) │        │    api (puerto 8025)     │
│   entrena y guarda .pkl │        │   sirve /predict         │
└────────────┬─────────────┘        └────────────┬─────────────┘
             │  escribe                           │  observa (watchdog)
             └──────────────┐       ┌─────────────┘
                             ▼       ▼
                      volumen compartido: ./models
```

## Estructura de carpetas

```
Taller_contenedores/
├── docker-compose.yml
├── jupyter/          # Dockerfile + dependencias (uv) del servicio de entrenamiento
├── api/              # Dockerfile + dependencias (uv) + main.py del API
├── data/             # penguins.csv — solo lo usa jupyter
├── notebooks/        # train.ipynb — se edita desde JupyterLab
└── models/           # volumen compartido: aquí aparecen los .pkl entrenados
```

## Requisitos (Linux)

- Docker Engine y el plugin de Docker Compose (comando `docker compose`,
  con espacio — no el antiguo `docker-compose`).

Si no los tienes instalados:

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

Cierra sesión y vuelve a entrar (o `newgrp docker`) para que el cambio de
grupo tome efecto sin necesitar `sudo` en cada comando `docker`.

Verifica que todo esté disponible:

```bash
docker --version
docker compose version
```

## Cómo correr el proyecto

Ubícate en la carpeta del proyecto:

```bash
cd Taller_contenedores
```

### 1. Construir las imágenes

```bash
docker compose build
```

### 2. Levantar los dos servicios

```bash
docker compose up -d
```

### 3. Verificar que ambos estén corriendo

```bash
docker compose ps
```

### 4. Confirmar que el API arranca vacío (aún no hay modelos)

```bash
curl -s http://localhost:8025/
```

Debe responder `"current_model": null` y `"available_models": []`.

### 5. Entrenar los modelos

**Opción A — desde la interfaz de JupyterLab:** abre
`http://localhost:8888/lab` en el navegador, abre `notebooks/train.ipynb`
y corre todas las celdas.

**Opción B — desde la terminal, sin abrir el navegador:**

```bash
docker compose exec jupyter jupyter nbconvert --to notebook --execute --inplace /app/notebooks/train.ipynb
```

### 6. Confirmar que el API detectó los modelos solo

```bash
curl -s http://localhost:8025/models
```

Debe listar los 4 modelos con su accuracy, sin haber reiniciado el
contenedor del API. Para ver la recarga automática en los logs:

```bash
docker compose logs api | grep registry
```

### 7. Probar inferencia

```bash
curl -s -X POST http://localhost:8025/predict \
  -H "Content-Type: application/json" \
  -d '{"island":"Torgersen","bill_length_mm":39.1,"bill_depth_mm":18.7,"flipper_length_mm":181,"body_mass_g":3750,"sex":"male"}'
```

### 8. Cambiar el modelo activo (bono)

```bash
curl -s -X POST http://localhost:8025/select-model \
  -H "Content-Type: application/json" \
  -d '{"model_name":"knn"}'
```

Repite el `POST /predict` del paso 7 y observa que `model_used` ahora
dice `"knn"`.

### 9. Apagar todo

```bash
docker compose down
```

## Endpoints del API

| Método | Ruta             | Qué hace                                              |
|--------|------------------|--------------------------------------------------------|
| GET    | `/`              | Info general y modelo actualmente seleccionado.        |
| GET    | `/models`        | Lista los modelos disponibles y su accuracy.            |
| POST   | `/select-model`  | Cambia cuál modelo se usa para inferencia (bono).       |
| POST   | `/predict`       | Predice la especie con el modelo activo.                |
| POST   | `/reload-models` | Fuerza una recarga manual (watchdog ya lo hace solo).   |

---

## Glosario de comandos

Explicación de cada comando y bandera (`flag`) usado arriba.

### `docker compose build`

Construye las imágenes definidas en `docker-compose.yml` (una por cada
servicio con una sección `build:`), siguiendo cada `Dockerfile`.

- Sin banderas: reutiliza capas ya construidas si no cambiaron (más
  rápido, normal en el día a día).
- `--no-cache`: ignora toda caché y reconstruye desde cero. Útil cuando
  quieres probar "de verdad desde cero", como pediste antes.

### `docker compose up`

Crea y arranca los contenedores de todos los servicios.

- **`-d`** = **detached** (desacoplado). Sin `-d`, la terminal queda
  "pegada" mostrando los logs en vivo de los contenedores y se bloquea
  hasta que los detengas con `Ctrl+C` (lo que también los apaga). Con
  `-d`, Docker los arranca en segundo plano y te devuelve la terminal de
  inmediato — por eso lo usamos, para poder seguir corriendo `curl` y
  otros comandos después.

  ⚠️ Ojo: **`-d` significa algo distinto en `curl`** (ver más abajo). Es
  la misma letra, pero cada programa define sus propias banderas —
  no hay relación entre ellas.

### `docker compose ps`

Lista los contenedores del proyecto actual (los definidos en este
`docker-compose.yml`) y su estado (`Up`, `Exited`, etc.), con sus
puertos mapeados.

### `docker compose logs api`

Muestra el log acumulado del servicio `api` (todo lo que ese proceso ha
impreso por `stdout`/`stderr` desde que arrancó). `| grep registry`
filtra solo las líneas que contienen la palabra `registry`, que es lo
que imprime nuestro código cada vez que el watchdog recarga los modelos.

### `docker compose exec jupyter <comando>`

Ejecuta `<comando>` **dentro** del contenedor `jupyter` que ya está
corriendo (no crea uno nuevo). Lo usamos para correr el notebook desde
la terminal sin necesidad de abrir la interfaz web de Jupyter.

### `docker compose down`

Detiene y **elimina** los contenedores y la red creados por
`docker compose up` (no las imágenes ni los volúmenes con bind mount
como `./models`, esos quedan intactos en disco).

### `curl`

Herramienta de línea de comandos para hacer peticiones HTTP.

- **`-s`** = **silent**. Oculta la barra de progreso de `curl`, para que
  solo se vea la respuesta del servidor.
- **`-X POST`**: fuerza el método HTTP a usar (`GET` es el default de
  `curl` si no se especifica nada).
- **`-H "Content-Type: application/json"`**: agrega una cabecera
  (*header*) a la petición, indicándole al servidor que el cuerpo que
  se envía es JSON.
- **`-d '{...}'`** = **data**. El cuerpo (*body*) de la petición — en
  este caso, el JSON que espera el endpoint. **Esta es la otra `-d`**:
  en `curl` significa "datos a enviar", en `docker compose up` significa
  "desacoplado". Coinciden en la letra por casualidad de diseño de cada
  herramienta, no por una convención compartida.

### `jupyter nbconvert --to notebook --execute --inplace <archivo>`

Ejecuta un notebook de principio a fin desde la terminal (como si
apretaras "Run All" en la interfaz web) y guarda los resultados.

- `--to notebook`: el formato de salida es otro `.ipynb` (por defecto
  `nbconvert` se usa para exportar a HTML/PDF/etc., aquí lo queremos de
  vuelta como notebook).
- `--execute`: corre las celdas de código antes de exportar.
- `--inplace`: sobrescribe el mismo archivo de entrada en vez de crear
  uno nuevo con otro nombre.
