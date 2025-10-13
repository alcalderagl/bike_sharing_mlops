# DVC + Google Drive Starter (Maestría)

Plantilla mínima para reproducibilidad con **DVC** y remoto en **Google Drive** (gdrive). Incluye pipeline con etapas:

- `clean_data`
- `prepare_features`
- `train_model`

## Requisitos

- Python 3.10+
- Git
- DVC con soporte Google Drive:
  ```bash
  pip install "dvc[gdrive]"
  ```

---

## Uso rápido

### 1) Clonar y preparar entorno

```bash
git clone <URL-de-tu-repo>
cd <carpeta>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Inicializar DVC (si tu repo no lo trae)

```bash
dvc init
git add .dvc .gitignore
git commit -m "Init DVC"
```

### 3) Crear carpeta en Google Drive y obtener su **Folder ID**

- Crea una carpeta, por ejemplo: `DVC_Remote_Maestria`
- Copia el **ID** de la URL de Drive, similar a: `https://drive.google.com/drive/folders/1AbCDefGhIJkLMn0Pq`
  - El ID sería: `1AbCDefGhIJkLMn0Pq`

### 4) Configurar remoto DVC con Google Drive

```bash
dvc remote add -d storage gdrive://<TU_FOLDER_ID>
git add .dvc/config
git commit -m "Configura remoto DVC (Google Drive)"
```

> La **primera vez** que hagas `dvc push`/`pull` pedirá un flujo OAuth (abre un enlace, pega el código).

### 5) Versionar un dataset de ejemplo

```bash
# Coloca tu archivo en data/raw/dataset.csv
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc
git commit -m "Track dataset con DVC (v1)"
dvc push     # sube los binarios a Drive
git push     # sube metadatos/código a GitHub
```

### 6) Ejecutar el pipeline

```bash
dvc repro
dvc push
git add -A && git commit -m "Actualiza outputs del pipeline" && git push
```

---

## Configuración del entorno

Sigue estos pasos para clonar el repositorio y preparar el entorno de trabajo:

```bash
# Clonar el repositorio
git clone <URL-de-tu-repo>
cd <carpeta>

# Crear y activar un entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install dvc dvc-gs

# Autenticación con Google Cloud
gcloud auth application-default login

# Descargar los datos versionados con DVC
dvc pull
```

# Tambien se puede correr el siguiente script para automatizar la configuracion

```bash
## Dar permisos de ejecución

chmod +x scripts/init_env.sh

## Ejecutar la configuración completa

.scripts/init_env.sh
```

---

## CI/CD en GitHub Actions (opcional)

### Opción A: Autenticación interactiva (no recomendada para CI)

El flujo OAuth de Drive requiere intervención, por lo que en CI no funciona.

### Opción B: Service Account (recomendado para CI)

1. Crea una **Service Account** en Google Cloud y comparte la carpeta de Drive con el email de esa cuenta (o usa Drive de un proyecto).
2. Descarga el **JSON** de la Service Account.
3. En tu repo de GitHub → **Settings → Secrets and variables → Actions**, crea un secreto llamado `GDRIVE_SA_JSON` con el **contenido del JSON**.
4. En tu DVC remote, añade la configuración para usar la Service Account:
   ```bash
   dvc remote modify storage gdrive_service_account_json_file_path .gdrive_sa.json
   ```
   > (Este paso normalmente se aplica en CI **temporalmente** antes de `dvc pull`.)

El workflow incluido escribe el JSON en un archivo temporal `.gdrive_sa.json` y ejecuta `dvc pull`.

---

## Estructura del proyecto

```
src/
  clean_data.py
  prepare_features.py
  train_model.py
data/
  raw/      # datasets de entrada (versionados por DVC)
  clean/    # datos procesados (outs del pipeline)
models/     # modelos entrenados (outs del pipeline)
dvc.yaml    # etapas del pipeline
```

---

## Limpieza y buenas prácticas

- No borres archivos dentro del remoto de Drive manualmente; usa `dvc gc` para limpiar versiones que ya no están referenciadas.
- No subas credenciales a Git. Usa **Secrets** en GitHub Actions.
- Si el repo ya trae DVC configurado, **no reinicialices**: usa `dvc pull` directamente.

¡Listo! 🎯
