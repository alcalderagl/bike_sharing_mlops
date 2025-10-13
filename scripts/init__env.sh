#!/bin/bash
set -e

echo "=== Inicializando entorno de trabajo DVC + Google Cloud ==="

# 1. Crear entorno virtual si no existe
if [ ! -d ".venv" ]; then
  echo "Creando entorno virtual..."
  python3 -m venv .venv
fi

# 2. Activar entorno virtual
source .venv/bin/activate

# 3. Instalar dependencias si no están
echo "Instalando dependencias necesarias..."
pip install --upgrade pip
pip install dvc dvc-gs

# 4. Autenticación con Google Cloud
echo "Autenticación con Google Cloud (se abrirá una ventana del navegador)..."
if gcloud auth application-default login; then
  echo "Autenticación completada correctamente."
else
  echo "Error en la autenticación. Intenta ejecutar 'gcloud auth application-default login' manualmente."
  exit 1
fi

# 5. Descargar los datos versionados con DVC
echo "Descargando datos versionados con DVC..."
if dvc pull; then
  echo "Datos descargados correctamente."
else
  echo "Error al ejecutar 'dvc pull'. Verifica la configuración del remoto."
  exit 1
fi

echo "=== Entorno configurado correctamente ==="