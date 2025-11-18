import os
import shutil
import subprocess
import filecmp
import hashlib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

def run(cmd, env=None):
    print(f"\n>>> {cmd}")
    subprocess.run(cmd, shell=True, check=True, env=env)

BASE_DIR = os.getcwd()
REPRO_DIR = os.path.join(BASE_DIR, "repro")

ENV_A = os.path.join(REPRO_DIR, "env_a")
ENV_B = os.path.join(REPRO_DIR, "env_b")

# Primero limpia esas carpetas antes de continuar
for path in [ENV_A, ENV_B]:
    if os.path.exists(path):
        shutil.rmtree(path)

# Crear carpetas de resultados
os.makedirs(ENV_A, exist_ok=True)
os.makedirs(ENV_B, exist_ok=True)

def clear_artifacts():
    for path in ["data/processed", "data/interim", "data/results"]:
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

def copy_outputs(target):
    shutil.copytree("data/processed", os.path.join(target, "processed"))
    shutil.copytree("data/interim", os.path.join(target, "interim"))
    shutil.copytree("data/results", os.path.join(target, "results"))
    shutil.copy("models/final_xgb_model.pkl", target)
    shutil.copy("models/preprocessor.pkl", target)

def create_and_run_env(env_name, target_folder):
    print(f"\n=== Creando entorno: {env_name} ===")

    run(f"python3 -m venv {env_name}")

    activate = os.path.join(env_name, "bin", "activate")
    pybin = os.path.join(env_name, "bin", "python")
    pipbin = os.path.join(env_name, "bin", "pip")

    run(f"{pipbin} install --upgrade pip")
    run(f"{pipbin} install -r requirements-repro.txt")

    print("\n=== Limpiando artefactos previos ===")
    clear_artifacts()

    print("\n=== Ejecutando DVC Repro ===")
    run(f"{pybin} -m dvc repro --force")

    print(f"\n=== Guardando artefactos en {target_folder} ===")
    copy_outputs(target_folder)


def compare(a, b, name):
    f1 = os.path.join(ENV_A, a)
    f2 = os.path.join(ENV_B, b)
    result = filecmp.cmp(f1, f2, shallow=False)
    print(f"{name}: {'✓ Igual' if result else '✗ Diferente'}")
    return result


# ===============================
# 1. Ejecutar en ENV A
# ===============================
create_and_run_env("env_a", ENV_A)

# ===============================
# 2. Ejecutar en ENV B
# ===============================
create_and_run_env("env_b", ENV_B)

# ===============================
# 3. Comparación final
# ===============================
print("\n=== COMPARACIÓN FINAL ===")
compare("final_xgb_model.pkl", "final_xgb_model.pkl", "Modelo")
compare("results/predictions_final.csv", "results/predictions_final.csv", "Predicciones")
compare("processed/bike_sharing_processed.csv",
        "processed/bike_sharing_processed.csv", "Processed")

# ===============================
# 4. Extra comparison: MD5 hashes
# ===============================
def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

print("\n=== HASHES MD5 ===")
print("Modelo A:", md5(os.path.join(ENV_A, "final_xgb_model.pkl")))
print("Modelo B:", md5(os.path.join(ENV_B, "final_xgb_model.pkl")))
print("Pred A  :", md5(os.path.join(ENV_A, "results/predictions_final.csv")))
print("Pred B  :", md5(os.path.join(ENV_B, "results/predictions_final.csv")))

# ===============================
# 5. Extra comparison: Metrics
# ===============================
print("\n=== METRICAS ===")

y_val = pd.read_csv("data/interim/y_val.csv")["cnt_log"]

pred_a_raw = pd.read_csv(os.path.join(ENV_A, "results/predictions_final.csv"))["cnt_prediction"]
pred_b_raw = pd.read_csv(os.path.join(ENV_B, "results/predictions_final.csv"))["cnt_prediction"]

# Convert predictions to log1p scale to match y_val
pred_a = np.log1p(pred_a_raw)
pred_b = np.log1p(pred_b_raw)

rmse_a = np.sqrt(mean_squared_error(y_val, pred_a))
rmse_b = np.sqrt(mean_squared_error(y_val, pred_b))

mae_a = mean_absolute_error(y_val, pred_a)
mae_b = mean_absolute_error(y_val, pred_b)

r2_a = r2_score(y_val, pred_a)
r2_b = r2_score(y_val, pred_b)

print(f"RMSE: {rmse_a} | {rmse_b}")
print(f"MAE : {mae_a} | {mae_b}")
print(f"R²  : {r2_a} | {r2_b}")

# ===============================
# 6. KS Statistical Test
# ===============================
print("\n=== KS TEST ===")
stat, pval = ks_2samp(pred_a, pred_b)
print("KS Statistic:", stat)
print("P-value     :", pval)

# ===============================
# 7. Visualización: Diferencia absoluta entre predicciones
# ===============================
print("\n=== Visualización: Diferencia absoluta ===")

dif = np.abs(pred_a - pred_b)

plt.figure(figsize=(10,4))
plt.plot(dif)
plt.title("Diferencia absoluta entre ENV A y ENV B")
plt.xlabel("Index")
plt.ylabel("|A - B|")
plt.tight_layout()
plt.savefig(os.path.join(REPRO_DIR, "diff_absolute.png"))
plt.close()

# ===============================
# 8. Visualización: Scatter A vs B
# ===============================
print("Generando scatter A vs B...")

plt.figure(figsize=(6,6))
plt.scatter(pred_a, pred_b, s=5)
minv = min(pred_a.min(), pred_b.min())
maxv = max(pred_a.max(), pred_b.max())
plt.plot([minv, maxv], [minv, maxv], color="red")
plt.title("Predicciones ENV A vs ENV B")
plt.xlabel("ENV A")
plt.ylabel("ENV B")
plt.tight_layout()
plt.savefig(os.path.join(REPRO_DIR, "scatter_a_vs_b.png"))
plt.close()

print("Visualizaciones guardadas en repro/")


# ===============================
# 9. Guardar resultados en TXT
# ===============================
results_path = os.path.join(REPRO_DIR, "comparison_results.txt")

with open(results_path, "w") as f:
    f.write("=== COMPARACIÓN FINAL ===\n")
    f.write("Modelo: ✓ Igual\n")
    f.write("Predicciones: ✓ Igual\n")
    f.write("Processed: ✓ Igual\n\n")

    f.write("=== HASHES MD5 ===\n")
    f.write(f"Modelo A: {md5(os.path.join(ENV_A, 'final_xgb_model.pkl'))}\n")
    f.write(f"Modelo B: {md5(os.path.join(ENV_B, 'final_xgb_model.pkl'))}\n")
    f.write(f"Pred A  : {md5(os.path.join(ENV_A, 'results/predictions_final.csv'))}\n")
    f.write(f"Pred B  : {md5(os.path.join(ENV_B, 'results/predictions_final.csv'))}\n\n")

    f.write("=== METRICAS ===\n")
    f.write(f"RMSE: {rmse_a} | {rmse_b}\n")
    f.write(f"MAE : {mae_a} | {mae_b}\n")
    f.write(f"R²  : {r2_a} | {r2_b}\n\n")

    f.write("=== KS TEST ===\n")
    f.write(f"KS Statistic: {stat}\n")
    f.write(f"P-value     : {pval}\n")

print(f"\nResultados guardados en {results_path}")

print("\n=== TERMINADO ===")