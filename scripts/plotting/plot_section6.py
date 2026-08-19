"""
Última Fecha de Modificación: 08/Aug/2026
Descripción plot_section6.py: Script que produce la evaluación y las figuras de
la Sección 6 de la memoria, dedicada a la validación crítica del sistema
híbrido PINN+DRL. Cubre cuatro análisis complementarios; cada uno se omite con
un aviso si le falta algún checkpoint, sin abortar el resto:

    1. Ablation study empírico del PINN:
       PINN completo vs. sin constraint layers vs. sin física. Los tres
       checkpoints se han entrenado sobre el mismo corpus con la misma
       configuración de optimización, aislando el efecto de cada componente
       sobre la fidelidad al simulador T-MATS.

    2. Pruebas Out-of-Distribution del PINN:
       Verifica la coherencia física de las predicciones para condiciones
       fuera del envolvente de entrenamiento.

    3. Análisis de sensibilidad de la recompensa DRL:
       Evalúa el agente TD3 tabula rasa bajo cinco configuraciones distintas
       de pesos de recompensa para verificar la robustez de la política
       aprendida ante perturbaciones de la calibración heurística de los
       coeficientes.

    4. Perfilado SWaP-C (Size, Weight, Power, Cost):
       Latencia de inferencia por percentiles y escalabilidad por batch
       para evaluar viabilidad de despliegue Edge.

Los resultados se guardan en `results/section_results/RESULTADOS_SECCION_6.json`
y generan cuatro figuras (fig24, fig25, fig26, fig27) en `results/figures/`.

Uso:
    python scripts/plotting/plot_section6.py
"""

import json
import logging
import sys
import time
from pathlib import Path
import os

# UTF-8 portable
if sys.platform == 'win32':
    os.system('')
    try:
        os.system('chcp 65001 >nul 2>&1')
    except Exception:
        pass

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pinn import BraytonPINN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger(__name__)


# Rutas y constantes
PINN_CHECKPOINT_FULL = Path('checkpoints/digital_twin/pinn.pt')
PINN_CHECKPOINT_NO_CL = Path('checkpoints/digital_twin/pinn_no_cl.pt')
PINN_CHECKPOINT_NO_PHYSICS = Path('checkpoints/digital_twin/pinn_no_physics.pt')

TD3_CHECKPOINT = Path('checkpoints/control/td3_mixed/best_model.zip')

CORPUS_PATH = Path('data/synthetic/ace_dataset_5000.csv')
FIG_DIR = Path('results/figures')
RESULTS_JSON = Path('results/section_results/RESULTADOS_SECCION_6.json')
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)

# Configuración de latency profiling.
LATENCY_WARMUP = 200
LATENCY_ITERATIONS = 2000
BATCH_SIZES = [1, 8, 32, 64, 128]
BATCH_ITERATIONS = 500

# Umbrales de compatibilidad con perfiles DO-178C.
# Level A (5 ms):  funciones cuyo fallo tiene efecto catastrófico
# Level B (10 ms): funciones cuyo fallo tiene efecto peligroso
# Level C (20 ms): funciones cuyo fallo tiene efecto mayor
DO178C_LEVEL_C_MS = 20.0
DO178C_LEVEL_B_MS = 10.0
DO178C_LEVEL_A_MS = 5.0

# Configuración del análisis de sensibilidad.
SENSITIVITY_EPISODES = 10
SENSITIVITY_MAX_STEPS = 200

# Estilo científico.
plt.rcParams.update({
    'figure.dpi':        150,
    'font.size':         12,
    'font.family':       'serif',
    'axes.titlesize':    13,
    'axes.labelsize':    12,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


# Utilidades de carga
def detect_pinn_architecture(state_dict: dict) -> tuple[int, int]:
    """Deduce la dimensión oculta y el número de bloques a partir del state_dict.

    Es especialmente necesario en este script, que carga tres checkpoints distintos:
    el modelo completo y sus dos variantes ablacionadas, que no tienen por qué
    compartir configuración si se reentrenaron por separado.

    param state_dict: Diccionario de pesos del checkpoint.
    return: Tupla (hidden_dim, n_layers) deducida, con los valores por defecto 128 y
        6 si las claves esperadas no aparecen.
    """
    proj = state_dict.get('input_proj.0.weight')
    hidden_dim = proj.shape[0] if proj is not None else 128

    residual_blocks = {
        int(k.split('residual_blocks.')[1].split('.')[0])
        for k in state_dict if 'residual_blocks' in k
    }
    n_layers = max(residual_blocks) + 1 if residual_blocks else 6
    return hidden_dim, n_layers


def load_full_pinn(checkpoint_path: Path) -> BraytonPINN:
    """Carga un checkpoint con la arquitectura completa del gemelo digital.

    Sirve tanto para el modelo de referencia como para la variante entrenada sin
    término físico, ya que ambas comparten arquitectura y solo difieren en la pérdida
    con la que se optimizaron: por eso una única función cubre los dos casos.

    param checkpoint_path: Ruta del archivo de pesos a cargar.
    return: Modelo BraytonPINN con los pesos cargados y en modo evaluación.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu',
                      weights_only=False)
    state_dict = ckpt if isinstance(ckpt, dict) and \
        'input_proj.0.weight' in ckpt else \
        ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))

    hidden_dim, n_layers = detect_pinn_architecture(state_dict)
    model = BraytonPINN(hidden_dim=hidden_dim, n_layers=n_layers)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_no_cl_pinn(checkpoint_path: Path):
    """Carga la variante ablacionada que carece de constraint layers.

    Necesita una función propia porque su cabeza de salida tiene 14 neuronas en lugar
    de 11: instanciarla con la clase del modelo completo produciría una discrepancia de
    formas y, con carga no estricta, un modelo silenciosamente mal inicializado.

    param checkpoint_path: Ruta del archivo de pesos a cargar.
    return: Modelo BraytonPINNNoCL con los pesos cargados y en modo evaluación.
    """
    from src.models.pinn_ablated import BraytonPINNNoCL

    ckpt = torch.load(checkpoint_path, map_location='cpu',
                      weights_only=False)
    state_dict = ckpt if isinstance(ckpt, dict) and \
        'input_proj.0.weight' in ckpt else \
        ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))

    hidden_dim, n_layers = detect_pinn_architecture(state_dict)
    model = BraytonPINNNoCL(hidden_dim=hidden_dim, n_layers=n_layers)
    # strict=False por coherencia con eval_pinn.py y plot_drl_section5.py.
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def compute_mape_per_variable(y_true: np.ndarray, y_pred: np.ndarray,
                              variable_names: list) -> dict:
    """Calcula el error porcentual medio de cada variable, con protección numérica.

    Acota el denominador por abajo en lugar de descartar muestras, de modo que todas
    las configuraciones del estudio de ablación se midan exactamente sobre el mismo
    número de puntos y sus errores sean directamente comparables.

    param y_true: Matriz de valores de referencia del corpus.
    param y_pred: Matriz de valores predichos por el modelo.
    param variable_names: Nombres de las columnas, en el orden de las matrices.
    return: Diccionario {variable: MAPE en porcentaje}, omitiendo las columnas que el
        modelo no llega a predecir.
    """
    mapes = {}
    for i, name in enumerate(variable_names):
        if i >= y_pred.shape[1]:
            continue
        errors = np.abs(y_true[:, i] - y_pred[:, i])
        denominators = np.clip(np.abs(y_true[:, i]), 1e-8, None)
        mapes[name] = float(np.mean(errors / denominators) * 100)
    return mapes


# 1. Ablation study (fig24)
def evaluate_ablation() -> dict:
    """Evalúa las tres configuraciones del estudio de ablación sobre el corpus.

    Mide el error de la configuración completa, de la que carece de constraint layers
    y de la entrenada sin física, usando el mismo corpus y el mismo cálculo para las
    tres. Es lo que permite cuantificar por separado la aportación de cada ingrediente
    del gemelo digital, que es la pregunta central de la Sección 6. Las variantes
    ausentes se avisan indicando qué script debe ejecutarse para generarlas.

    return: Diccionario {configuración: {variable: MAPE}}, vacío si no existe el
        corpus, y con solo las configuraciones cuyo checkpoint esté disponible.
    """
    logger.info("Ablation study empírico del PINN")

    if not CORPUS_PATH.exists():
        logger.error(f"Corpus no encontrado: {CORPUS_PATH}")
        return {}

    df = pd.read_csv(CORPUS_PATH)
    X = df[BraytonPINN.INPUT_FEATURES].values.astype(np.float32)
    y_cols = [c for c in BraytonPINN.OUTPUT_FEATURES if c in df.columns]
    y_true = df[y_cols].values.astype(np.float32)
    X_tensor = torch.tensor(X)

    results = {}

    if PINN_CHECKPOINT_FULL.exists():
        logger.info(f"  A) PINN completo: {PINN_CHECKPOINT_FULL}")
        model = load_full_pinn(PINN_CHECKPOINT_FULL)
        with torch.no_grad():
            y_pred = model(X_tensor).numpy()
        results['PINN_completo'] = compute_mape_per_variable(
            y_true, y_pred, y_cols)
        logger.info(f"     MAPE medio: "
                    f"{np.mean(list(results['PINN_completo'].values())):.2f}%")
    else:
        logger.warning(f"  A) Checkpoint no encontrado: "
                       f"{PINN_CHECKPOINT_FULL}")

    if PINN_CHECKPOINT_NO_CL.exists():
        logger.info(f"  B) Sin constraint layers: {PINN_CHECKPOINT_NO_CL}")
        model = load_no_cl_pinn(PINN_CHECKPOINT_NO_CL)
        with torch.no_grad():
            y_pred = model(X_tensor).numpy()
        results['Sin_Constraint_Layers'] = compute_mape_per_variable(
            y_true, y_pred, y_cols)
        logger.info(f"     MAPE medio: "
                    f"{np.mean(list(results['Sin_Constraint_Layers'].values())):.2f}%")
    else:
        logger.warning(f"  B) Checkpoint no encontrado: "
                       f"{PINN_CHECKPOINT_NO_CL}. "
                       "Ejecutar train_pinn_ablation.py")

    if PINN_CHECKPOINT_NO_PHYSICS.exists():
        logger.info(f"  C) Sin PDEs: {PINN_CHECKPOINT_NO_PHYSICS}")
        model = load_full_pinn(PINN_CHECKPOINT_NO_PHYSICS)
        with torch.no_grad():
            y_pred = model(X_tensor).numpy()
        results['Sin_PDEs'] = compute_mape_per_variable(
            y_true, y_pred, y_cols)
        logger.info(f"     MAPE medio: "
                    f"{np.mean(list(results['Sin_PDEs'].values())):.2f}%")
    else:
        logger.warning(f"  C) Checkpoint no encontrado: "
                       f"{PINN_CHECKPOINT_NO_PHYSICS}. "
                       "Ejecutar train_pinn_ablation.py")

    return results


def fig24_ablation_study(results: dict, output_path: Path) -> None:
    """Dibuja el error por variable de las tres configuraciones del ablation.

    Barras agrupadas con los umbrales de calidad marcados como líneas de referencia.
    El eje se satura y los valores que lo desbordan se anotan como texto: sin ese
    recorte, el error desmedido de alguna variable en las configuraciones degradadas
    aplastaría visualmente el resto de la figura y la haría ilegible, que es
    justamente lo que se quiere comparar.

    param results: Diccionario {configuración: {variable: MAPE}} del ablation.
    param output_path: Ruta del archivo PNG a generar.
    return: None; la figura queda escrita en output_path, o se omite si no hay datos.
    """
    if not results:
        logger.warning("Sin datos de ablation; se omite fig24")
        return

    variables = ['thrust_approx', 'sfc', 'T2', 'T4', 'T30', 'T50',
                 'P2', 'P30', 'farB', 'wf']
    var_labels = ['$F$', 'SFC', '$T_2$', '$T_4$', '$T_{30}$', '$T_{50}$',
                  '$P_2$', '$P_{30}$', '$farB$', '$w_f$']

    configs = ['PINN_completo', 'Sin_Constraint_Layers', 'Sin_PDEs']
    config_labels = ['PINN completo\n(6 PDEs + 3 CLs)',
                     'Sin Constraint\nLayers',
                     'Sin PDEs\n(red pura)']
    colors = ['#2A9D8F', '#E9C46A', '#E63946']

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(variables))
    width = 0.25
    y_cap = 50.0

    for i, (cfg, label, color) in enumerate(
            zip(configs, config_labels, colors)):
        if cfg not in results:
            continue
        values = [results[cfg].get(v, 0) for v in variables]
        capped = [min(v, y_cap) for v in values]
        bars = ax.bar(x + i * width, capped, width,
                      label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            if val > y_cap:
                ax.text(bar.get_x() + bar.get_width() / 2, y_cap + 1,
                        f'{val:.0f}%', ha='center', fontsize=8, rotation=45)

    ax.set_xlabel('Variable')
    ax.set_ylabel('MAPE [%]')
    ax.set_xticks(x + width)
    ax.set_xticklabels(var_labels)
    ax.axhline(y=5, color='green', linestyle=':', alpha=0.5,
               label='5% (excepcional)')
    ax.axhline(y=15, color='orange', linestyle=':', alpha=0.5,
               label='15% (aceptable)')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, y_cap + 5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {output_path}")


# 2. Pruebas Out-of-Distribution (fig26)
# Cada caso: [altitude_ft, mach, TRA_pct, bpr_ts]
# In-distribution: dentro del envelope de entrenamiento [0-42000 ft, 0.1-1.8, 30-100 %, 0.05-0.50]
IN_DIST_CASES = {
    'Crucero nominal':  [35000, 0.85, 80,  0.30],
    'Combate nominal':  [15000, 1.50, 100, 0.10],
    'Takeoff nominal':  [0,     0.30, 100, 0.20],
}

# Out-of-distribution: violan al menos una variable del envelope
OOD_CASES = {
    'Altitud 55 kft':          [55000, 0.85, 80,  0.30],
    'Mach 2.2':                [30000, 2.20, 80,  0.30],
    'TRA 10 %':                [35000, 0.85, 10,  0.30],
    'TRA 110 %':               [35000, 0.85, 110, 0.30],
    'Combo extremo':           [50000, 2.00, 100, 0.60],
    'Nivel del mar, subsonico': [0,    0.05, 50,  0.50],
}


def evaluate_ood() -> tuple:
    """Consulta el gemelo digital dentro y fuera de su envelope de entrenamiento.

    Evalúa casos nominales y casos deliberadamente extremos, comprobando en estos
    últimos si se mantienen las desigualdades básicas del ciclo y el signo de las
    magnitudes. Es necesario porque el error medio sobre el corpus no dice nada sobre
    qué hace el modelo ante una condición que nunca vio: en un sistema de control,
    extrapolar de forma físicamente absurda es un fallo mucho más grave que un error
    porcentual algo mayor.

    return: Tupla (predicciones de los casos dentro del envelope, predicciones de los
        casos fuera de él), ambas vacías si el checkpoint no está disponible.
    """
    logger.info("Pruebas Out-of-Distribution del PINN")

    if not PINN_CHECKPOINT_FULL.exists():
        logger.warning(f"PINN no disponible: {PINN_CHECKPOINT_FULL}")
        return {}, {}

    model = load_full_pinn(PINN_CHECKPOINT_FULL)
    in_dist = {name: model.predict(*cond)
               for name, cond in IN_DIST_CASES.items()}
    ood = {name: model.predict(*cond)
           for name, cond in OOD_CASES.items()}

    logger.info("  Condiciones In-Distribution:")
    for name, pred in in_dist.items():
        logger.info(f"    {name}: thrust = {pred['thrust_approx']:.0f}, "
                    f"SFC = {pred['sfc']:.3f}, T4 = {pred['T4']:.0f}")

    logger.info("  Condiciones Out-of-Distribution:")
    for name, pred in ood.items():
        coherent = (pred['T4'] > pred['T30'] > pred['T2']
                    and pred['thrust_approx'] > 0
                    and pred['sfc'] > 0)
        status = "coherente" if coherent else "incoherente"
        logger.info(f"    {name}: thrust = {pred['thrust_approx']:.0f}, "
                    f"SFC = {pred['sfc']:.3f}, T4 = {pred['T4']:.0f} "
                    f"[{status}]")

    return in_dist, ood


def fig26_ood_tests(results_in: dict, results_ood: dict,
                    output_path: Path) -> None:
    """Dibuja la comparación entre casos dentro y fuera del envelope.

    Tres paneles, uno por magnitud, con los casos ordenados y una línea divisoria
    entre ambos grupos. La separación visual es lo que da sentido a la figura: permite
    ver de un vistazo si las predicciones fuera del envelope se mantienen en el mismo
    orden de magnitud que las nominales o si se disparan hacia valores imposibles.

    param results_in: Predicciones de los casos dentro del envelope.
    param results_ood: Predicciones de los casos fuera del envelope.
    param output_path: Ruta del archivo PNG a generar.
    return: None; la figura queda escrita en output_path, o se omite si no hay datos.
    """
    if not results_in and not results_ood:
        logger.warning("Sin datos OOD; se omite fig26")
        return

    all_names = list(results_in.keys()) + list(results_ood.keys())
    all_results = {**results_in, **results_ood}
    n_in = len(results_in)
    colors = ['#2A9D8F'] * n_in + ['#E63946'] * len(results_ood)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plots = [
        ('thrust_approx', '$F$ [lbf]'),
        ('sfc',           'SFC [lbm/(lbf·h)]'),
        ('T4',            '$T_4$ [°R]'),
    ]

    for ax, (variable, ylabel) in zip(axes, plots):
        values = [all_results[n].get(variable, 0) for n in all_names]
        ax.bar(range(len(all_names)), values, color=colors, alpha=0.85)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(all_names)))
        ax.set_xticklabels(all_names, rotation=45, ha='right', fontsize=8)
        ax.axvline(x=n_in - 0.5, color='black', linestyle='--',
                   linewidth=2, alpha=0.5)
        y_top = ax.get_ylim()[1] * 0.95
        ax.text(n_in / 2 - 0.5, y_top, 'In-Dist',
                ha='center', fontsize=10, color='#2A9D8F',
                fontweight='bold')
        ax.text(n_in + len(results_ood) / 2 - 0.5, y_top, 'OOD',
                ha='center', fontsize=10, color='#E63946',
                fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {output_path}")


# 3. Analisis de sensibilidad DRL (fig25)
# Configuraciones de pesos de la funcion de recompensa.
# 'Nominal' = pesos usados durante el entrenamiento del agente activo
# (definidos en src/environments/ace_env.py).
# Las demas son perturbaciones que alteran las prioridades relativas.
SENSITIVITY_CONFIGS = {
    'Nominal': {
        'takeoff':  {'thrust': 0.45, 'sfc': 0.10, 'power': 0.05, 'cooling': 0.10, 'safety': 0.30},
        'climb':    {'thrust': 0.30, 'sfc': 0.25, 'power': 0.05, 'cooling': 0.10, 'safety': 0.30},
        'cruise':   {'thrust': 0.10, 'sfc': 0.40, 'power': 0.15, 'cooling': 0.10, 'safety': 0.25},
        'combat':   {'thrust': 0.45, 'sfc': 0.05, 'power': 0.05, 'cooling': 0.15, 'safety': 0.30},
        'descent':  {'thrust': 0.05, 'sfc': 0.30, 'power': 0.20, 'cooling': 0.15, 'safety': 0.30},
    },
    'Thrust heavy': {
        'takeoff':  {'thrust': 0.60, 'sfc': 0.05, 'power': 0.05, 'cooling': 0.05, 'safety': 0.25},
        'climb':    {'thrust': 0.50, 'sfc': 0.15, 'power': 0.05, 'cooling': 0.05, 'safety': 0.25},
        'cruise':   {'thrust': 0.30, 'sfc': 0.25, 'power': 0.10, 'cooling': 0.10, 'safety': 0.25},
        'combat':   {'thrust': 0.60, 'sfc': 0.00, 'power': 0.05, 'cooling': 0.10, 'safety': 0.25},
        'descent':  {'thrust': 0.20, 'sfc': 0.20, 'power': 0.15, 'cooling': 0.15, 'safety': 0.30},
    },
    'SFC heavy': {
        'takeoff':  {'thrust': 0.30, 'sfc': 0.30, 'power': 0.05, 'cooling': 0.10, 'safety': 0.25},
        'climb':    {'thrust': 0.15, 'sfc': 0.45, 'power': 0.05, 'cooling': 0.10, 'safety': 0.25},
        'cruise':   {'thrust': 0.05, 'sfc': 0.60, 'power': 0.10, 'cooling': 0.05, 'safety': 0.20},
        'combat':   {'thrust': 0.30, 'sfc': 0.25, 'power': 0.05, 'cooling': 0.15, 'safety': 0.25},
        'descent':  {'thrust': 0.05, 'sfc': 0.50, 'power': 0.15, 'cooling': 0.05, 'safety': 0.25},
    },
    'Safety heavy': {
        'takeoff':  {'thrust': 0.30, 'sfc': 0.05, 'power': 0.05, 'cooling': 0.10, 'safety': 0.50},
        'climb':    {'thrust': 0.20, 'sfc': 0.15, 'power': 0.05, 'cooling': 0.10, 'safety': 0.50},
        'cruise':   {'thrust': 0.05, 'sfc': 0.25, 'power': 0.10, 'cooling': 0.10, 'safety': 0.50},
        'combat':   {'thrust': 0.30, 'sfc': 0.05, 'power': 0.05, 'cooling': 0.10, 'safety': 0.50},
        'descent':  {'thrust': 0.05, 'sfc': 0.20, 'power': 0.15, 'cooling': 0.10, 'safety': 0.50},
    },
    'Balanced': {
        'takeoff':  {'thrust': 0.25, 'sfc': 0.25, 'power': 0.15, 'cooling': 0.10, 'safety': 0.25},
        'climb':    {'thrust': 0.25, 'sfc': 0.25, 'power': 0.15, 'cooling': 0.10, 'safety': 0.25},
        'cruise':   {'thrust': 0.25, 'sfc': 0.25, 'power': 0.15, 'cooling': 0.10, 'safety': 0.25},
        'combat':   {'thrust': 0.25, 'sfc': 0.25, 'power': 0.15, 'cooling': 0.10, 'safety': 0.25},
        'descent':  {'thrust': 0.25, 'sfc': 0.25, 'power': 0.15, 'cooling': 0.10, 'safety': 0.25},
    },
}


def evaluate_sensitivity() -> dict:
    """Evalua el agente TD3 bajo distintas configuraciones de pesos.

    Para cada configuracion se ejecutan SENSITIVITY_EPISODES episodios,
    registrando el reward acumulado, empuje medio y SFC medio.

    Es necesario porque los pesos de la recompensa se calibraron de forma heurística, y
    una conclusión que solo se sostuviera con esos valores exactos sería frágil. Se
    evalúa la política ya entrenada bajo pesos alterados, sin reentrenar: lo que se
    mide es la robustez del comportamiento aprendido, no su capacidad de readaptarse.
    Aquí sí se fija la semilla por episodio, de modo que todas las configuraciones se
    enfrenten a las mismas misiones y la diferencia sea atribuible solo a los pesos.

    return: Diccionario {configuración: métricas} con recompensa media y desviación,
        empuje y consumo medios, tasa de seguridad y número de episodios; vacío si
        falta el agente o el gemelo digital.
    """
    logger.info("Analisis de sensibilidad del reward DRL")

    if not TD3_CHECKPOINT.exists():
        logger.warning(f"TD3 no disponible: {TD3_CHECKPOINT}")
        logger.warning("Se omite el analisis de sensibilidad (fig25)")
        return {}

    if not PINN_CHECKPOINT_FULL.exists():
        logger.warning(f"PINN no disponible: {PINN_CHECKPOINT_FULL}")
        return {}

    from stable_baselines3 import TD3
    from src.environments.ace_env import ACEEnv

    pinn = load_full_pinn(PINN_CHECKPOINT_FULL)
    logger.info(f"  PINN cargado ({sum(p.numel() for p in pinn.parameters()):,} params)")

    td3 = TD3.load(str(TD3_CHECKPOINT))
    logger.info(f"  TD3 cargado desde {TD3_CHECKPOINT}")

    results = {}

    for config_name, weights in SENSITIVITY_CONFIGS.items():
        logger.info(f"  Configuracion: {config_name}")
        env = ACEEnv(
            pinn_model=pinn,
            mission_profile='mixed',
            max_steps=SENSITIVITY_MAX_STEPS,
            reward_weights=weights,
        )

        rewards = []
        thrusts = []
        sfcs = []
        safety_rates = []

        for ep in range(SENSITIVITY_EPISODES):
            obs, _ = env.reset(seed=42 + ep)
            ep_reward = 0.0
            ep_thrusts = []
            ep_sfcs = []
            ep_safe = 0
            ep_steps = 0

            done = False
            truncated = False
            while not (done or truncated):
                action, _ = td3.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward
                ep_thrusts.append(info.get('thrust', 0))
                ep_sfcs.append(info.get('sfc', 0))
                if info.get('safe', True):
                    ep_safe += 1
                ep_steps += 1

            rewards.append(ep_reward)
            thrusts.append(float(np.mean(ep_thrusts)) if ep_thrusts else 0.0)
            sfcs.append(float(np.mean(ep_sfcs)) if ep_sfcs else 0.0)
            safety_rates.append(ep_safe / ep_steps if ep_steps else 0.0)

        results[config_name] = {
            'reward_mean':  float(np.mean(rewards)),
            'reward_std':   float(np.std(rewards)),
            'thrust_mean':  float(np.mean(thrusts)),
            'sfc_mean':     float(np.mean(sfcs)),
            'safety_rate':  float(np.mean(safety_rates)),
            'n_episodes':   SENSITIVITY_EPISODES,
        }

        logger.info(f"    reward = {results[config_name]['reward_mean']:.2f} "
                    f"+/- {results[config_name]['reward_std']:.2f}, "
                    f"thrust = {results[config_name]['thrust_mean']:.0f} lbf, "
                    f"sfc = {results[config_name]['sfc_mean']:.3f}, "
                    f"safety = {results[config_name]['safety_rate']*100:.1f}%")

    return results


def fig25_sensitivity_drl(results: dict, output_path: Path) -> None:
    """Dibuja el efecto de cada configuración de pesos sobre el comportamiento.

    Tres paneles con recompensa, empuje y consumo medios por configuración, el primero
    con barras de error. Es necesario mirarlos juntos: que la recompensa cambie al
    cambiar los pesos es trivial, porque cambia la propia métrica; lo informativo es si
    el empuje y el consumo, que son magnitudes físicas independientes de la
    ponderación, se mantienen estables o se desplazan.

    param results: Diccionario {configuración: métricas} del análisis de sensibilidad.
    param output_path: Ruta del archivo PNG a generar.
    return: None; la figura queda escrita en output_path, o se omite si no hay datos.
    """
    if not results:
        logger.warning("Sin datos de sensibilidad; se omite fig25")
        return

    configs = list(results.keys())
    rewards = [results[c]['reward_mean'] for c in configs]
    reward_stds = [results[c]['reward_std'] for c in configs]
    thrusts = [results[c]['thrust_mean'] for c in configs]
    sfcs = [results[c]['sfc_mean'] for c in configs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: reward medio con barras de error
    colors = ['#2A9D8F', '#E9C46A', '#F4A261', '#E63946', '#264653']
    bars = axes[0].bar(configs, rewards, yerr=reward_stds,
                       color=colors[:len(configs)], alpha=0.85, capsize=5)
    axes[0].set_ylabel('Reward acumulado')
    axes[0].set_title('Reward medio por configuracion')
    for bar, val in zip(bars, rewards):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', fontsize=10)
    axes[0].tick_params(axis='x', rotation=30)

    # Panel 2: thrust medio
    bars = axes[1].bar(configs, thrusts, color=colors[:len(configs)],
                        alpha=0.85)
    axes[1].set_ylabel('Empuje medio [lbf]')
    axes[1].set_title('Empuje medio por configuracion')
    for bar, val in zip(bars, thrusts):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(thrusts) * 0.01,
                     f'{val:.0f}', ha='center', fontsize=10)
    axes[1].tick_params(axis='x', rotation=30)

    # Panel 3: SFC medio
    bars = axes[2].bar(configs, sfcs, color=colors[:len(configs)],
                        alpha=0.85)
    axes[2].set_ylabel('SFC medio [lbm/(lbf$\\cdot$h)]')
    axes[2].set_title('SFC medio por configuracion')
    for bar, val in zip(bars, sfcs):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(sfcs) * 0.01,
                     f'{val:.3f}', ha='center', fontsize=10)
    axes[2].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {output_path}")


# 4. Perfilado SWaP-C (fig27)
def profile_latency() -> dict:
    """Perfila el gemelo digital en tamaño, latencia y escalabilidad por lote.

    Mide parámetros, memoria y espacio en disco, cronometra la inferencia unitaria tras
    un calentamiento y repite la medida para varios tamaños de lote. Es necesario
    porque el argumento de viabilidad del TFG no es solo de precisión sino de
    despliegue: el modelo debe caber y responder dentro de los presupuestos de una
    unidad de control embarcada, y por eso el resultado se contrasta con los umbrales
    temporales asociados a los niveles de criticidad aeronáuticos.

    return: Diccionario con las secciones 'size', 'latency' (incluida la escalabilidad
        por lote) y 'compliance' con el cumplimiento de cada umbral; vacío si el
        checkpoint no está disponible.
    """
    logger.info("Perfilado SWaP-C del PINN")

    if not PINN_CHECKPOINT_FULL.exists():
        logger.warning(f"PINN no disponible: {PINN_CHECKPOINT_FULL}")
        return {}

    model = load_full_pinn(PINN_CHECKPOINT_FULL)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)
    memory_mb = sum(p.nelement() * p.element_size()
                    for p in model.parameters()) / 1e6
    disk_mb = PINN_CHECKPOINT_FULL.stat().st_size / 1e6

    size_info = {
        'params_total':     n_params,
        'params_trainable': n_trainable,
        'memory_mb':        memory_mb,
        'disk_mb':          disk_mb,
    }

    logger.info(f"  Parametros:      {n_params:,}")
    logger.info(f"  Memoria modelo:  {memory_mb:.2f} MB")
    logger.info(f"  Tamano en disco: {disk_mb:.2f} MB")

    x_single = torch.randn(1, 4)
    for _ in range(LATENCY_WARMUP):
        with torch.no_grad():
            model(x_single)

    times_single = []
    for _ in range(LATENCY_ITERATIONS):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x_single)
        times_single.append((time.perf_counter() - t0) * 1000)

    batch_scaling = {}
    for bs in BATCH_SIZES:
        x_batch = torch.randn(bs, 4)
        times = []
        for _ in range(BATCH_ITERATIONS):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(x_batch)
            times.append((time.perf_counter() - t0) * 1000)
        batch_scaling[bs] = {
            'mean':       float(np.mean(times)),
            'p95':        float(np.percentile(times, 95)),
            'p99':        float(np.percentile(times, 99)),
            'per_sample': float(np.mean(times) / bs),
        }

    latency_info = {
        'mean_ms':       float(np.mean(times_single)),
        'std_ms':        float(np.std(times_single)),
        'p50_ms':        float(np.percentile(times_single, 50)),
        'p95_ms':        float(np.percentile(times_single, 95)),
        'p99_ms':        float(np.percentile(times_single, 99)),
        'max_ms':        float(max(times_single)),
        'batch_scaling': batch_scaling,
    }

    logger.info(f"  Latencia single (CPU):")
    logger.info(f"    media: {latency_info['mean_ms']:.3f} ms")
    logger.info(f"    P50:   {latency_info['p50_ms']:.3f} ms")
    logger.info(f"    P95:   {latency_info['p95_ms']:.3f} ms")
    logger.info(f"    P99:   {latency_info['p99_ms']:.3f} ms")

    p99 = latency_info['p99_ms']
    compliance = {
        'DO178C_level_C_20ms': p99 < DO178C_LEVEL_C_MS,
        'DO178C_level_B_10ms': p99 < DO178C_LEVEL_B_MS,
        'DO178C_level_A_5ms':  p99 < DO178C_LEVEL_A_MS,
        'memory_under_10mb':   memory_mb < 10,
    }

    logger.info("  Compatibilidad SWaP-C:")
    for key, ok in compliance.items():
        logger.info(f"    {key}: {'si' if ok else 'no'}")

    return {
        'size':       size_info,
        'latency':    latency_info,
        'compliance': compliance,
    }


def fig27_latency_swap(results: dict, output_path: Path) -> None:
    """Dibuja los percentiles de latencia y la escalabilidad por tamaño de lote.

    El panel izquierdo sitúa media, percentiles y máximo frente a los umbrales de
    criticidad; el derecho muestra el coste por muestra al agrupar consultas. Ambos
    responden a preguntas distintas: el primero, si el modelo cumple el requisito de
    tiempo real en el peor caso, que es el que decide; el segundo, cuánto se
    amortizaría el coste fijo si la aplicación permitiera agrupar consultas.

    param results: Diccionario de perfilado devuelto por profile_latency().
    param output_path: Ruta del archivo PNG a generar.
    return: None; la figura queda escrita en output_path, o se omite si no hay datos.
    """
    if not results:
        logger.warning("Sin datos de perfilado; se omite fig27")
        return

    latency = results['latency']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    metrics = ['Media', 'P50', 'P95', 'P99', 'Max']
    values = [latency['mean_ms'], latency['p50_ms'], latency['p95_ms'],
              latency['p99_ms'], latency['max_ms']]
    colors = ['#2A9D8F', '#2A9D8F', '#E9C46A', '#E63946', '#E63946']

    bars = axes[0].bar(metrics, values, color=colors, alpha=0.85)
    axes[0].set_ylabel('Latencia [ms]')
    axes[0].axhline(y=DO178C_LEVEL_C_MS, color='red',
                    linestyle='--', alpha=0.5,
                    label=f'Nivel C ({DO178C_LEVEL_C_MS} ms)')
    axes[0].axhline(y=DO178C_LEVEL_A_MS, color='green',
                    linestyle='--', alpha=0.5,
                    label=f'Nivel A ({DO178C_LEVEL_A_MS} ms)')
    axes[0].legend(fontsize=9)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.1,
                     f'{val:.2f}', ha='center', fontsize=10)

    batch_data = latency['batch_scaling']
    batch_sizes = list(batch_data.keys())
    per_sample = [batch_data[bs]['per_sample'] for bs in batch_sizes]
    axes[1].plot(batch_sizes, per_sample, 'o-', color='#2A9D8F',
                 linewidth=2, markersize=8)
    axes[1].set_xlabel('Batch size')
    axes[1].set_ylabel('Latencia por muestra [ms]')
    for bs, ps in zip(batch_sizes, per_sample):
        axes[1].annotate(f'{ps:.3f}', (bs, ps),
                         textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {output_path}")


# Punto de entrada
def main() -> None:
    """Punto de entrada que ejecuta los cuatro análisis y genera sus figuras.

    Encadena ablación, pruebas fuera del envelope, sensibilidad y perfilado, consolida
    todo en un único JSON y dibuja después las figuras. Volcar el JSON antes de
    dibujar protege el resultado de la parte cara del proceso frente a un fallo en la
    generación de gráficos. Cada análisis devuelve vacío si le faltan checkpoints, de
    modo que el script produce todo lo que sí es posible producir.

    return: None; deja el JSON en results/section_results/ y los PNG en results/figures/.    """
    logger.info("Evaluacion Seccion 6: Validacion critica del sistema PINN+DRL")
    ablation_results = evaluate_ablation()
    ood_in, ood_out = evaluate_ood()
    sensitivity_results = evaluate_sensitivity()
    latency_results = profile_latency()

    all_results = {
        'ablation':     ablation_results,
        'ood_in_dist':  {k: {kk: float(vv) for kk, vv in v.items()}
                         for k, v in ood_in.items()},
        'ood_out_dist': {k: {kk: float(vv) for kk, vv in v.items()}
                         for k, v in ood_out.items()},
        'sensitivity':  sensitivity_results,
        'latency':      latency_results,
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False,
                  default=str)
    logger.info(f"Resultados guardados en {RESULTS_JSON}")

    logger.info("Generando figuras...")
    n_figs = 0
    if ablation_results:
        fig24_ablation_study(ablation_results,
                             FIG_DIR / 'fig24_ablation_study.png')
        n_figs += 1
    if sensitivity_results:
        fig25_sensitivity_drl(sensitivity_results,
                              FIG_DIR / 'fig25_sensitivity_drl.png')
        n_figs += 1
    if ood_in or ood_out:
        fig26_ood_tests(ood_in, ood_out,
                        FIG_DIR / 'fig26_ood_tests.png')
        n_figs += 1
    if latency_results:
        fig27_latency_swap(latency_results,
                           FIG_DIR / 'fig27_latency_swap.png')
        n_figs += 1

    logger.info(f"Seccion 6 completada: {n_figs}/4 figuras generadas")


if __name__ == '__main__':
    main()