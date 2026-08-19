"""
Última Fecha de Modificación: 08/Aug/2026
Descripción plot_rnn_single.py: Script que genera las figuras de análisis en
profundidad de una única arquitectura del monitor de salud. Complementa a
plot_rnn_comparison.py: aquel compara las tres RNN entre sí, y este examina
el comportamiento temporal de una sola. Sobre el motor
más largo del conjunto de validación dibuja la predicción de vida remanente con
sus bandas de incertidumbre bayesiana obtenidas por MC-Dropout (Gal & Ghahramani,
2016) y el seguimiento del vector de degradación componente a componente. El
script solo requiere el checkpoint activo de la RNN y el sub-dataset C-MAPSS
correspondiente; no re-entrena en ningún momento.

Figuras generadas en results/figures/ (nombre parametrizado por
arquitectura, por ejemplo fig32_rnn_lstm128_uncertainty.png):
    - fig31_rnn_{name}_uncertainty.png       MC-Dropout sobre un motor
    - fig32_rnn_{name}_degradation.png       Tracking por componente

Uso:
    python scripts/plotting/plot_rnn_single.py
    python scripts/plotting/plot_rnn_single.py --rnn gru --hidden 64
    python scripts/plotting/plot_rnn_single.py --rnn lstm --hidden 128
"""

import argparse
import logging
import sys
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
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rnn_health import HealthMonitoringRNN
from src.preprocessing.cmapss_loader import CMAPSSLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path('checkpoints/health_monitoring')
CMAPSS_DIR = Path('data/CMAPSS_DATA')
FIG_DIR = Path('results/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

MC_DROPOUT_SAMPLES = 50
RUL_CLIP = 125
BASELINE_CYCLES = 10

COMPONENT_NAMES = ['Fan (η_fan)', 'HPC (η_hpc)',
                   'HPT (η_hpt)', 'LPT (η_lpt)']
COMPONENT_COLORS = ['#2A9D8F', '#E63946', '#E9C46A', '#457B9D']

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


def load_rnn(checkpoint_path: Path, rnn_type: str, hidden_dim: int,
             n_layers: int) -> HealthMonitoringRNN:
    """Construye la red de monitorización y carga en ella los pesos del checkpoint.

    La arquitectura se recibe como argumento, y no se deduce del archivo, porque el
    nombre del checkpoint ya la codifica y el script la necesita también para nombrar
    las figuras. La carga es no estricta para tolerar diferencias menores de claves.

    param checkpoint_path: Ruta del archivo de pesos a cargar.
    param rnn_type: Variante recurrente del checkpoint, 'lstm' o 'gru'.
    param hidden_dim: Dimensión del estado oculto de la red.
    param n_layers: Número de capas recurrentes apiladas.
    return: Red HealthMonitoringRNN con los pesos cargados y en modo evaluación.
    """
    model = HealthMonitoringRNN(
        n_sensors=15,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        rnn_type=rnn_type,
    )
    state_dict = torch.load(checkpoint_path, map_location='cpu',
                            weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def select_longest_val_unit(df, val_units: np.ndarray) -> int:
    """Selecciona el motor de validación con mayor número de ciclos registrados.

    Se elige el más largo porque es el que ofrece una trayectoria de degradación
    completa, desde el estado sano hasta el fallo: en un motor corto la curva de vida
    remanente apenas muestra tramo útil y la figura resultaría poco informativa. La
    elección es determinista, de modo que la figura sea siempre la misma.

    param df: DataFrame del sub-dataset con la columna unit_id.
    param val_units: Identificadores de los motores del conjunto de validación.
    return: Identificador del motor de validación con más ciclos.
    """
    lengths = {u: len(df[df['unit_id'] == u]) for u in val_units}
    return max(lengths, key=lengths.get)


def compute_mc_dropout_predictions(model: HealthMonitoringRNN,
                                   sensor_vals: np.ndarray,
                                   seq_len: int) -> tuple:
    """Recorre la vida del motor prediciendo vida remanente con incertidumbre.

    Desliza la ventana ciclo a ciclo y aplica MC-Dropout en cada posición. Es
    necesario para poder dibujar la evolución temporal de la incertidumbre, que es
    justamente lo interesante: se espera que las bandas se estrechen conforme el
    motor se acerca al fallo y la señal de degradación se hace inequívoca. Se fija
    la semilla antes del bucle para que las bandas dibujadas sean reproducibles
    entre ejecuciones sucesivas del script, coherente con plot_rnn_comparison.py.

    param model: Red de monitorización cargada.
    param sensor_vals: Matriz (ciclos, sensores) de la trayectoria del motor.
    param seq_len: Longitud de la ventana deslizante.
    return: Tupla de tres arrays alineados: índices del ciclo final de cada ventana,
        medias de la vida remanente predicha y desviaciones típicas de Monte Carlo.
    """
    # Semilla fija para reproducibilidad de MC-Dropout entre ejecuciones.
    # Sin ella, las bandas de incertidumbre varían ligeramente entre corridas.
    torch.manual_seed(42)
    np.random.seed(42)

    cycles_idx = []
    means, stds = [], []

    for i in range(len(sensor_vals) - seq_len + 1):
        seq = torch.tensor(sensor_vals[i:i + seq_len]).unsqueeze(0)
        uq = model.predict_with_uncertainty(seq, n_samples=MC_DROPOUT_SAMPLES)

        cycles_idx.append(i + seq_len - 1)
        means.append(uq['rul_mean'].item())
        stds.append(uq['rul_std'].item())

    return (np.array(cycles_idx),
            np.array(means),
            np.array(stds))


def compute_degradation_predictions(model: HealthMonitoringRNN,
                                    sensor_vals: np.ndarray,
                                    seq_len: int) -> tuple:
    """Recorre la vida del motor prediciendo el vector de degradación por componente.

    A diferencia de la vida remanente, aquí la predicción es determinista: interesa la
    trayectoria del deterioro estimado, no su incertidumbre, y desactivar el dropout
    evita que el ruido de muestreo enmascare la tendencia que la figura debe mostrar.

    param model: Red de monitorización cargada.
    param sensor_vals: Matriz (ciclos, sensores) de la trayectoria del motor.
    param seq_len: Longitud de la ventana deslizante.
    return: Tupla (índices del ciclo final de cada ventana, matriz de predicciones de
        degradación con una columna por componente).
    """
    cycles_idx = []
    preds = []

    with torch.no_grad():
        for i in range(len(sensor_vals) - seq_len + 1):
            seq = torch.tensor(sensor_vals[i:i + seq_len]).unsqueeze(0)
            _, deg = model(seq)
            cycles_idx.append(i + seq_len - 1)
            preds.append(deg.squeeze().numpy())

    return np.array(cycles_idx), np.array(preds)


def compute_pseudo_labels(unit_df, deg_sensors: list) -> np.ndarray:
    """Calcula la referencia de degradación como desviación respecto al motor sano.

    Reproduce exactamente el mismo cálculo que emplea el Dataset durante el
    entrenamiento, con el mismo número de ciclos de baseline. Esa coincidencia es
    imprescindible: si la referencia dibujada se construyese de otra forma, el error
    mostrado en la figura no sería el que el modelo realmente optimizó.

    param unit_df: Filas de un único motor, ordenadas por ciclo.
    param deg_sensors: Sensores usados como proxy de degradación por componente.
    return: Matriz (ciclos, componentes) con el delta relativo de cada sensor
        respecto a su baseline.
    """
    n_base = min(BASELINE_CYCLES, len(unit_df))
    baselines = {}
    for sensor in deg_sensors:
        baseline_mean = unit_df[sensor].iloc[:n_base].mean()
        baselines[sensor] = (baseline_mean if abs(baseline_mean) > 1e-8
                             else 1e-8)

    deg_labels = np.zeros((len(unit_df), len(deg_sensors)))
    for j, sensor in enumerate(deg_sensors):
        deg_labels[:, j] = (unit_df[sensor].values - baselines[sensor]) \
                           / abs(baselines[sensor])
    return deg_labels


def plot_uncertainty(cycles: np.ndarray, real: np.ndarray,
                     means: np.ndarray, stds: np.ndarray,
                     unit_id: int, subset: str,
                     output_path: Path) -> None:
    """Dibuja la predicción de vida remanente con sus bandas de incertidumbre.

    Superpone la vida remanente real, la predicha y dos bandas anidadas de una y dos
    desviaciones típicas. Mostrar ambas bandas y no solo el intervalo al 95% permite
    juzgar de un vistazo si la curva real se mantiene dentro de la región de mayor
    densidad o si solo cabe en el intervalo ancho, que es la diferencia entre una
    incertidumbre bien calibrada y una simplemente generosa.

    param cycles: Ciclos de operación correspondientes a cada predicción.
    param real: Vida remanente real, ya saturada al límite del entrenamiento.
    param means: Media de la vida remanente predicha por MC-Dropout.
    param stds: Desviación típica de la predicción en cada ciclo.
    param unit_id: Identificador del motor representado, para la anotación.
    param subset: Sub-dataset de C-MAPSS de procedencia, para la anotación.
    param output_path: Ruta del archivo PNG a generar.
    return: None; la figura queda escrita en output_path.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(cycles, real, 'k-', linewidth=2, label='RUL real', zorder=3)
    ax.plot(cycles, means, 'r-', linewidth=1.5,
            label='Predicción RNN', zorder=2)

    ax.fill_between(cycles, means - 2 * stds, means + 2 * stds,
                    alpha=0.25, color='red',
                    label='Incertidumbre ±2σ')
    ax.fill_between(cycles, means - stds, means + stds,
                    alpha=0.35, color='salmon',
                    label='IC 68% (±1σ)')

    ax.set_xlabel('Ciclo de operación')
    ax.set_ylabel('RUL [ciclos]')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=-5)

    rmse = float(np.sqrt(np.mean((means - real) ** 2)))
    info = (f'Motor #{unit_id} — C-MAPSS {subset}\n'
            f'RMSE = {rmse:.1f} ciclos\n'
            f'σ medio = ±{stds.mean():.1f} ciclos')
    ax.text(0.02, 0.95, info,
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {output_path}")


def plot_degradation_tracking(cycles: np.ndarray, real: np.ndarray,
                              pred: np.ndarray, unit_id: int,
                              output_path: Path) -> None:
    """Dibuja el seguimiento del deterioro estimado para cada componente.

    Un panel por componente, contrastando la referencia con la predicción y anotando
    el error. El desglose es necesario porque el interés del segundo cabezal está
    precisamente en distinguir qué se está degradando: un panel agregado ocultaría que
    el modelo puede seguir bien al HPC, el modo de fallo dominante, y mucho peor al
    resto de componentes.

    param cycles: Ciclos de operación correspondientes a cada predicción.
    param real: Matriz de referencia de degradación por componente.
    param pred: Matriz de degradación predicha por la red.
    param unit_id: Identificador del motor representado, para los títulos.
    param output_path: Ruta del archivo PNG a generar.
    return: None; la figura queda escrita en output_path.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    n_components = min(len(COMPONENT_NAMES), pred.shape[1], real.shape[1])
    for idx in range(n_components):
        ax = axes.flat[idx]
        ax.plot(cycles, real[:, idx] * 100, 'k-',
                linewidth=1.5, alpha=0.7, label='Pseudo-label')
        ax.plot(cycles, pred[:, idx] * 100, '-',
                color=COMPONENT_COLORS[idx], linewidth=2,
                label='Predicción RNN')

        rmse = float(np.sqrt(np.mean(
            (pred[:, idx] - real[:, idx]) ** 2)))
        ax.text(0.02, 0.05, f'RMSE = {rmse:.4f}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Ciclo de operación')
        ax.set_ylabel('Δη [%]')
        ax.set_title(f'{COMPONENT_NAMES[idx]} (Motor #{unit_id})')
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {output_path}")


def main() -> None:
    """Punto de entrada que genera las figuras de análisis de una arquitectura.

    Carga el checkpoint, reproduce la partición por motor del entrenamiento con la
    misma semilla, elige el motor de validación más largo y encadena las dos figuras.
    Reproducir la partición es imprescindible: dibujar la predicción sobre un motor
    que la red vio al entrenar daría una imagen engañosamente buena. El nombre de las
    figuras incorpora la arquitectura para no sobrescribir las de otras variantes.

    return: None; sale sin generar nada si falta el checkpoint, y en caso contrario
        deja los PNG en results/figures.
    """
    parser = argparse.ArgumentParser(
        description="Figuras de análisis de una RNN individual")
    parser.add_argument('--rnn', choices=['lstm', 'gru'], default='gru')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    # La ventana debe ser la misma con la que se entrenó el checkpoint
    # (train_rnn.py --seq-len, 30 ciclos): la red se evalúa así en el mismo
    # régimen temporal que vio durante el ajuste, y las figuras son
    # directamente comparables con las métricas de plot_rnn_comparison.py.
    parser.add_argument('--seq-len', type=int, default=30)
    parser.add_argument('--subset', default='FD001')
    parser.add_argument('--checkpoint', type=Path, default=None)
    args = parser.parse_args()

    name = f'{args.rnn}{args.hidden}'
    checkpoint_path = args.checkpoint or (
        CHECKPOINT_DIR / f'rnn_{args.rnn}_{args.hidden}.pt')

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint no encontrado: {checkpoint_path}")
        return

    logger.info(f"Arquitectura: {args.rnn.upper()}-{args.hidden}")
    logger.info(f"Checkpoint:   {checkpoint_path}")

    model = load_rnn(checkpoint_path, args.rnn, args.hidden, args.layers)

    loader = CMAPSSLoader(data_path=str(CMAPSS_DIR))
    df = loader.load_subset(args.subset)
    df = loader.compute_health_index(df)
    df = loader.normalize(df, method='minmax')

    # Reproducir el split de validación usado en entrenamiento.
    units = df['unit_id'].unique()
    np.random.RandomState(42).shuffle(units)
    n_train = int(len(units) * 0.8)
    val_units = units[n_train:]

    target_unit = select_longest_val_unit(df, val_units)
    unit_df = df[df['unit_id'] == target_unit].sort_values('cycle')

    sensors = [s for s in HealthMonitoringRNN.SENSOR_FEATURES
               if s in unit_df.columns]
    sensor_vals = unit_df[sensors].values.astype(np.float32)
    rul_real = np.clip(unit_df['RUL'].values.astype(np.float32),
                       0, RUL_CLIP)
    cycles = unit_df['cycle'].values

    logger.info(f"Motor de referencia (más largo de val): "
                f"#{target_unit} ({len(unit_df)} ciclos)")

    # MC-Dropout para incertidumbre epistémica.
    logger.info("Calculando predicciones con MC-Dropout...")
    idx, means, stds = compute_mc_dropout_predictions(
        model, sensor_vals, args.seq_len)
    plot_uncertainty(
        cycles=cycles[idx],
        real=rul_real[idx],
        means=means,
        stds=stds,
        unit_id=int(target_unit),
        subset=args.subset,
        output_path=FIG_DIR / f'fig31_rnn_{name}_uncertainty.png',
    )

    # Tracking del vector de degradación.
    logger.info("Calculando tracking de degradación...")
    deg_sensors = [s for s in HealthMonitoringRNN.DEGRADATION_SENSORS
                   if s in unit_df.columns]
    real_deg_full = compute_pseudo_labels(unit_df, deg_sensors)
    idx_deg, pred_deg = compute_degradation_predictions(
        model, sensor_vals, args.seq_len)
    plot_degradation_tracking(
        cycles=cycles[idx_deg],
        real=real_deg_full[idx_deg],
        pred=pred_deg,
        unit_id=int(target_unit),
        output_path=FIG_DIR / f'fig32_rnn_{name}_degradation.png',
    )

if __name__ == '__main__':
    main()