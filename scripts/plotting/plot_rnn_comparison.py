"""
Última Fecha de Modificación: 08/Aug/2026
Descripción plot_rnn_comparison.py: Script que genera las figuras comparativas
de las tres arquitecturas del monitor de salud (LSTM-128, LSTM-64 y GRU-64).
Evalúa los tres checkpoints sobre el mismo conjunto de validación de C-MAPSS
FD001 (val_loader único compartido, es la comparativa oficial de la memoria del
TFG) y produce las barras de métricas, los diagramas de dispersión de vida
remanente, el seguimiento de degradación por componente y la curva de
calibración de la incertidumbre. Los resultados soportan la selección del
modelo GRU-64 como configuración final.

Figuras generadas en results/figures/:
    fig15_rnn_metrics_comparison.png     Barras: RMSE, MAE, Deg RMSE, IC95
    fig16_rul_scatter_comparison.png     Scatter RUL predicho vs real
    fig17_degradation_tracking.png       Tracking temporal por componente
    fig18_uncertainty_calibration.png    Cobertura empírica vs nominal

Uso:
    python scripts/plotting/plot_rnn_comparison.py
"""

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

from src.models.rnn_health import HealthMonitor, HealthMonitoringRNN
from src.preprocessing.cmapss_loader import CMAPSSLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints' / 'health_monitoring'
CMAPSS_DIR = PROJECT_ROOT / 'data' / 'CMAPSS_DATA'
FIG_DIR = PROJECT_ROOT / 'results' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

MC_DROPOUT_PASSES = 50
COMPONENT_NAMES = ['Fan', 'HPC', 'HPT', 'LPT']

# Configuración visual (paper style, sin títulos implícitos en figuras)
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 12,
    'font.family': 'serif',
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

MODELS = [
    {
        'checkpoint': 'rnn_lstm_128.pt',
        'label':      'LSTM-128',
        'rnn_type':   'lstm',
        'hidden':     128,
        'color':      '#E63946',  
    },
    {
        'checkpoint': 'rnn_lstm_64.pt',
        'label':      'LSTM-64',
        'rnn_type':   'lstm',
        'hidden':     64,
        'color':      '#E9C46A',  
    },
    {
        'checkpoint': 'rnn_gru_64.pt',
        'label':      'GRU-64',
        'rnn_type':   'gru',
        'hidden':     64,
        'color':      '#2A9D8F',  
    },
]


def evaluate_checkpoint(cfg: dict, val_loader) -> dict:
    """Evalúa un checkpoint y devuelve tanto sus métricas como sus predicciones crudas.

    Además de los agregados habituales conserva los vectores de predicción y de
    dispersión de Monte Carlo. Esa retención es necesaria porque las figuras
    posteriores no se conforman con los resúmenes: el diagrama de dispersión, el
    seguimiento temporal y la curva de calibración necesitan las predicciones punto a
    punto, y recalcularlas para cada figura multiplicaría por cuatro el coste. Se
    fija la semilla antes de la inferencia para que las 50 pasadas de MC-Dropout
    sean reproducibles entre ejecuciones sucesivas del script.

    param cfg: Entrada de MODELS con el checkpoint, la etiqueta, la arquitectura y el
        color asignado al modelo.
    param val_loader: DataLoader de validación compartido por todos los modelos.
    return: Diccionario con etiqueta, color, número de parámetros, RMSE y MAE de vida
        remanente, errores de degradación por componente, dispersión media, cobertura
        y los arrays de predicciones y referencias.
    """
    # Semilla fija para reproducibilidad de MC-Dropout entre ejecuciones.
    # Sin ella, las 50 pasadas producen σ ligeramente distintos y la
    # cobertura IC95 puede variar ±10 puntos porcentuales entre corridas.
    torch.manual_seed(42)
    np.random.seed(42)

    path = CHECKPOINT_DIR / cfg['checkpoint']
    ckpt = torch.load(path, map_location='cpu')
    # Configuración estándar de los tres modelos (definida en
    # src/models/rnn_health.py y usada en el entrenamiento):
    # - 15 sensores: del subset de FD001 que quedan tras eliminar los
    #   sensores constantes según Saxena et al. (2008 PHM).
    # - 2 capas RNN: profundidad estándar para series temporales cortas.
    # - dropout=0.3: valor usado durante el entrenamiento; MC-Dropout
    #   reactiva estas dropouts en inferencia para obtener incertidumbre.
    model = HealthMonitoringRNN(
        n_sensors=15,
        hidden_dim=cfg['hidden'],
        n_layers=2,
        rnn_type=cfg['rnn_type'],
        dropout=0.3,
        bidirectional=False,
    )
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # Predicción determinista
    rul_pred, rul_true, deg_pred, deg_true = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            r_pred, d_pred = model(batch[0])
            rul_pred.append(r_pred.numpy())
            rul_true.append(batch[1].numpy())
            deg_pred.append(d_pred.numpy())
            deg_true.append(batch[2].numpy())

    rul_pred = np.concatenate(rul_pred).flatten()
    rul_true = np.concatenate(rul_true).flatten()
    deg_pred = np.concatenate(deg_pred)
    deg_true = np.concatenate(deg_true)

    rmse = np.sqrt(np.mean((rul_pred - rul_true) ** 2))
    mae = np.mean(np.abs(rul_pred - rul_true))
    deg_rmses = [
        np.sqrt(np.mean((deg_pred[:, j] - deg_true[:, j]) ** 2))
        for j in range(4)
    ]

    # MC-Dropout para incertidumbre epistémica
    model.train()
    mc = []
    with torch.no_grad():
        for _ in range(MC_DROPOUT_PASSES):
            bp = []
            for batch in val_loader:
                r, _ = model(batch[0])
                bp.append(r.numpy())
            mc.append(np.concatenate(bp).flatten())
    mc = np.array(mc)
    mc_mean = mc.mean(axis=0)
    mc_std = mc.std(axis=0)
    lower = mc_mean - 1.96 * mc_std
    upper = mc_mean + 1.96 * mc_std
    coverage = np.mean((rul_true >= lower) & (rul_true <= upper)) * 100

    return {
        'label':     cfg['label'],
        'color':     cfg['color'],
        'params':    sum(p.numel() for p in model.parameters()),
        'rmse':      rmse,
        'mae':       mae,
        'deg_rmses': deg_rmses,
        'sigma':     mc_std.mean(),
        'coverage':  coverage,
        'rul_pred':  rul_pred,
        'rul_true':  rul_true,
        'deg_pred':  deg_pred,
        'deg_true':  deg_true,
        'mc_std':    mc_std,
    }


def plot_metrics_comparison(results: list, output_path: Path) -> None:
    """Dibuja el panel de barras con las cuatro métricas comparativas.

    Reúne error de vida remanente, error absoluto medio, error de degradación por
    componente y cobertura del intervalo frente a su valor nominal. Presentarlas
    juntas es lo que permite ver el compromiso real: la arquitectura más precisa en
    vida remanente no tiene por qué ser la mejor calibrada.

    param results: Lista de diccionarios de resultados, uno por modelo evaluado.
    param output_path: Ruta del archivo PNG a generar.
    return: None; la figura queda escrita en output_path.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    labels = [r['label'] for r in results]
    colors = [r['color'] for r in results]

    # RMSE
    vals = [r['rmse'] for r in results]
    bars = axes[0].bar(labels, vals, color=colors, edgecolor='white')
    axes[0].set_ylabel('RUL RMSE [ciclos]')
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01,
                     f'{v:.2f}', ha='center', fontweight='bold')

    # MAE
    vals = [r['mae'] for r in results]
    bars = axes[1].bar(labels, vals, color=colors, edgecolor='white')
    axes[1].set_ylabel('RUL MAE [ciclos]')
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01,
                     f'{v:.2f}', ha='center', fontweight='bold')

    # Degradación por componente
    x = np.arange(len(COMPONENT_NAMES))
    w = 0.25
    for i, r in enumerate(results):
        axes[2].bar(x + i * w, r['deg_rmses'], w,
                    label=r['label'], color=r['color'], edgecolor='white')
    axes[2].set_xticks(x + w)
    axes[2].set_xticklabels(COMPONENT_NAMES)
    axes[2].set_ylabel('Deg RMSE')
    axes[2].legend(fontsize=8)

    # Cobertura IC95
    vals = [r['coverage'] for r in results]
    bars = axes[3].bar(labels, vals, color=colors, edgecolor='white')
    axes[3].axhline(y=95, color='red', linestyle='--', alpha=0.5,
                    label='Nominal (95%)')
    axes[3].set_ylabel('Cobertura empírica [%]')
    axes[3].legend(fontsize=8)
    for b, v in zip(bars, vals):
        axes[3].text(b.get_x() + b.get_width() / 2, v + 1,
                     f'{v:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {output_path}")


def plot_rul_scatter(results: list, output_path: Path) -> None:
    """Dibuja un diagrama de dispersión de vida remanente por cada modelo.

    Enfrenta predicción y valor real con la diagonal ideal como referencia. Es
    necesario porque el error agregado no distingue entre un modelo que se equivoca de
    forma homogénea y otro que acierta en la zona de fallo inminente pero se desvía
    con el motor sano, siendo esta última la región que de verdad importa para la
    decisión de mantenimiento.

    param results: Lista de diccionarios de resultados, uno por modelo evaluado.
    param output_path: Ruta del archivo PNG a generar.
    return: None; la figura queda escrita en output_path.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6))
    if len(results) == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        ax.scatter(r['rul_true'], r['rul_pred'], alpha=0.3, s=5,
                   color=r['color'], edgecolors='none')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel('RUL real (normalizado)')
        ax.set_ylabel('RUL predicho (normalizado)')
        ax.set_title(f"{r['label']}\nRMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}")
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {output_path}")


def plot_degradation_tracking(results: list, output_path: Path,
                              n_show: int = 200) -> None:
    """Dibuja el seguimiento de degradación de los tres modelos, componente a componente.

    Superpone las predicciones de todas las arquitecturas sobre la misma referencia,
    lo que permite comparar directamente su suavidad y su retardo de seguimiento.
    Se limita a las primeras n_show muestras porque con el conjunto completo las
    curvas se solapan hasta hacerse ilegibles.

    param results: Lista de diccionarios de resultados, uno por modelo evaluado.
    param output_path: Ruta del archivo PNG a generar.
    param n_show: Número máximo de muestras representadas por panel.
    return: None; la figura queda escrita en output_path.
    """
    n_show = min(n_show, len(results[0]['deg_true']))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for j, (ax, comp) in enumerate(zip(axes.flat, COMPONENT_NAMES)):
        ax.plot(results[0]['deg_true'][:n_show, j], 'k-',
                alpha=0.4, linewidth=0.8, label='Real')
        for r in results:
            ax.plot(r['deg_pred'][:n_show, j], color=r['color'],
                    alpha=0.7, linewidth=1, label=r['label'])
        rmse_str = ', '.join(f"{r['label']}={r['deg_rmses'][j]:.3f}"
                             for r in results)
        ax.set_title(f'{comp} — RMSE {rmse_str}')
        ax.set_xlabel('Muestra')
        ax.set_ylabel('Degradación')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {output_path}")


def plot_uncertainty_calibration(results: list, output_path: Path) -> None:
    """Dibuja la curva de calibración de la incertidumbre de cada modelo.

    Barre distintos niveles de confianza y contrasta la cobertura realmente observada
    con la nominal, usando la diagonal como calibración perfecta. Es necesario porque
    la cobertura al 95% es un único punto: solo la curva completa revela si el modelo
    es sistemáticamente confiado en exceso o si el desajuste se concentra en un tramo
    concreto. La conversión de cuantil de cobertura a z-score usa la inversa exacta
    de la CDF normal (norm.ppf), asumiendo residuos gaussianos: la interpretación es
    cuantitativa siempre que ese supuesto se sostenga.

    param results: Lista de diccionarios de resultados, uno por modelo evaluado.
    param output_path: Ruta del archivo PNG a generar.
    return: None; la figura queda escrita en output_path.
    """
    from scipy.stats import norm

    fig, ax = plt.subplots(figsize=(8, 8))

    # Cuantiles de cobertura nominal a evaluar (5%, 10%, ..., 95%).
    quantiles = np.linspace(0.05, 0.95, 19)
    # Z-score correcto: cuantil de la normal estándar. Para IC bilateral
    # al nivel q, se usa z = Φ⁻¹((1+q)/2), donde Φ⁻¹ es la inversa de la CDF.
    # Ejemplos: q=0.5 → z=0.674, q=0.95 → z=1.960, q=0.99 → z=2.576.
    z_scores = [norm.ppf((1 + q) / 2) for q in quantiles]

    for r in results:
        empirical = []
        for z in z_scores:
            lower = r['rul_pred'] - z * r['mc_std']
            upper = r['rul_pred'] + z * r['mc_std']
            emp = np.mean((r['rul_true'] >= lower) & (r['rul_true'] <= upper))
            empirical.append(emp * 100)
        ax.plot(quantiles * 100, empirical, marker='o',
                color=r['color'], label=r['label'])

    ax.plot([0, 100], [0, 100], 'k--', linewidth=1, label='Calibración ideal')
    ax.set_xlabel('Cobertura nominal [%]')
    ax.set_ylabel('Cobertura empírica [%]')
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {output_path}")


def log_summary_table(results: list) -> None:
    """Emite por el log la tabla resumen de la comparación entre modelos.

    Deja en el log las mismas cifras que sustentan las figuras, en formato tabular.

    param results: Lista de diccionarios de resultados, uno por modelo evaluado.
    return: None; la tabla se emite por el logger del módulo.
    """
    logger.info("")
    logger.info("Resumen comparativo:")
    logger.info(f"  {'Modelo':<10} {'Params':>10} {'RMSE':>8} "
                f"{'MAE':>8} {'DegAvg':>8} {'IC95%':>8}")
    for r in results:
        logger.info(f"  {r['label']:<10} {r['params']:>10,} "
                    f"{r['rmse']:>8.2f} {r['mae']:>8.2f} "
                    f"{np.mean(r['deg_rmses']):>8.4f} "
                    f"{r['coverage']:>7.1f}%")


def main() -> None:
    """Punto de entrada que evalúa los tres modelos y genera las cuatro figuras.

    Construye un único DataLoader de validación y lo reutiliza para todos los
    modelos: comparar arquitecturas exige que vean exactamente las mismas ventanas,
    de lo contrario las diferencias observadas podrían venir del reparto de datos.
    Los checkpoints ausentes se avisan y se saltan, y solo se aborta si no queda
    ninguno evaluable.

    return: None; sale sin generar figuras si no hay checkpoints disponibles, y en
        caso contrario deja los PNG en results/figures.
    """
    logger.info("Generación de figuras comparativas RNN")
    logger.info(f"Checkpoints en: {CHECKPOINT_DIR}")
    logger.info(f"Figuras en:     {FIG_DIR}")

    loader = CMAPSSLoader(data_path=str(CMAPSS_DIR))
    monitor = HealthMonitor(hidden_dim=64, rnn_type='lstm')
    _, val_loader = monitor.prepare_cmapss(loader, subset='FD001')

    results = []
    for cfg in MODELS:
        path = CHECKPOINT_DIR / cfg['checkpoint']
        if not path.exists():
            logger.warning(f"Checkpoint no encontrado: {path}")
            continue
        logger.info(f"Evaluando {cfg['label']}...")
        results.append(evaluate_checkpoint(cfg, val_loader))

    if not results:
        logger.error("No hay checkpoints evaluables; se aborta.")
        return

    plot_metrics_comparison(
        results, FIG_DIR / 'fig15_rnn_metrics_comparison.png')
    plot_rul_scatter(
        results, FIG_DIR / 'fig16_rul_scatter_comparison.png')
    plot_degradation_tracking(
        results, FIG_DIR / 'fig17_degradation_tracking.png')
    plot_uncertainty_calibration(
        results, FIG_DIR / 'fig18_uncertainty_calibration.png')

    log_summary_table(results)

if __name__ == '__main__':
    main()