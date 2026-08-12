"""
Última Fecha de Modificación: 08/Aug/2026
Descripción plot_pinn_evaluation.py: Script que genera las figuras de evaluación
del gemelo digital PINN. Produce las curvas de entrenamiento con el currículo de 
pesos físicos, los diagramas de dispersión que contrastan la predicción con la 
referencia T-MATS variable a variable y la comprobación gráfica de la relación 
isentrópica del compresor. Trabaja únicamente en modo inferencia sobre el 
checkpoint guardado, sin reentrenar nada.

Las curvas de entrenamiento (fig28) se leen del historial guardado en el
checkpoint. Si el checkpoint solo contiene los pesos (sin historial), esa figura
se omite y solo se generan las de fidelidad y cumplimiento PDE.

Figuras generadas en results/figures/:
    fig28_pinn_training_curves.png       Loss total, componentes, ω₂/ω₃
    fig29_pinn_vs_tmats.png              Scatter fidelidad por variable
    fig30_pde_isentropic_compliance.png  Cumplimiento PDE 1

Uso:
    python scripts/plotting/plot_pinn_evaluation.py
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

DEFAULT_CHECKPOINT = Path('checkpoints/digital_twin/pinn.pt')
DEFAULT_DATASET = Path('data/synthetic/ace_dataset_5000.csv')
FIG_DIR = Path('results/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# (γ - 1) / γ para aire frío del compresor. γ = 1.4 es el valor estándar
# antes de la combustión (aire seco). Después del combustor se usa γ = 1.33
# (gases calientes), pero aquí la relación isentrópica solo se comprueba
# en el COMPRESOR, donde el aire aún no ha sido quemado.
GAMMA_RATIO_COLD = 0.2857  

plt.rcParams.update({
    'figure.dpi':          150,
    'font.size':           12,
    'font.family':         'serif',
    'axes.titlesize':      13,
    'axes.labelsize':      12,
    'xtick.labelsize':     10,
    'ytick.labelsize':     10,
    'legend.fontsize':     10,
    'axes.grid':           True,
    'grid.alpha':          0.3,
    'axes.spines.top':     False,
    'axes.spines.right':   False,
})

SCATTER_COLORS = [
    '#2A9D8F', '#E63946', '#457B9D', '#E9C46A', '#264653',
    '#8338EC', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
    '#F4A261', '#A8DADC', '#1D3557', '#6A4C93',
]


def detect_architecture(state_dict: dict) -> tuple[int, int]:
    """Deduce la dimensión oculta y el número de bloques a partir del state_dict.

    Evita tener que declarar la arquitectura al generar las figuras, de modo que el
    script sigue funcionando aunque el checkpoint activo se reentrene con una
    configuración distinta.

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


def load_pinn(checkpoint_path: Path) -> tuple[BraytonPINN, list | None]:
    """Carga el modelo y recupera además el historial de entrenamiento si existe.

    Busca el historial primero dentro del propio checkpoint y, si no está, en el CSV
    que el entrenamiento deja junto a él. Esa doble vía es necesaria porque los
    checkpoints guardados como state_dict plano no contienen historial, y sin él la
    figura de curvas de entrenamiento no podría generarse.

    param checkpoint_path: Ruta del archivo de pesos a cargar.
    return: Tupla (modelo en modo evaluación, lista de registros del historial por
        época, o None si no se encontró historial por ninguna de las dos vías).
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'input_proj.0.weight' in ckpt:
        state_dict = ckpt
        history = None
    else:
        state_dict = ckpt.get('state_dict',
                              ckpt.get('model_state_dict', ckpt))
        history = ckpt.get('history') if isinstance(ckpt, dict) else None

    # Si el checkpoint no tiene historial, intentar leer del CSV externo
    if history is None:
        csv_path = checkpoint_path.parent / 'training_history_pinn.csv'
        if csv_path.exists():
            df_hist = pd.read_csv(csv_path)
            history = df_hist.to_dict('records')
            logger.info(f"Historial cargado desde CSV: {csv_path}")

    hidden_dim, n_layers = detect_architecture(state_dict)
    logger.info(f"Arquitectura detectada: hidden_dim = {hidden_dim}, "
                f"n_layers = {n_layers}")
    model = BraytonPINN(hidden_dim=hidden_dim, n_layers=n_layers)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, history


def plot_training_curves(history: list, output_path: Path) -> None:
    """Dibuja la figura de curvas de entrenamiento del gemelo digital.

    Compone tres paneles: pérdida total en train y validación, desglose por
    componente y evolución de los pesos del currículo. Es necesario mostrar los tres
    juntos para poder leer la figura como una sola historia: se ve cómo, al crecer ω₂
    y ω₃, la pérdida física baja y la de datos se reajusta, que es exactamente el
    comportamiento que el currículo pretende inducir. Las escalas son logarítmicas
    porque los términos difieren en varios órdenes de magnitud.

    param history: Lista de registros por época con las claves de pérdida y pesos.
    param output_path: Ruta del archivo PNG a generar.
    return: None; la figura queda escrita en output_path.
    """
    epochs = [h['epoch'] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, [h['train_loss'] for h in history], label='Train')
    axes[0].plot(epochs, [h['val_loss'] for h in history], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss total')
    axes[0].set_yscale('log')
    axes[0].legend()

    axes[1].plot(epochs, [h['train_data'] for h in history],
                 label='L_data')
    axes[1].plot(epochs, [h['train_physics'] for h in history],
                 label='L_physics')
    axes[1].plot(epochs, [h['train_collocation'] for h in history],
                 label='L_colloc', linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Componente')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=9)

    axes[2].plot(epochs, [h['omega2'] for h in history], 'r-',
                 label='ω₂ (physics)')
    axes[2].plot(epochs, [h['omega3'] for h in history], 'b-',
                 label='ω₃ (colocación)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Peso')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {output_path}")


def plot_pinn_vs_tmats(model: BraytonPINN, df: pd.DataFrame,
                       output_path: Path) -> None:
    """Dibuja la rejilla de dispersión que compara el PINN con la referencia T-MATS.

    Un panel por variable, con la diagonal ideal y el R² y el MAPE anotados. Es
    necesario porque un único error promedio no revela el tipo de fallo: la nube de
    puntos muestra si el modelo tiene sesgo sistemático, si pierde precisión solo en
    los extremos del rango o si el error se concentra en unas pocas variables. Los
    paneles sobrantes de la rejilla se ocultan para no dejar ejes vacíos.

    param model: Modelo PINN cargado y en modo evaluación.
    param df: Corpus con las columnas de entrada y las de referencia.
    param output_path: Ruta del archivo PNG a generar.
    return: None; la figura queda escrita en output_path.
    """
    X = torch.tensor(
        df[BraytonPINN.INPUT_FEATURES].values.astype(np.float32))
    y_cols = [c for c in BraytonPINN.OUTPUT_FEATURES if c in df.columns]
    y_true = df[y_cols].values.astype(np.float32)

    with torch.no_grad():
        y_pred = model(X).numpy()

    n_vars = min(len(y_cols), y_pred.shape[1])

    fig, axes = plt.subplots(3, 5, figsize=(26, 15))

    for idx in range(n_vars):
        ax = axes[idx // 5, idx % 5]
        real = y_true[:, idx]
        pred = y_pred[:, idx]
        color = SCATTER_COLORS[idx % len(SCATTER_COLORS)]

        ax.scatter(real, pred, s=8, alpha=0.5,
                   c=color, edgecolors='none')

        vmin = min(real.min(), pred.min())
        vmax = max(real.max(), pred.max())
        margin = (vmax - vmin) * 0.05
        ax.plot([vmin - margin, vmax + margin],
                [vmin - margin, vmax + margin],
                'k--', linewidth=1, alpha=0.5)

        ss_res = np.sum((real - pred) ** 2)
        ss_tot = np.sum((real - real.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mape = np.mean(np.abs((real - pred)
                              / np.clip(np.abs(real), 1e-8, None))) * 100

        ax.set_xlabel(f'{y_cols[idx]} (T-MATS)')
        ax.set_ylabel(f'{y_cols[idx]} (PINN)')
        ax.set_title(f'{y_cols[idx]}\nR² = {r2:.3f}   MAPE = {mape:.1f}%',
                     fontsize=10)

    for i in range(n_vars, 15):
        axes[i // 5, i % 5].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {output_path}")


def plot_pde_isentropic_compliance(model: BraytonPINN, df: pd.DataFrame,
                                   output_path: Path) -> None:
    """Dibuja el cumplimiento de la relación isentrópica del compresor.

    Enfrenta el salto de temperaturas contra el predicho por la relación isentrópica,
    en dos paneles: el del PINN y el de la referencia T-MATS. Comparar ambos es lo que
    da sentido a la figura: el residuo del simulador fija el nivel de cumplimiento
    alcanzable, de modo que el del modelo solo puede juzgarse frente a esa referencia
    y no frente al cero ideal.

    param model: Modelo PINN cargado y en modo evaluación.
    param df: Corpus con las columnas de entrada y las de referencia.
    param output_path: Ruta del archivo PNG a generar.
    return: None; la figura queda escrita en output_path.
    """
    X = torch.tensor(
        df[BraytonPINN.INPUT_FEATURES].values.astype(np.float32))
    y_cols = [c for c in BraytonPINN.OUTPUT_FEATURES if c in df.columns]
    y_true = df[y_cols].values.astype(np.float32)

    with torch.no_grad():
        y_pred = model(X).numpy()

    def extract(y):
        """Extrae de una matriz de salidas el cociente real y el isentrópico.

        Se define localmente para aplicar exactamente el mismo cálculo a las
        predicciones y a la referencia; duplicarlo abriría la puerta a comparar dos
        magnitudes calculadas de forma sutilmente distinta. Las presiones se acotan
        por abajo para evitar potencias de valores no físicos.

        param y: Matriz de salidas con el orden de columnas de OUTPUT_FEATURES.
        return: Tupla (cociente de temperaturas T30/T2, cociente de presiones elevado
            al exponente isentrópico).
        """
        T2 = y[:, 2]
        T30 = y[:, 4]
        P2 = np.clip(y[:, 6], 1, None)
        P30 = np.clip(y[:, 7], 1, None)
        return T30 / T2, (P30 / P2) ** GAMMA_RATIO_COLD

    T_ratio_pinn,  isen_pinn  = extract(y_pred)
    T_ratio_tmats, isen_tmats = extract(y_true)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, isen, T_ratio, label, color in [
        (axes[0], isen_pinn,  T_ratio_pinn,  'PINN',   '#2A9D8F'),
        (axes[1], isen_tmats, T_ratio_tmats, 'T-MATS', '#E63946'),
    ]:
        ax.scatter(isen, T_ratio, s=15, alpha=0.6,
                   c=color, edgecolors='none')
        lim = [min(isen.min(), T_ratio.min()) * 0.95,
               max(isen.max(), T_ratio.max()) * 1.05]
        ax.plot(lim, lim, 'r--', linewidth=2, label='PDE exacta')

        residuo = np.mean(np.abs(T_ratio - isen) / isen) * 100
        ax.text(0.05, 0.95, f'Residuo medio: {residuo:.2f}%',
                transform=ax.transAxes, fontsize=11, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(r'$(P_{30}/P_2)^{0.286}$')
        ax.set_ylabel(r'$T_{30}/T_2$')
        ax.set_title(f'Predicciones {label}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {output_path}")


def main() -> None:
    """Punto de entrada que genera las tres figuras de evaluación del gemelo digital.

    Verifica que existan checkpoint y corpus, carga ambos una sola vez y encadena las
    funciones de dibujo. La figura de curvas se omite con un aviso cuando no hay
    historial disponible, de modo que la ausencia de ese dato no impide generar las
    otras dos, que solo necesitan los pesos.

    return: None; sale sin generar nada si falta el checkpoint o el dataset, y en caso
        contrario deja los PNG en results/figures.
    """
    logger.info("Generación de figuras de evaluación del PINN")
    logger.info(f"Checkpoint: {DEFAULT_CHECKPOINT}")
    logger.info(f"Corpus:     {DEFAULT_DATASET}")

    if not DEFAULT_CHECKPOINT.exists():
        logger.error(f"Checkpoint no encontrado: {DEFAULT_CHECKPOINT}")
        return
    if not DEFAULT_DATASET.exists():
        logger.error(f"Dataset no encontrado: {DEFAULT_DATASET}")
        return

    model, history = load_pinn(DEFAULT_CHECKPOINT)
    df = pd.read_csv(DEFAULT_DATASET)

    if history:
        plot_training_curves(
            history, FIG_DIR / 'fig28_pinn_training_curves.png')
    else:
        logger.warning("Checkpoint sin historial de entrenamiento; "
                       "se omite fig28.")

    plot_pinn_vs_tmats(
        model, df, FIG_DIR / 'fig29_pinn_vs_tmats.png')
    plot_pde_isentropic_compliance(
        model, df, FIG_DIR / 'fig30_pde_isentropic_compliance.png')


if __name__ == '__main__':
    main()