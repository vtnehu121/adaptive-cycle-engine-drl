"""
Última Fecha de Modificación: 19/Aug/2026
Descripción train_rnn.py: Interfaz de línea de comandos para entrenar el monitor
de salud recurrente sobre el dataset NASA C-MAPSS. Construye el HealthMonitor
(LSTM o GRU con cabezal dual RUL + degradación) con la variante y el tamaño
pedidos, y lanza el entrenamiento con early stopping sobre el RMSE de vida
remanente. Es el script que genera los tres checkpoints comparados en el
apartado del monitor de salud del proyecto (LSTM-128, LSTM-64 y GRU-64), cada
uno con un nombre derivado de su propia configuración.

Reproducibilidad: HealthMonitor.__init__ fija internamente torch.manual_seed(42)
y np.random.seed(42) antes de instanciar la red, para que las tres arquitecturas
comparadas partan de la misma semilla base y la comparación entre ellas sea
justa. Por esa razón este CLI no expone la semilla como argumento: cambiarla
rompería la equivalencia entre los tres checkpoints.

Las figuras comparativas de las tres RNN se generan aparte con
`scripts/plotting/plot_rnn_comparison.py` (val_loader compartido, comparativa
oficial); las figuras específicas de una única RNN (curvas de entrenamiento,
MC-Dropout, tracking por componente) con `scripts/plotting/plot_rnn_single.py`.

Uso:
    python src/training/train_rnn.py
    python src/training/train_rnn.py
    python src/training/train_rnn.py --rnn lstm --hidden 128
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.rnn_health import HealthMonitor  # noqa: E402
from src.preprocessing.cmapss_loader import CMAPSSLoader  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path('checkpoints/health_monitoring')
CMAPSS_DIR = Path('data/CMAPSS_DATA')


def build_checkpoint_name(rnn_type: str, hidden: int) -> str:
    """Deriva el nombre del checkpoint a partir de la configuración de la red.

    Codificar tipo y tamaño en el nombre es lo que permite entrenar las tres
    variantes en la misma carpeta sin que la última sobrescriba a las anteriores, y
    que los scripts de comparación las localicen después por convención.

    param rnn_type: Variante recurrente empleada, 'lstm' o 'gru'.
    param hidden: Dimensión oculta de la red.
    return: Nombre de archivo del checkpoint, con la forma rnn_{tipo}_{hidden}.pt.
    """
    return f'rnn_{rnn_type}_{hidden}.pt'


def main() -> None:
    """Punto de entrada por línea de comandos del entrenamiento del monitor de salud.

    Comprueba que el dataset esté disponible, resuelve la ruta del checkpoint, carga
    C-MAPSS y lanza el entrenamiento. Exponer arquitectura, longitud de ventana y peso
    de la pérdida dual como argumentos es lo que permite reproducir con una sola orden
    cualquiera de las variantes comparadas, sin tocar el código.

    return: None; sale sin entrenar si falta el dataset, y en caso contrario deja el
        checkpoint escrito en checkpoints/health_monitoring.
    """
    parser = argparse.ArgumentParser(
        description="Entrenamiento del módulo Health Monitoring RNN")
    parser.add_argument('--rnn', choices=['lstm', 'gru'], default='gru',
                        help="Tipo de RNN")
    parser.add_argument('--hidden', type=int, default=64,
                        help="Dimensión oculta")
    parser.add_argument('--layers', type=int, default=2,
                        help="Número de capas apiladas")
    parser.add_argument('--seq-len', type=int, default=30,
                        help="Longitud de las ventanas deslizantes")
    parser.add_argument('--subset', default='FD001',
                        help="Sub-dataset C-MAPSS a usar")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Epochs máximos")
    parser.add_argument('--patience', type=int, default=20,
                        help="Paciencia para early stopping")
    parser.add_argument('--alpha', type=float, default=0.3,
                        help="Peso del término L_degradación")
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help="Ruta del checkpoint (autogenerada si es None)")
    args = parser.parse_args()

    if not CMAPSS_DIR.exists():
        logger.error(f"Dataset C-MAPSS no encontrado en {CMAPSS_DIR}")
        return

    checkpoint_path = args.checkpoint or (
        CHECKPOINT_DIR / build_checkpoint_name(args.rnn, args.hidden))

    logger.info(f"Configuración: {args.rnn.upper()}, "
                f"hidden = {args.hidden}, layers = {args.layers}, "
                f"seq_len = {args.seq_len}, alpha = {args.alpha}")
    logger.info(f"Sub-dataset: {args.subset}")
    logger.info(f"Checkpoint:  {checkpoint_path}")
    logger.info("Reproducibilidad: HealthMonitor fija torch/numpy seed=42 "
                "internamente para comparación justa entre arquitecturas.")

    loader = CMAPSSLoader(data_path=str(CMAPSS_DIR))
    loader.load_all()

    monitor = HealthMonitor(
        rnn_type=args.rnn,
        hidden_dim=args.hidden,
        n_layers=args.layers,
        sequence_length=args.seq_len,
        alpha=args.alpha,
    )
    monitor.prepare_cmapss(loader, subset=args.subset)

    monitor.fit(
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=str(checkpoint_path),
    )


if __name__ == '__main__':
    main()