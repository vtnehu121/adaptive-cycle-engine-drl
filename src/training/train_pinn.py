"""
Última Fecha de Modificación: 08/Aug/2026
Descripción train_pinn.py: Interfaz de línea de comandos para entrenar el gemelo
digital PINN sobre el corpus sintético del ACE. Carga el corpus indicado, delega
en ACEDigitalTwin la preparación de DataLoaders (con la normalización aprendida
del train split), el entrenamiento supervisado + informado por física con
currículo progresivo (Wang et al., 2021) y el guardado del mejor checkpoint
según MAPE medio de validación. Termina midiendo la latencia de inferencia.

Es el script que produce el checkpoint activo del gemelo digital (arquitectura
224/8, 838,443 parámetros, MAPE 4.64% en validación, latencia P99 1.85 ms
compatible con DO-178C Level A), que después consumen el entorno de control y
todos los scripts de evaluación.

Reproducibilidad: ACEDigitalTwin.__init__ fija internamente torch.manual_seed(42)
y np.random.seed(42) antes de instanciar el PINN, para que las tres
configuraciones comparadas en el ablation (PINN completo, sin_CL, sin_PDEs)
partan de la misma semilla base. Por esa razón este CLI no expone la semilla
como argumento: cambiarla rompería la equivalencia entre los tres checkpoints.

Las figuras de validación del PINN se generan aparte con
`scripts/plotting/plot_pinn_evaluation.py`, que solo necesita el checkpoint
guardado y no requiere re-entrenar.

Uso:
    python src/training/train_pinn.py
    python src/training/train_pinn.py --epochs 600 --patience 80
    python src/training/train_pinn.py --dataset data/synthetic/ace_dataset_5000.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.pinn import ACEDigitalTwin  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path('data/synthetic/ace_dataset_5000.csv')
DEFAULT_CHECKPOINT = Path('checkpoints/digital_twin/pinn.pt')


def main() -> None:
    """Punto de entrada por línea de comandos del entrenamiento del gemelo digital.

    Valida que el corpus exista, lo carga, construye el ACEDigitalTwin con la
    arquitectura pedida y lanza el entrenamiento, cerrando con el resumen y la medida
    de latencia. Exponer arquitectura, épocas y paciencia como argumentos es lo que
    permite reproducir desde la misma orden tanto el checkpoint activo como las
    variantes exploradas durante el ajuste, sin editar código.

    return: None; sale sin entrenar si el dataset indicado no existe, y en caso
        contrario deja el checkpoint y el historial escritos en disco.
    """
    parser = argparse.ArgumentParser(
        description="Entrenamiento del Gemelo Digital PINN del ACE")
    parser.add_argument('--epochs', type=int, default=600,
                        help="Número máximo de epochs")
    parser.add_argument('--patience', type=int, default=80,
                        help="Paciencia para early stopping")
    parser.add_argument('--hidden', type=int, default=224,
                        help="Dimensión oculta de los bloques residuales "
                             "(224 reproduce el checkpoint activo)")
    parser.add_argument('--layers', type=int, default=8,
                        help="Número de bloques residuales "
                             "(8 reproduce el checkpoint activo)")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate inicial de AdamW "
                             "(1e-3 reproduce el checkpoint activo)")
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET,
                        help=f"Ruta al corpus (default: {DEFAULT_DATASET})")
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT,
                        help=f"Ruta del checkpoint (default: {DEFAULT_CHECKPOINT})")
    args = parser.parse_args()

    if not args.dataset.exists():
        logger.error(f"Dataset no encontrado: {args.dataset}")
        return

    logger.info(f"Corpus:     {args.dataset}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Arquitectura: hidden_dim = {args.hidden}, "
                f"n_layers = {args.layers}, lr = {args.lr}")

    df = pd.read_csv(args.dataset)
    logger.info(f"Datos cargados: {df.shape[0]} muestras, "
                f"{df.shape[1]} columnas")

    twin = ACEDigitalTwin(
        hidden_dim=args.hidden,
        n_layers=args.layers,
        lr=args.lr,
    )
    train_loader, val_loader = twin.prepare_data(df)

    twin.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=str(args.checkpoint),
    )

    logger.info(twin.get_training_summary())
    twin.measure_latency()


if __name__ == '__main__':
    main()