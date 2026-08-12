"""
Última Fecha de Modificación: 08/Aug/2026
Descripción train_pinn_ablation.py: Interfaz de línea de comandos que entrena
las dos variantes ablacionadas del gemelo digital para el estudio de ablation
del proyecto. Ambas comparten corpus, partición, optimizador y criterio de
parada con el modelo completo, de modo que la diferencia de MAPE resultante
pueda atribuirse al ingrediente retirado y no al montaje experimental:

    1. BraytonPINNNoCL         — sin constraint layers (arquitectura 14 outputs).
                                 Genera pinn_no_cl.pt. HALLAZGO: MAPE 7.31%.
    2. BraytonPINN + PINNLossDataOnly — arquitectura completa, pérdida sin
                                 física (etiqueta 'BraytonPINNNoPhysics').
                                 Genera pinn_no_physics.pt. HALLAZGO: MAPE 4.62%.

Comparación con el modelo completo (BraytonPINN + PINNLoss): MAPE 4.64%,
838,443 parámetros, arquitectura 224/8. Los tres checkpoints se guardan en
`checkpoints/digital_twin/` con nomenclatura clara para su uso en scripts de
evaluación posteriores.

Reproducibilidad: los tres entrenamientos fijan torch.manual_seed(42) y
np.random.seed(42) ANTES de instanciar el modelo, coherentes con
ACEDigitalTwin.__init__, para que las tres configuraciones partan de la misma
semilla base y la comparación entre ellas sea justa.

Uso:
    python src/training/train_pinn_ablation.py
    python src/training/train_pinn_ablation.py --epochs 600 --patience 80
    python src/training/train_pinn_ablation.py --only no_cl
    python src/training/train_pinn_ablation.py --only no_physics
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.pinn import ACEDigitalTwin  # noqa: E402
from src.models.pinn_ablated import (  # noqa: E402
    BraytonPINNNoCL,
    PINNLossDataOnly,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


DEFAULT_DATASET = Path('data/synthetic/ace_dataset_5000.csv')
CHECKPOINT_DIR = Path('checkpoints/digital_twin')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data_no_cl(df: pd.DataFrame, model: BraytonPINNNoCL,
                       train_ratio: float = 0.8) -> tuple:
    """Prepara DataLoaders para el modelo sin constraint layers.

    A diferencia del pipeline del modelo completo, aquí la
    normalización interna cubre las 14 variables de salida (todas
    predichas), no las 11 variables internas.

    Existe como función aparte porque ACEDigitalTwin.prepare_data() asume la
    arquitectura con constraint layers. La partición reproduce la del modelo completo,
    con la misma semilla y proporción, para que ambas variantes se evalúen exactamente
    sobre las mismas muestras de validación.

    param df: Corpus con las columnas de entrada y de salida del modelo.
    param model: Instancia de BraytonPINNNoCL en la que fijar la normalización.
    param train_ratio: Fracción de filas destinada a entrenamiento.
    return: Tupla (train_loader, val_loader) con lotes de 32 y 64 muestras.
    """
    X = df[BraytonPINNNoCL.INPUT_FEATURES].values.astype(np.float32)
    y_cols = [c for c in BraytonPINNNoCL.OUTPUT_FEATURES
              if c in df.columns]
    y = df[y_cols].values.astype(np.float32)

    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)

    model.set_normalization(
        torch.tensor(X_mean), torch.tensor(X_std),
        torch.tensor(y_mean), torch.tensor(y_std),
    )

    n_train = int(len(X) * train_ratio)
    indices = np.random.RandomState(42).permutation(len(X))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X[train_idx]), torch.tensor(y[train_idx]))
    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(X[val_idx]), torch.tensor(y[val_idx]))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=64, shuffle=False)

    return train_loader, val_loader


def train_no_cl(df: pd.DataFrame, epochs: int, patience: int,
                lr: float, hidden_dim: int, n_layers: int) -> Path:
    """Entrena la variante sin constraint layers con su propio bucle de optimización.

    Implementa aquí el bucle en lugar de reutilizar ACEDigitalTwin porque esta
    variante no tiene salidas internas ni pérdida física: solo MSE contra el corpus.
    La métrica de selección excluye sfc, farB y P2, las tres variables que el modelo
    completo calcula de forma exacta, para que el MAPE de ambas configuraciones se
    refiera al mismo subconjunto de variables y sea directamente comparable.

    param df: Corpus sobre el que entrenar y validar.
    param epochs: Número máximo de épocas.
    param patience: Épocas sin mejora del MAPE medio antes de parar.
    param lr: Tasa de aprendizaje inicial del optimizador AdamW.
    param hidden_dim: Dimensión oculta de los bloques residuales.
    param n_layers: Número de bloques residuales apilados.
    return: Ruta del checkpoint guardado, correspondiente a la mejor época.
    """
    checkpoint_path = CHECKPOINT_DIR / 'pinn_no_cl.pt'
    logger.info("Entrenando BraytonPINNNoCL (sin constraint layers)")
    logger.info(f"  Checkpoint: {checkpoint_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(42)
    np.random.seed(42)

    model = BraytonPINNNoCL(
        hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    train_loader, val_loader = prepare_data_no_cl(df, model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200, eta_min=1e-6)
    loss_fn = torch.nn.MSELoss()

    best_val_mape = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Val
        model.eval()
        with torch.no_grad():
            preds, targets = [], []
            for X_batch, y_batch in val_loader:
                preds.append(model(X_batch.to(device)).cpu())
                targets.append(y_batch)
            preds = torch.cat(preds)
            targets = torch.cat(targets)

        excluded = {'sfc', 'farB', 'P2'}  # comparables al modelo completo
        mapes = []
        for i, name in enumerate(BraytonPINNNoCL.OUTPUT_FEATURES):
            if name in excluded:
                continue
            mape = (torch.abs(preds[:, i] - targets[:, i])
                    / targets[:, i].clamp(min=1e-8)).mean() * 100
            mapes.append(mape.item())
        avg_mape = float(np.mean(mapes))

        if avg_mape < best_val_mape:
            best_val_mape = avg_mape
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch + 1:>3}/{epochs} | "
                        f"MAPE medio: {avg_mape:.2f}% | "
                        f"Mejor: {best_val_mape:.2f}%")

        if patience_counter >= patience:
            logger.info(f"  Early stopping en epoch {epoch + 1}")
            break

    logger.info(f"BraytonPINNNoCL completado. "
                f"Mejor MAPE medio: {best_val_mape:.2f}%")
    return checkpoint_path


def train_no_physics(df: pd.DataFrame, epochs: int, patience: int,
                     lr: float, hidden_dim: int,
                     n_layers: int) -> Path:
    """Entrena la variante con arquitectura completa pero sin término físico.

    Reutiliza ACEDigitalTwin y se limita a sustituir el criterio por la pérdida
    solo-datos. Esa sustitución quirúrgica es justamente lo que hace válido el
    experimento: optimizador, planificador, partición y protocolo de parada quedan
    idénticos a los del modelo completo, y la única variable que cambia es la física.

    param df: Corpus sobre el que entrenar y validar.
    param epochs: Número máximo de épocas.
    param patience: Épocas sin mejora antes de detener el entrenamiento.
    param lr: Tasa de aprendizaje inicial del optimizador AdamW.
    param hidden_dim: Dimensión oculta de los bloques residuales.
    param n_layers: Número de bloques residuales apilados.
    return: Ruta del checkpoint guardado, correspondiente a la mejor época.
    """
    checkpoint_path = CHECKPOINT_DIR / 'pinn_no_physics.pt'
    logger.info("Entrenando BraytonPINNNoPhysics "
                "(arquitectura completa, pérdida sin física)")
    logger.info(f"  Checkpoint: {checkpoint_path}")

    # Se reutiliza ACEDigitalTwin (mismo optimizador, scheduler,
    # protocolo de fit) pero sustituyendo la loss por PINNLossDataOnly.
    twin = ACEDigitalTwin(hidden_dim=hidden_dim,
                          n_layers=n_layers, lr=lr)
    twin.criterion = PINNLossDataOnly()

    train_loader, val_loader = twin.prepare_data(df)

    twin.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=patience,
        checkpoint_path=str(checkpoint_path),
    )
    return checkpoint_path


def main() -> None:
    """Punto de entrada por línea de comandos del estudio de ablación.

    Analiza los argumentos, carga el corpus una sola vez y lanza las variantes
    seleccionadas. El argumento --only permite reentrenar una sola configuración sin
    repetir la otra, algo necesario en la práctica porque cada entrenamiento es largo
    y con frecuencia solo hace falta rehacer uno de los dos.

    return: None; sale sin entrenar si el dataset no existe, y en caso contrario deja
        los checkpoints de las variantes en checkpoints/digital_twin.
    """
    parser = argparse.ArgumentParser(
        description="Entrena las variantes ablated del PINN")
    parser.add_argument('--epochs', type=int, default=600,
                        help="Número máximo de epochs")
    parser.add_argument('--patience', type=int, default=80,
                        help="Paciencia para early stopping")
    parser.add_argument('--hidden', type=int, default=224,
                        help="Dimensión oculta de los bloques residuales "
                             "(224 reproduce los checkpoints del ablation)")
    parser.add_argument('--layers', type=int, default=8,
                        help="Número de bloques residuales "
                             "(8 reproduce los checkpoints del ablation)")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate inicial de AdamW "
                             "(1e-3 reproduce los checkpoints del ablation)")
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET,
                        help=f"Ruta al corpus (default: {DEFAULT_DATASET})")
    parser.add_argument('--only', choices=['no_cl', 'no_physics', 'both'],
                        default='both',
                        help="Qué variante entrenar")
    args = parser.parse_args()

    if not args.dataset.exists():
        logger.error(f"Dataset no encontrado: {args.dataset}")
        return

    df = pd.read_csv(args.dataset)
    logger.info(f"Corpus: {args.dataset} ({df.shape[0]} muestras)")
    logger.info(f"Configuración: epochs = {args.epochs}, "
                f"patience = {args.patience}, hidden = {args.hidden}, "
                f"layers = {args.layers}, lr = {args.lr}")

    if args.only in ('no_cl', 'both'):
        train_no_cl(df, args.epochs, args.patience,
                    args.lr, args.hidden, args.layers)

    if args.only in ('no_physics', 'both'):
        train_no_physics(df, args.epochs, args.patience,
                         args.lr, args.hidden, args.layers)

    logger.info("Entrenamientos de ablation completados")


if __name__ == '__main__':
    main()