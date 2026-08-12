"""
Última Fecha de Modificación: 08/Aug/2026
Descripción eval_pde_residuals.py: Script que mide la consistencia física del
gemelo digital fuera del conjunto de entrenamiento. Carga el checkpoint activo,
reconstruye la partición de test y evalúa sobre ella cada residuo del ciclo
Brayton impuesto por BraytonPhysicsLoss (conservación de masa, balance de
energía, ecuación de Newton para rotores, etc.).

Complementa a eval_pinn.py: aquel mide cuánto se aproxima el modelo a los datos
y este cuánto respeta las leyes termodinámicas, dos cosas que un regresor
puramente supervisado puede satisfacer de forma muy desigual. Valores de residuo
próximos a cero indican consistencia física alta; valores grandes revelan zonas
del espacio de operación donde el PINN es termodinámicamente inconsistente.

Uso:
    python scripts/evaluation/eval_pde_residuals.py
    python scripts/evaluation/eval_pde_residuals.py --seed 42 --split 0.8

Nota sobre la interpretación de los residuos:
    Los residuos locales (mass, energy, isentropic, combustion) se calculan
    punto a punto y suelen ser pequeños (~0.01-0.05). El residuo de thrust
    (pde_thrust) integra efectos globales del ciclo Brayton completo y
    tiende a ser 1-2 órdenes de magnitud mayor, especialmente en régimen
    supersónico donde T-MATS extrapola fuera de calibración (Mach > 1.5).
    Un residuo alto en pde_thrust NO indica un modelo defectuoso: refleja
    la dificultad estructural de aproximar el empuje bajo extrapolación,
    y es coherente con los edge cases documentados en corpus_analysis.py.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

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

import numpy as np
import pandas as pd
import torch

# Añade la raíz del proyecto al path para importar src.models.pinn
# scripts/evaluation/eval_pde_residuals.py → sube 3 niveles → raíz del repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.pinn import BraytonPINN, BraytonPhysicsLoss

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
DEFAULT_SEED = 42
DEFAULT_TRAIN_SPLIT = 0.80


def detect_architecture(state_dict: dict) -> tuple[int, int]:
    """Deduce la dimensión oculta y el número de bloques a partir del state_dict.

    Permite evaluar cualquiera de los checkpoints del proyecto sin declarar su
    arquitectura, evitando el error silencioso de instanciar un modelo con una forma
    que no corresponde a los pesos que se van a cargar.

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


def load_pinn(checkpoint_path: Path) -> BraytonPINN:
    """Carga un checkpoint del PINN y lo deja listo para inferencia.

    Acepta tanto state_dict planos como checkpoints envueltos en diccionarios con
    metadatos, ya que ambos formatos conviven en el proyecto, y usa carga no estricta
    para tolerar diferencias en los buffers de normalización.

    param checkpoint_path: Ruta del archivo de pesos a cargar.
    return: Modelo BraytonPINN con los pesos cargados y en modo evaluación.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt if isinstance(ckpt, dict) and 'input_proj.0.weight' in ckpt \
        else ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))

    hidden_dim, n_layers = detect_architecture(state_dict)
    logger.info(f"Arquitectura detectada: hidden_dim={hidden_dim}, "
                f"n_layers={n_layers}")

    model = BraytonPINN(hidden_dim=hidden_dim, n_layers=n_layers)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def split_test_indices(n_samples: int, train_split: float,
                       seed: int) -> np.ndarray:
    """Reconstruye los índices del conjunto de test empleado al entrenar.

    Reproduce la misma permutación con semilla fija que usó el entrenamiento, lo que
    garantiza que los residuos se midan sobre puntos que el modelo no vio: evaluarlos
    sobre datos de entrenamiento diría poco, porque allí la pérdida física formó parte
    explícita del objetivo optimizado.

    param n_samples: Número de filas del corpus.
    param train_split: Fracción de filas que se destinaron a entrenamiento.
    param seed: Semilla de la permutación.
    return: Array con los índices de las filas del conjunto de test.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_samples)
    if not 0 < train_split < 1:
        raise ValueError(f"train_split debe estar en (0, 1), no {train_split}")
    n_train = int(train_split * n_samples)
    return idx[n_train:]


def compute_residuals(model: BraytonPINN,
                      X: torch.Tensor) -> tuple[float, dict]:
    """Evalúa la pérdida física del modelo y su desglose sobre las muestras dadas.

    Solo necesita las entradas, no las etiquetas: los residuos se calculan entre las
    propias salidas predichas y por eso pueden medirse en cualquier punto del
    envelope. El desglose por componente es lo que permite localizar qué ley concreta
    incumple el modelo, en lugar de quedarse en un único número agregado.

    param model: Modelo PINN cargado y en modo evaluación.
    param X: Tensor de entradas sobre las que evaluar los residuos.
    return: Tupla (pérdida física total como float, diccionario con el valor de cada
        residuo y regularizador por separado).
    """
    physics = BraytonPhysicsLoss()
    with torch.no_grad():
        y_pred = model(X)
        total_loss, components = physics(y_pred)
    return total_loss.item(), components


def main() -> None:
    """Punto de entrada por línea de comandos de la evaluación de residuos físicos.

    Comprueba que existan checkpoint y corpus, reconstruye la partición de test,
    evalúa los residuos y los tabula ordenados. Que la fracción de entrenamiento y la
    semilla sean argumentos permite verificar que el resultado no depende de un
    reparto concreto y descartar así que los residuos bajos sean casualidad.

    return: None; sale sin evaluar si falta el checkpoint o el dataset, y en caso
        contrario emite los residuos por el logger.
    """
    parser = argparse.ArgumentParser(
        description="Evalúa residuos PDE del PINN sobre el test set")
    parser.add_argument('--checkpoint', type=Path,
                        default=DEFAULT_CHECKPOINT,
                        help=f"Ruta al checkpoint (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET,
                        help=f"Ruta al CSV del corpus (default: {DEFAULT_DATASET})")
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f"Semilla del split (default: {DEFAULT_SEED})")
    parser.add_argument('--split', type=float, default=DEFAULT_TRAIN_SPLIT,
                        help=f"Fracción de entrenamiento (default: {DEFAULT_TRAIN_SPLIT})")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        logger.error(f"Checkpoint no encontrado: {args.checkpoint}")
        return
    if not args.dataset.exists():
        logger.error(f"Dataset no encontrado: {args.dataset}")
        return

    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Dataset:    {args.dataset}")
    logger.info(f"Split:      {args.split*100:.0f}% train / "
                f"{(1-args.split)*100:.0f}% test (seed={args.seed})")

    model = load_pinn(args.checkpoint)

    df = pd.read_csv(args.dataset)
    logger.info(f"Corpus: {len(df)} muestras")

    test_idx = split_test_indices(len(df), args.split, args.seed)
    logger.info(f"Conjunto de test: {len(test_idx)} muestras")

    # INPUT_FEATURES está definido en src/models/pinn.py como los features
    # del ciclo Brayton: T2, T4, T30, P2, P30, Nf, Nc, etc.
    X_test = torch.tensor(
        df[BraytonPINN.INPUT_FEATURES].values[test_idx],
        dtype=torch.float32,
    )

    total_loss, components = compute_residuals(model, X_test)

    logger.info("Residuos PDE por componente:")
    for k, v in sorted(components.items()):
        logger.info(f"  {k:<32}: {v:.6f}")

    logger.info("")
    logger.info(f"Pérdida física total: {total_loss:.6f}")
    logger.info("Nota: pde_thrust suele dominar la pérdida física porque integra")
    logger.info("efectos globales del ciclo Brayton, especialmente en régimen")
    logger.info("supersónico donde T-MATS extrapola fuera de calibración.")
    logger.info("Los otros residuos son locales por punto y por tanto menores.")


if __name__ == '__main__':
    main()