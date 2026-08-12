"""
Última Fecha de Modificación: 08/Aug/2026
Descripción eval_pinn.py: Script de evaluación del gemelo digital PINN sobre el
corpus sintético. Reproduce la partición train/val/test del entrenamiento,
calcula el MAPE por variable de salida y mide la latencia de inferencia,
pudiendo además contrastar train contra test para detectar sobreajuste y
recorrer varios checkpoints en una sola ejecución. Detecta la arquitectura
directamente del state_dict, de modo que no hace falta declararla al invocarlo.

Splits (80/15/5 por defecto, reproducibles con seed=42):
    train: primer 80% del corpus tras shuffle
    val:   siguiente 15%
    test:  último  5%

Uso:
    # Evaluación estándar del checkpoint activo sobre test
    python scripts/evaluation/eval_pinn.py

    # Comparar train vs test (detección de overfitting)
    python scripts/evaluation/eval_pinn.py --compare-train-test

    # Evaluar sobre validación
    python scripts/evaluation/eval_pinn.py --subset val

    # Evaluar múltiples checkpoints (ablation: activo vs sin CL vs sin PDEs)
    python scripts/evaluation/eval_pinn.py --checkpoints \\
        checkpoints/digital_twin/pinn.pt \\
        checkpoints/digital_twin/pinn_no_cl.pt \\
        checkpoints/digital_twin/pinn_no_physics.pt
"""
import argparse
import logging
import os
import sys
import time
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.pinn import BraytonPINN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

for _h in logging.root.handlers:
    if hasattr(_h, 'stream') and hasattr(_h.stream, 'reconfigure'):
        try:
            _h.stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path('checkpoints/digital_twin/pinn.pt')
DEFAULT_DATASET = Path('data/synthetic/ace_dataset_5000.csv')
DEFAULT_SEED = 42
DEFAULT_SPLITS = (0.80, 0.15, 0.05)   # train / val / test
# Umbral empírico: gap MAPE (test - train) > 2% indica que el modelo memoriza
# el conjunto de entrenamiento en detrimento de la generalización. El valor
# está calibrado para el corpus ACE (5000 muestras, MAPE nominal ~4-5%);
# proyectos con MAPE nominal muy distinto podrían requerir un umbral distinto.
OVERFIT_THRESHOLD_PCT = 2.0            # gap MAPE test-train para marcar overfit
# 500 iteraciones: compromiso entre estabilidad estadística de los percentiles
# P95/P99 (requiere ≥100 muestras) y tiempo de ejecución total (~2 segundos).
LATENCY_ITERATIONS = 500


def detect_architecture(state_dict: dict) -> tuple[int, int]:
    """Deduce la dimensión oculta y el número de bloques a partir del state_dict.

    Lee la forma de la proyección de entrada y cuenta los índices de bloque residual
    presentes en las claves. Es necesario porque los checkpoints del proyecto se
    entrenaron con arquitecturas distintas: exigir que quien evalúa recuerde y declare
    la configuración de cada uno sería una fuente segura de errores silenciosos.

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


def load_pinn(checkpoint_path: Path) -> tuple[BraytonPINN, int, int, int]:
    """Carga un checkpoint del PINN construyendo la arquitectura que le corresponde.

    Tolera tanto checkpoints guardados como state_dict plano como los envueltos en un
    diccionario con metadatos, porque a lo largo del proyecto se han generado de las
    dos formas. La carga es no estricta para admitir buffers de normalización que
    puedan faltar o sobrar respecto a la definición actual del modelo.

    param checkpoint_path: Ruta del archivo de pesos a cargar.
    return: Tupla (modelo en modo evaluación, número de parámetros, dimensión oculta,
        número de bloques residuales).
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt if isinstance(ckpt, dict) and 'input_proj.0.weight' in ckpt \
        else ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))

    hidden_dim, n_layers = detect_architecture(state_dict)
    model = BraytonPINN(hidden_dim=hidden_dim, n_layers=n_layers)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    n_params = sum(p.numel() for p in state_dict.values()
                   if hasattr(p, 'numel'))
    return model, n_params, hidden_dim, n_layers


def split_indices(n_samples: int, splits: tuple[float, float, float],
                  seed: int) -> dict[str, np.ndarray]:
    """Genera los índices de las particiones de forma reproducible.

    Usa un generador con semilla explícita para reconstruir exactamente la misma
    permutación que se empleó al entrenar. Es la condición sin la cual la evaluación
    carecería de sentido: si el reparto cambiase, el conjunto de test podría contener
    muestras ya vistas por el modelo y el error reportado sería optimista.

    param n_samples: Número de filas del corpus a repartir.
    param splits: Tupla de fracciones (train, val, test).
    param seed: Semilla de la permutación.
    return: Diccionario con los arrays de índices 'train', 'val', 'test' y 'full'.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_samples)

    n_train = int(splits[0] * n_samples)
    n_val = int(splits[1] * n_samples)

    return {
        'train': idx[:n_train],
        'val':   idx[n_train:n_train + n_val],
        'test':  idx[n_train + n_val:],
        'full':  idx,
    }


def compute_mape(model: BraytonPINN, X: torch.Tensor,
                 Y: np.ndarray, y_cols: list) -> dict[str, float]:
    """Calcula el error porcentual absoluto medio de cada variable de salida.

    Empareja cada columna del corpus con su posición en las salidas del modelo y
    descarta las muestras con valor verdadero prácticamente nulo, ya que el MAPE no
    está definido en ellas y una sola división por un valor diminuto bastaría para
    disparar la media de la variable entera.

    param model: Modelo PINN cargado y en modo evaluación.
    param X: Tensor de entradas del subconjunto a evaluar.
    param Y: Array de valores de referencia del subconjunto.
    param y_cols: Nombres de las columnas de Y, en su mismo orden.
    return: Diccionario {variable: MAPE en porcentaje}, omitiendo las variables que
        el modelo no predice o que carecen de muestras válidas.
    """
    with torch.no_grad():
        preds = model(X).numpy()

    mapes = {}
    for i, col in enumerate(y_cols):
        if col not in BraytonPINN.OUTPUT_FEATURES:
            continue
        idx_out = BraytonPINN.OUTPUT_FEATURES.index(col)
        if idx_out >= preds.shape[1]:
            continue
        y_true = Y[:, i]
        y_pred = preds[:, idx_out]
        mask = np.abs(y_true) > 1e-8
        if not mask.any():
            continue
        mape = np.mean(np.abs(y_pred[mask] - y_true[mask])
                       / np.abs(y_true[mask])) * 100
        mapes[col] = mape
    return mapes


def measure_latency(model: BraytonPINN, X_sample: torch.Tensor,
                    iterations: int = LATENCY_ITERATIONS) -> dict[str, float]:
    """Mide la latencia de inferencia del modelo sobre una única muestra.

    Cronometra consultas individuales, no lotes, porque así es como consulta el
    entorno de control al gemelo digital en cada paso. Se reportan percentiles además
    de la media, ya que el requisito de tiempo real lo marca la cola de la
    distribución y no el caso promedio.

    param model: Modelo PINN cargado y en modo evaluación.
    param X_sample: Tensor de entradas del que se toma la primera fila.
    param iterations: Número de inferencias cronometradas.
    return: Diccionario con la latencia media y los percentiles p50, p95 y p99, en
        milisegundos.
    """
    x1 = X_sample[:1]
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x1)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        'mean': float(np.mean(times)),
        'p50':  times[len(times) // 2],
        'p95':  times[int(0.95 * len(times))],
        'p99':  times[int(0.99 * len(times))],
    }


def log_mape_table(mapes: dict[str, float], title: str) -> None:
    """Emite por el log la tabla de MAPE por variable y su promedio.

    Presenta el resultado en columnas alineadas para que pueda trasladarse a la
    memoria sin reformatear, y añade el promedio como cifra resumen de la calidad
    global del gemelo digital.

    param mapes: Diccionario {variable: MAPE en porcentaje}.
    param title: Encabezado que describe el subconjunto evaluado.
    return: None; la tabla se emite por el logger del módulo.
    """
    logger.info(f"{title}:")
    for col, mape in mapes.items():
        logger.info(f"  {col:<20}: {mape:6.2f}%")
    if mapes:
        avg = np.mean(list(mapes.values()))
        logger.info(f"  {'PROMEDIO':<20}: {avg:6.2f}%")


def log_comparison_table(train_mapes: dict, test_mapes: dict) -> None:
    """Emite la tabla comparativa train frente a test y señala el sobreajuste.

    Marca las variables cuyo error crece más de OVERFIT_THRESHOLD_PCT puntos al pasar
    de entrenamiento a test. Es necesario porque el sobreajuste del gemelo digital
    rara vez es uniforme: suele concentrarse en unas pocas variables, y el promedio
    global por sí solo lo enmascararía.

    param train_mapes: MAPE por variable sobre el conjunto de entrenamiento.
    param test_mapes: MAPE por variable sobre el conjunto de test.
    return: None; la tabla y el promedio se emiten por el logger del módulo.
    """
    logger.info(f"  {'Variable':<20} {'TRAIN':>10} {'TEST':>10} {'Gap':>10}")
    logger.info(f"  {'-' * 52}")
    common = [c for c in train_mapes if c in test_mapes]
    for col in common:
        tr = train_mapes[col]
        te = test_mapes[col]
        gap = te - tr
        flag = "  <<< overfit" if gap > OVERFIT_THRESHOLD_PCT else ""
        logger.info(f"  {col:<20} {tr:>9.2f}% {te:>9.2f}% {gap:>+9.2f}%{flag}")

    avg_tr = np.mean([train_mapes[c] for c in common])
    avg_te = np.mean([test_mapes[c] for c in common])
    logger.info(f"  {'-' * 52}")
    logger.info(f"  {'PROMEDIO':<20} {avg_tr:>9.2f}% {avg_te:>9.2f}% "
                f"{avg_te - avg_tr:>+9.2f}%")


def evaluate_checkpoint(checkpoint_path: Path, df: pd.DataFrame,
                        indices: dict, y_cols: list,
                        compare_train_test: bool,
                        subset: str) -> None:
    """Evalúa un checkpoint completo: precisión sobre el subconjunto y latencia.

    Avisa y regresa sin fallar si el checkpoint no existe, de modo que una lista de
    varios modelos pueda evaluarse aunque alguno todavía no se haya entrenado. La
    latencia se mide siempre, con independencia del modo de precisión elegido, porque
    es una propiedad del modelo y no del subconjunto sobre el que se evalúa.

    param checkpoint_path: Ruta del checkpoint a evaluar.
    param df: Corpus completo sobre el que se han calculado las particiones.
    param indices: Diccionario de índices por partición.
    param y_cols: Columnas de salida presentes en el corpus.
    param compare_train_test: Si es True se contrastan train y test en lugar de
        evaluar un único subconjunto.
    param subset: Partición a evaluar cuando no se pide la comparación.
    return: None; los resultados se emiten por el logger del módulo.
    """
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint no encontrado: {checkpoint_path}")
        return

    logger.info("")
    logger.info(f"Evaluando: {checkpoint_path}")

    model, n_params, hidden, n_layers = load_pinn(checkpoint_path)
    logger.info(f"  Arquitectura: hidden_dim={hidden}, n_layers={n_layers}")
    logger.info(f"  Parámetros:   {n_params:,}")

    X_all = torch.tensor(
        df[BraytonPINN.INPUT_FEATURES].values, dtype=torch.float32)
    Y_all = df[y_cols].values

    if compare_train_test:
        train_idx, test_idx = indices['train'], indices['test']
        train_mapes = compute_mape(model, X_all[train_idx],
                                    Y_all[train_idx], y_cols)
        test_mapes = compute_mape(model, X_all[test_idx],
                                   Y_all[test_idx], y_cols)
        logger.info(f"  Comparación train ({len(train_idx)}) vs "
                    f"test ({len(test_idx)}):")
        log_comparison_table(train_mapes, test_mapes)
    else:
        eval_idx = indices[subset]
        mapes = compute_mape(model, X_all[eval_idx],
                              Y_all[eval_idx], y_cols)
        log_mape_table(
            mapes, f"MAPE sobre {subset} ({len(eval_idx)} muestras)")

    latency = measure_latency(model, X_all[indices.get('test',
                                                        indices['full'])])
    logger.info(f"  Latencia inferencia ({LATENCY_ITERATIONS} evals):")
    logger.info(f"    media: {latency['mean']:.3f} ms  |  "
                f"P50: {latency['p50']:.3f} ms  |  "
                f"P95: {latency['p95']:.3f} ms  |  "
                f"P99: {latency['p99']:.3f} ms")


def main() -> None:
    """Punto de entrada por línea de comandos de la evaluación del gemelo digital.

    Carga el corpus una sola vez, calcula las particiones y recorre los checkpoints
    solicitados. Compartir corpus y particiones entre todos ellos es lo que hace
    comparables sus métricas: evaluar cada modelo con su propio reparto invalidaría
    cualquier conclusión sobre cuál es mejor.

    return: None; sale sin evaluar si el dataset no existe, y en caso contrario emite
        los resultados por el logger.
    """
    parser = argparse.ArgumentParser(
        description="Evaluación del PINN Brayton sobre el corpus")
    parser.add_argument('--checkpoints', type=Path, nargs='+',
                        default=[DEFAULT_CHECKPOINT],
                        help="Uno o más checkpoints a evaluar")
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET,
                        help=f"Corpus a evaluar (default: {DEFAULT_DATASET})")
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f"Semilla del split (default: {DEFAULT_SEED})")
    parser.add_argument('--subset', choices=['train', 'val', 'test', 'full'],
                        default='test',
                        help="Subconjunto a evaluar (default: test)")
    parser.add_argument('--compare-train-test', action='store_true',
                        help="Comparar train vs test y detectar overfitting")
    args = parser.parse_args()

    if not args.dataset.exists():
        logger.error(f"Dataset no encontrado: {args.dataset}")
        return

    df = pd.read_csv(args.dataset)
    y_cols = [c for c in BraytonPINN.OUTPUT_FEATURES if c in df.columns]
    indices = split_indices(len(df), DEFAULT_SPLITS, args.seed)

    logger.info(f"Corpus:  {args.dataset} ({len(df)} muestras)")
    logger.info(f"Splits:  train={len(indices['train'])} "
                f"val={len(indices['val'])} test={len(indices['test'])} "
                f"(seed={args.seed})")

    for ckpt_path in args.checkpoints:
        evaluate_checkpoint(ckpt_path, df, indices, y_cols,
                            args.compare_train_test, args.subset)


if __name__ == '__main__':
    main()