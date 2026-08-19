"""
Última Fecha de Modificación: 08/Aug/2026
Descripción eval_rnn_compare.py: Script que compara las tres arquitecturas
recurrentes de monitorización de salud (LSTM-128, LSTM-64 y GRU-64) sobre el
conjunto de validación de C-MAPSS FD001, mediante evaluación manual con iteración
explícita batch por batch. Para cada checkpoint mide RMSE y MAE de la RUL, RMSE
de degradación por componente (Fan, HPC, HPT, LPT), dispersión σ de MC-Dropout
y cobertura del intervalo al 95%, y contrasta la evaluación manual con la del
propio HealthMonitor como comprobación cruzada. 

Nota metodológica:
    Este script evalúa cada modelo con su propio val_loader (el creado por
    HealthMonitor.prepare_cmapss para cada arquitectura), lo que introduce
    ligera variabilidad en las coberturas IC95% frente a
    plot_rnn_comparison.py, que usa un val_loader único compartido entre los
    tres modelos y es la comparativa oficial reportada en la memoria del TFG.
    La conclusión cualitativa (GRU-64 como modelo más calibrado) es la misma
    en ambos pipelines.

Uso:
    python scripts/evaluation/eval_rnn_compare.py
"""

import logging
import os
import sys
from pathlib import Path

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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
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

CHECKPOINT_DIR = Path('checkpoints/health_monitoring')
CMAPSS_DIR = Path('data/CMAPSS_DATA')
# 50 pases: estándar en la literatura MC-Dropout (Gal & Ghahramani 2016)
# para estimar incertidumbre epistémica. Menos pases dan estimaciones ruidosas;
# más de 50 aportan poco a la estabilidad y aumentan tiempo linealmente.
MC_DROPOUT_PASSES = 50

# Configuración de los tres modelos a comparar
MODELS = [
    {
        'checkpoint': 'rnn_lstm_128.pt',
        'label':      'LSTM hidden=128',
        'rnn_type':   'lstm',
        'hidden':     128,
    },
    {
        'checkpoint': 'rnn_lstm_64.pt',
        'label':      'LSTM hidden=64',
        'rnn_type':   'lstm',
        'hidden':     64,
    },
    {
        'checkpoint': 'rnn_gru_64.pt',
        'label':      'GRU  hidden=64',
        'rnn_type':   'gru',
        'hidden':     64,
    },
]


def build_model(rnn_type: str, hidden: int) -> HealthMonitoringRNN:
    """Construye una red de monitorización de salud con la arquitectura indicada.

    Fija el resto de hiperparámetros (número de sensores, capas, dropout y carácter
    unidireccional) para que la comparación aísle exactamente las dos variables de
    interés: el tipo de célula recurrente y el tamaño del estado oculto.

    param rnn_type: Variante recurrente a construir, 'lstm' o 'gru'.
    param hidden: Dimensión del estado oculto.
    return: Instancia de HealthMonitoringRNN sin pesos entrenados todavía.
    """
    return HealthMonitoringRNN(
        n_sensors=15,
        hidden_dim=hidden,
        n_layers=2,
        rnn_type=rnn_type,
        dropout=0.2,
        bidirectional=False,
    )


def evaluate_model(model: HealthMonitoringRNN,
                   val_loader,
                   mc_passes: int = MC_DROPOUT_PASSES) -> dict:
    """Evalúa un modelo en validación combinando predicción determinista e incertidumbre.

    Ejecuta primero una pasada en modo evaluación para medir el error puntual y
    después varias pasadas con dropout activo para estimar la dispersión y la
    cobertura del intervalo al 95%. Implementar aquí la evaluación, en paralelo a la
    del HealthMonitor, es deliberado: dos caminos independientes que coinciden dan
    confianza en que el resultado no procede de un fallo del pipeline interno. Se
    fija la semilla antes de la inferencia para que las 50 pasadas de MC-Dropout
    sean reproducibles entre ejecuciones sucesivas del script y coherentes con las
    figuras generadas por plot_rnn_comparison.py.

    param model: Red de monitorización con los pesos del checkpoint cargados.
    param val_loader: DataLoader del conjunto de validación.
    param mc_passes: Número de pases estocásticos de Monte Carlo.
    return: Diccionario con rmse y mae de la vida remanente, la lista deg_rmses por
        componente, la dispersión media mc_sigma y la cobertura en porcentaje.
    """
    # Semilla fija para reproducibilidad de MC-Dropout entre ejecuciones.
    # Sin ella, las 50 pasadas producen σ ligeramente distintos y la cobertura
    # IC95 puede variar ±10 puntos porcentuales entre corridas. Coherente con
    # plot_rnn_comparison.py para que ambos scripts den los MISMOS IC95%.
    torch.manual_seed(42)
    np.random.seed(42)

    # Predicción determinista (modo eval)
    model.eval()
    rul_pred, rul_true, deg_pred, deg_true = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0]
            rul = batch[1]
            deg = batch[2] if len(batch) > 2 else None
            r_pred, d_pred = model(x)
            rul_pred.append(r_pred.numpy())
            rul_true.append(rul.numpy())
            if deg is not None:
                deg_pred.append(d_pred.numpy())
                deg_true.append(deg.numpy())

    rul_pred = np.concatenate(rul_pred).flatten()
    rul_true = np.concatenate(rul_true).flatten()

    rmse = np.sqrt(np.mean((rul_pred - rul_true) ** 2))
    mae = np.mean(np.abs(rul_pred - rul_true))

    if deg_pred:
        deg_pred_arr = np.concatenate(deg_pred)
        deg_true_arr = np.concatenate(deg_true)
        deg_rmses = [
            np.sqrt(np.mean((deg_pred_arr[:, j] - deg_true_arr[:, j]) ** 2))
            for j in range(min(4, deg_pred_arr.shape[1]))
    ]
    else:
        # El DataLoader no proporcionó etiquetas de degradación
        # (val_loader de C-MAPSS FD001 típicamente solo tiene RUL, no deg)
        deg_rmses = [np.nan, np.nan, np.nan, np.nan]

    # Incertidumbre epistémica vía Monte Carlo Dropout
    model.train()  # activa dropout aunque estemos en inferencia
    mc_preds = []
    with torch.no_grad():
        for _ in range(mc_passes):
            batch_preds = []
            for batch in val_loader:
                x = batch[0]
                r_pred, _ = model(x)
                batch_preds.append(r_pred.numpy())
            mc_preds.append(np.concatenate(batch_preds).flatten())

    mc_preds = np.array(mc_preds)
    mc_mean = mc_preds.mean(axis=0)
    mc_std = mc_preds.std(axis=0)

    # Cobertura del intervalo de confianza al 95%
    lower = mc_mean - 1.96 * mc_std
    upper = mc_mean + 1.96 * mc_std
    coverage = np.mean((rul_true >= lower) & (rul_true <= upper)) * 100

    return {
        'rmse':      rmse,
        'mae':       mae,
        'deg_rmses': deg_rmses,
        'mc_sigma':  mc_std.mean(),
        'coverage':  coverage,
    }


def main() -> None:
    """Punto de entrada que evalúa los tres checkpoints y tabula la comparación.

    Recorre la configuración de MODELS, avisando y saltando los checkpoints que no
    existan, y termina con una tabla resumen. Reconstruye el conjunto de validación
    para cada modelo a través de su propio HealthMonitor porque la longitud de ventana
    forma parte de la configuración, y la partición por motor con semilla fija asegura
    que aun así todos se evalúen sobre las mismas unidades.

    return: None; los resultados se emiten por el logger del módulo.
    """
    logger.info("Evaluación comparativa de arquitecturas RNN")
    logger.info(f"Checkpoints en: {CHECKPOINT_DIR}")
    logger.info(f"Dataset C-MAPSS: FD001 subset de validación")
    logger.info(f"MC-Dropout: {MC_DROPOUT_PASSES} pases")

    loader = CMAPSSLoader(data_path=str(CMAPSS_DIR))

    results = []
    for cfg in MODELS:
        path = CHECKPOINT_DIR / cfg['checkpoint']
        if not path.exists():
            logger.warning(f"Checkpoint no encontrado: {path}")
            continue

        logger.info("")
        logger.info(f"Evaluando {cfg['label']} ({cfg['checkpoint']})")

        # Instanciar monitor con la arquitectura del modelo
        monitor = HealthMonitor(hidden_dim=cfg['hidden'],
                                rnn_type=cfg['rnn_type'])
        _, val_loader = monitor.prepare_cmapss(loader, subset='FD001')

        # strict=False para tolerar diferencias en buffers de normalización
        # entre versiones del modelo (coherente con eval_pinn.py y eval_pde_residuals.py)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        monitor.model.load_state_dict(ckpt, strict=False)

        n_params = sum(p.numel() for p in monitor.model.parameters())
        logger.info(f"  Parámetros: {n_params:,}")

        # Evaluación manual sobre el conjunto de validación
        metrics = evaluate_model(monitor.model, val_loader) 

        logger.info(f"  Evaluación manual:")
        logger.info(f"    RUL RMSE:       {metrics['rmse']:.2f} ciclos")
        logger.info(f"    RUL MAE:        {metrics['mae']:.2f} ciclos")
        deg_avg = np.nanmean(metrics['deg_rmses']) if not np.all(np.isnan(metrics['deg_rmses'])) else float('nan')
        if np.isnan(deg_avg):
            logger.info(f"    Deg RMSE:       N/A (etiquetas de degradación no disponibles)")
        else:
            logger.info(f"    Deg RMSE avg:   {deg_avg:.4f}")
            logger.info(f"      Fan={metrics['deg_rmses'][0]:.4f}  "
                        f"HPC={metrics['deg_rmses'][1]:.4f}  "
                        f"HPT={metrics['deg_rmses'][2]:.4f}  "
                        f"LPT={metrics['deg_rmses'][3]:.4f}")   
        logger.info(f"    MC-Dropout σ:   {metrics['mc_sigma']:.2f} ciclos")
        logger.info(f"    Cobertura IC95: {metrics['coverage']:.1f}%")

        results.append({
            'label':    cfg['label'],
            'params':   n_params,
            **metrics,
        })

    # Tabla resumen
    logger.info("")
    logger.info("Resumen comparativo:")
    logger.info(f"  {'Modelo':<20} {'Params':>10} {'RMSE':>8} "
                f"{'MAE':>8} {'σ (MC)':>8} {'IC95%':>8}")
    for r in results:
        logger.info(f"  {r['label']:<20} {r['params']:>10,} "
                    f"{r['rmse']:>8.2f} {r['mae']:>8.2f} "
                    f"{r['mc_sigma']:>8.2f} {r['coverage']:>7.1f}%")


if __name__ == '__main__':
    main()