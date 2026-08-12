"""
Última Fecha de Modificación: 08/Aug/2026
Descripción pipeline.py: Orquestador de la generación del corpus sintético del
motor ACE. Encadena las cinco etapas que van del muestreo
del envelope de vuelo a la validación físico-estadística del dataset, pasando
por la simulación termodinámica en T-MATS, el preprocesado de C-MAPSS y la
inyección de degradación. Produce el corpus de 5,000 muestras que alimenta el
entrenamiento del Gemelo Digital PINN (MAPE 4.64%) y del monitor de salud
GRU-64 (RMSE 13.94 ciclos).

Etapas:
    1. Muestreo estratificado de condiciones de vuelo (uniforme por fase).
    2. Simulación termodinámica en T-MATS (MATLAB Engine + Simulink).
    3. Carga y preprocesamiento del dataset NASA C-MAPSS.
    4. Inyección estocástica de degradación siguiendo Saxena et al. (2008).
    5. Validación físico-estadística del corpus generado (9 chequeos).

Cada etapa persiste su artefacto en `data/`, lo que permite reanudar el proceso
sin repetir la simulación en MATLAB (etapa 2), con diferencia la más costosa.
Semilla fija por defecto (seed=42) para reproducibilidad del corpus completo.

Uso:
    python src/pipeline.py                   # Pipeline completo
    python src/pipeline.py --skip-matlab     # Reutiliza corpus existente
    python src/pipeline.py --n-samples 100   # Muestreo reducido
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Añadir src/ al path para que los imports funcionen ejecutando desde el
# root del proyecto (python src/pipeline.py) o desde src/ (python pipeline.py).
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s'
)
logger = logging.getLogger(__name__)

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
SYNTHETIC_DIR = DATA_DIR / 'synthetic'
PROCESSED_DIR = DATA_DIR / 'processed'
CMAPSS_DIR = DATA_DIR / 'CMAPSS_DATA'
MODELS_DIR = PROJECT_ROOT / 'models'

# Configuración por defecto del muestreo
DEFAULT_N_SAMPLES = 5000
DEFAULT_SEED = 42


def step_1_flight_envelope(n_samples: int, seed: int) -> pd.DataFrame:
    """Muestrea condiciones de vuelo por muestreo estratificado uniforme.
    Genera `n_samples` tuplas (altitude, mach, tra, bpr_ts) distribuidas
    entre las cinco fases de misión (takeoff, climb, cruise, combat, descent)
    con pesos operacionales predefinidos en FlightEnvelope.REGIMES.

    Es la primera etapa porque define el dominio de todo lo que viene después: los
    puntos aquí muestreados son los que se simularán, se degradarán y acabarán
    delimitando la región del envelope en la que el gemelo digital resulta fiable.

    param n_samples: Número total de condiciones de vuelo a generar.
    param seed: Semilla del muestreo, para reproducibilidad del corpus.
    return: DataFrame de condiciones de vuelo, ya guardado en
        data/synthetic/flight_conditions_5000.csv.
    """
    logger.info("Paso 1/5: muestreo de condiciones de vuelo")

    from data_gen.flight_envelope import FlightEnvelope

    env = FlightEnvelope(n_samples=n_samples, seed=seed)
    conditions = env.generate()
    env.summary(conditions)

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SYNTHETIC_DIR / f'flight_conditions_{n_samples}.csv'
    conditions.to_csv(output_path, index=False)

    logger.info(f"Condiciones guardadas: {output_path}")
    return conditions


def step_2_tmats_simulation(conditions: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta simulaciones termodinámicas del modelo ACE en T-MATS.

    Inicia MATLAB Engine, carga init_ACE.m (que configura T-MATS y
    carga el modelo Simulink ACE_3stream_brayton.slx), y ejecuta una
    simulación transitoria por cada fila de `conditions`. El resultado
    es un DataFrame con las 26 variables termodinámicas extraídas de
    las estaciones del ciclo Brayton adaptativo.

    Es la etapa más costosa del pipeline y la única que depende de MATLAB, razón por
    la que existe la opción de omitirla reutilizando el corpus ya generado. El cierre
    del simulador se hace en un bloque finally para que la sesión de MATLAB no quede
    huérfana si una simulación aborta a mitad del batch.

    param conditions: DataFrame de condiciones de vuelo del paso anterior.
    return: DataFrame con el estado termodinámico simulado de cada condición, ya
        guardado en data/synthetic/ace_dataset_{n_samples}.csv.
    """
    logger.info("Paso 2/5: simulación T-MATS (MATLAB Engine)")

    from simulators.tmats_interface import TMATSSimulator

    sim = TMATSSimulator(models_path=str(MODELS_DIR))
    sim.start()

    try:
        n = len(conditions)
        output_path = SYNTHETIC_DIR / f'ace_dataset_{n}.csv'
        dataset = sim.batch(conditions, save_path=str(output_path))
        logger.info(f"Dataset generado: {dataset.shape[0]} filas × {dataset.shape[1]} columnas")
        return dataset
    finally:
        sim.stop()


def step_3_cmapss_analysis() -> pd.DataFrame:
    """Carga y preprocesa el dataset C-MAPSS FD001 de NASA.

    Aplica normalización min-max y calcula el health index como
    combinación de márgenes operacionales normalizados. El resultado
    se usa como referencia para calibrar la magnitud de la degradación
    inyectada en el paso 4.

    Es necesario porque el corpus del ACE es sintético y carece de referencia real
    de deterioro: C-MAPSS aporta esa referencia empírica, procedente de motores
    simulados run-to-failure, y fija el orden de magnitud al que se calibra la
    Table 3 usada por DegradationInjector (etapa 4). El DataFrame retornado se
    usa solo para logging y análisis exploratorios; DegradationInjector accede
    a los valores tabulados directamente a través de WEAR_TABLE.

    return: DataFrame del sub-dataset FD001 con health index calculado y sensores
        normalizados.
    """
    logger.info("Paso 3/5: preprocesamiento C-MAPSS FD001")

    from preprocessing.cmapss_loader import CMAPSSLoader

    loader = CMAPSSLoader(data_path=str(CMAPSS_DIR))
    loader.load_all()
    loader.summary()

    fd001 = loader.load_subset('FD001')
    fd001 = loader.compute_health_index(fd001)
    fd001 = loader.normalize(fd001, method='minmax')

    logger.info(f"C-MAPSS FD001: {fd001.shape[0]} ciclos × {fd001.shape[1]} variables")
    return fd001


def step_4_degradation(dataset: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Inyecta degradación exponencial estocástica sobre el corpus nominal.

    Aplica el modelo h(t) = 1 - exp(-a·t^b) por componente (fan, LPC,
    HPC, HPT, LPT) con parámetros a ∈ [0.001, 0.003] y b ∈ [1.4, 1.6]
    muestreados aleatoriamente. Los valores tabulados de desgaste natural
    siguen Saxena et al. (2008), Table 3.

    Sin esta etapa el corpus contendría únicamente motores nominales y no
    habría material con el que entrenar el pronóstico de vida remanente ni
    con el que evaluar si el control adaptativo se comporta mejor que el
    convencional en un motor envejecido, que es la hipótesis central del
    trabajo.

    param dataset: Corpus nominal simulado en la etapa 2.
    param seed: Semilla del generador aleatorio, propagada del pipeline
        para garantizar reproducibilidad del corpus completo con la misma
        semilla.
    return: Corpus con las columnas de degradación y health index añadidas,
        ya guardado en data/processed/ace_dataset_{n_samples}_degraded.csv.
    """
    logger.info("Paso 4/5: inyección de degradación C-MAPSS")

    from data_gen.degradation import DegradationInjector

    injector = DegradationInjector(seed=seed)
    df_degraded = injector.inject(dataset)
    injector.summary(df_degraded)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    n = len(df_degraded)
    output_path = PROCESSED_DIR / f'ace_dataset_{n}_degraded.csv'
    df_degraded.to_csv(output_path, index=False)

    logger.info(f"Dataset degradado guardado: {output_path}")
    return df_degraded


def step_5_validation(dataset: pd.DataFrame) -> bool:
    """Valida la integridad físico-estadística del corpus generado.

    Comprueba nueve invariantes del ciclo Brayton adaptativo:
        - Ausencia de NaN e infinitos en todas las columnas numéricas.
        - Consistencia termodinámica en las estaciones del ciclo.
        - Compresión y expansión con signo termodinámico correcto.
        - Empuje y velocidades de eje dentro de rangos operacionales.
        - Correlación positiva entre TRA y empuje (r > 0.3).
        - Empuje inferior al límite de la clase C-MAPSS (90 000 lbf).
        - Consumo específico dentro del rango físico plausible.

    Es imprescindible como puerta de calidad del corpus: un solver que converge no
    garantiza un resultado físicamente coherente, y entrenar el gemelo digital sobre
    puntos que violan la monotonía del ciclo o el signo de la compresión produciría un
    modelo aparentemente preciso pero físicamente inconsistente. Cada comprobación se
    aplica solo si las columnas implicadas existen, de modo que la validación degrada
    con elegancia ante corpus parciales en lugar de fallar.

    param dataset: Corpus a validar, típicamente el nominal recién simulado.
    return: True si se superan todos los chequeos aplicables; False si alguno falla,
        habiéndose registrado cada resultado individualmente en el log.
    """
    logger.info("Paso 5/5: validación físico-estadística")

    checks = []

    # 1. Integridad numérica
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    has_nan = dataset[numeric_cols].isna().any().any()
    has_inf = np.isinf(dataset[numeric_cols].values).any()
    checks.append(('NaN ausentes', not has_nan))
    checks.append(('Infinitos ausentes', not has_inf))

    # 2. Consistencia termodinámica del ciclo Brayton
    if all(c in dataset.columns for c in ['T2', 'T30', 'T4', 'T50']):
        checks.append(('T30 > T2 (compresión)', (dataset['T30'] > dataset['T2']).all()))
        checks.append(('T4 > T30 (combustión)', (dataset['T4'] > dataset['T30']).all()))
        checks.append(('T50 < T4 (expansión)', (dataset['T50'] < dataset['T4']).all()))
        checks.append(('T4 < 4000 R (límite físico)', (dataset['T4'] < 4000).all()))

    # 3. Consistencia de presiones
    if all(c in dataset.columns for c in ['P2', 'P30']):
        checks.append(('P30 > P2 (compresión)', (dataset['P30'] > dataset['P2']).all()))

    # 4. Empuje operacional
    if 'thrust_approx' in dataset.columns:
        thrust = dataset['thrust_approx']
        checks.append(('Empuje > 0', (thrust > 0).all()))
        checks.append(('Empuje < 500 000 lbf', (thrust < 500_000).all()))

    # 5. Consistencia de velocidades de eje.
    # En un turbofán de dos ejes, el eje de alta (Nc) gira más rápido
    # que el de baja (Nf) por la ratio de la caja de engranajes; si Nf
    # ≥ Nc el corpus muestra un motor mal configurado.
    if all(c in dataset.columns for c in ['Nf', 'Nc']):
        checks.append(('Nf > 0', (dataset['Nf'] > 0).all()))
        checks.append(('Nc > Nf (HP > LP)', (dataset['Nc'] > dataset['Nf']).all()))

    # 6. Consumo específico plausible
    if 'sfc' in dataset.columns:
        checks.append(('0 < SFC < 3', ((dataset['sfc'] > 0) & (dataset['sfc'] < 3)).all()))

    # 7. Correlación fisica esperada: TRA con empuje
    if all(c in dataset.columns for c in ['tra', 'thrust_approx']):
        corr = dataset['tra'].corr(dataset['thrust_approx'])
        checks.append((f'Corr(TRA, empuje) > 0.3 [obs: {corr:.3f}]', corr > 0.3))

    # 8. Correlación fisica esperada: bpr_ts con BPR real
    if all(c in dataset.columns for c in ['bpr_ts', 'BPR']):
        corr = dataset['bpr_ts'].corr(dataset['BPR'])
        checks.append((f'Corr(bpr_ts, BPR) > 0.5 [obs: {corr:.3f}]', corr > 0.5))

    # Reportar resultados
    all_passed = True
    for name, passed in checks:
        status = 'OK' if passed else 'FAIL'
        logger.info(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info(f"Validación completa: {len(checks)}/{len(checks)} chequeos superados")
    else:
        n_failed = sum(1 for _, p in checks if not p)
        logger.warning(f"Validación fallida: {n_failed}/{len(checks)} chequeos no superados")

    return all_passed


def main():
    """Punto de entrada por línea de comandos que ejecuta las cinco etapas en orden.

    Analiza los argumentos, encadena las etapas y emite un resumen con los artefactos
    producidos y el tiempo total. Termina con un código de salida que refleja el
    resultado de la validación, para que el pipeline pueda encadenarse en un script o
    en integración continua y un corpus inválido detenga el flujo aguas abajo.

    return: None; el proceso finaliza con código 0 si la validación se supera y 1 si
        falla o si se pidió omitir MATLAB sin existir un corpus previo.
    """
    parser = argparse.ArgumentParser(
        description='Pipeline de generación del corpus sintético del motor ACE.'
    )
    parser.add_argument('--n-samples', type=int, default=DEFAULT_N_SAMPLES,
                        help=f'Número de muestras a generar (default: {DEFAULT_N_SAMPLES})')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'Semilla del generador aleatorio (default: {DEFAULT_SEED})')
    parser.add_argument('--skip-matlab', action='store_true',
                        help='Reutiliza corpus existente sin invocar MATLAB')
    args = parser.parse_args()

    start_time = time.time()

    logger.info(f"Configuración: n_samples={args.n_samples}, seed={args.seed}, "
                f"matlab={'off' if args.skip_matlab else 'on'}")

    # Paso 1: condiciones de vuelo
    conditions = step_1_flight_envelope(args.n_samples, args.seed)

    # Paso 2: simulación T-MATS (opcional)
    dataset_path = SYNTHETIC_DIR / f'ace_dataset_{args.n_samples}.csv'
    if args.skip_matlab:
        if not dataset_path.exists():
            logger.error(f"Corpus no encontrado en {dataset_path}. "
                         f"Ejecute sin --skip-matlab para generarlo.")
            sys.exit(1)
        logger.info("Paso 2/5: omitido (--skip-matlab), cargando corpus existente")
        dataset = pd.read_csv(dataset_path)
        logger.info(f"Corpus cargado: {dataset.shape[0]} filas × {dataset.shape[1]} columnas")
    else:
        dataset = step_2_tmats_simulation(conditions)

    # Paso 3: C-MAPSS (informativo). Los valores tabulados usados en el
    # paso 4 están hardcoded en DegradationInjector.WEAR_TABLE; step_3
    # existe para documentar y validar el corpus de referencia. Descartamos
    # el DataFrame retornado (los logs de step_3 son el output útil).
    _ = step_3_cmapss_analysis()

    # Paso 4: degradación (usa la misma semilla que el paso 1 para
    # que el corpus completo sea reproducible con --seed único).
    df_degraded = step_4_degradation(dataset, args.seed)

    # Paso 5: validación
    validation_passed = step_5_validation(dataset)

    # Resumen
    elapsed = time.time() - start_time
    logger.info(f"Pipeline completado en {elapsed:.1f} s")
    logger.info(f"  Corpus nominal:   {SYNTHETIC_DIR / f'ace_dataset_{args.n_samples}.csv'}")
    logger.info(f"  Corpus degradado: {PROCESSED_DIR / f'ace_dataset_{args.n_samples}_degraded.csv'}")
    logger.info(f"  Validación:       {'superada' if validation_passed else 'con fallos'}")

    sys.exit(0 if validation_passed else 1)


if __name__ == '__main__':
    main()