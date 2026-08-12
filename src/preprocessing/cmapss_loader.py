"""
Última Fecha de Modificación: 08/Aug/2026
Descripción cmapss_loader.py: Carga y preprocesamiento del dataset NASA C-MAPSS.
Provee la clase CMAPSSLoader, que unifica el acceso a los
cuatro sub-datasets (FD001–FD004) de simulaciones run-to-failure de turbofán
generadas con el simulador Commercial Modular Aero-Propulsion System Simulation
y añade el cálculo de RUL y health index, la normalización de sensores y la
generación de features dinámicas por ventana deslizante.

Los cuatro sub-datasets se distinguen por el número de condiciones
operacionales y modos de fallo simulados:

    FD001 → 1 condición,  1 fallo  (HPC)         100 motores
    FD002 → 6 condiciones, 1 fallo  (HPC)         260 motores
    FD003 → 1 condición,  2 fallos (HPC + Fan)   100 motores
    FD004 → 6 condiciones, 2 fallos              249 motores

Referencia:
    Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). "Damage
    propagation modeling for aircraft engine run-to-failure simulation."
    International Conference on Prognostics and Health Management (PHM),
    IEEE. DOI: 10.1109/PHM.2008.4711414
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class CMAPSSLoader:
    """Carga y preprocesa los sub-datasets NASA C-MAPSS.

    Ofrece utilidades para calcular el health index a partir del RUL,
    normalizar sensores con MinMax o Standard scaling, generar features
    dinámicas por ventana deslizante y filtrar sensores constantes.

    Es necesaria porque los archivos originales de C-MAPSS son texto plano sin
    cabecera y sin etiqueta de RUL por ciclo: centralizar aquí el esquema de
    columnas, el cálculo de la vida remanente y la lista de sensores informativos
    evita que cada script de análisis reimplemente ese preprocesado y se desvíe.
    """

    # Nombres de las 26 columnas del formato C-MAPSS
    COLUMN_NAMES = (
        ['unit_id', 'cycle'] +
        ['op_setting_1', 'op_setting_2', 'op_setting_3'] +
        ['T2', 'T24', 'T30', 'T50',
         'P2', 'P15', 'P30',
         'Nf', 'Nc',
         'epr', 'Ps30', 'phi',
         'NRf', 'NRc',
         'BPR', 'farB', 'htBleed',
         'Nf_dmd', 'PCNfR_dmd',
         'W31', 'W32']
    )

    # Sensores con varianza significativa (15 informativos = 21 originales - 6 constantes)
    INFORMATIVE_SENSORS = [
        'T24', 'T30', 'T50', 'P30',
        'Nf', 'Nc', 'epr', 'Ps30', 'phi',
        'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32'
    ]

    # Sensores con varianza cuasi-nula (candidatos a eliminación)
    CONSTANT_SENSORS = ['T2', 'P2', 'P15', 'farB', 'Nf_dmd', 'PCNfR_dmd']

    # Correspondencia conceptual entre componentes C-MAPSS y actuadores ACE.
    # Usado por notebooks de análisis para validar la transferencia de
    # patrones de degradación desde C-MAPSS al motor ACE de tres flujos.
    CMAPSS_TO_ACE_MAP = {
        'fan_eff': {
            'cmapss_sensors': ['Nf', 'NRf', 'BPR'],
            'ace_actuator': 'VFGV',
            'description': 'Degradación fan → reparto tercer flujo',
        },
        'lpc_eff': {
            'cmapss_sensors': ['T24', 'P15'],
            'ace_actuator': None,
            'description': 'Degradación LPC → presión intermedia',
        },
        'hpc_eff': {
            'cmapss_sensors': ['T30', 'P30', 'Ps30', 'phi'],
            'ace_actuator': 'VGV-T',
            'description': 'Degradación HPC → geometría VGV',
        },
        'hpt_eff': {
            'cmapss_sensors': ['T50', 'epr', 'htBleed'],
            'ace_actuator': 'Sangrado',
            'description': 'Degradación HPT → refrigeración',
        },
        'lpt_eff': {
            'cmapss_sensors': ['Nc', 'NRc', 'W31', 'W32'],
            'ace_actuator': None,
            'description': 'Degradación LPT → potencia eje',
        },
    }

    SUBSETS = {
        'FD001': {'op_conditions': 1, 'fault_modes': 1, 'description': '1 cond, 1 fallo (HPC)'},
        'FD002': {'op_conditions': 6, 'fault_modes': 1, 'description': '6 cond, 1 fallo (HPC)'},
        'FD003': {'op_conditions': 1, 'fault_modes': 2, 'description': '1 cond, 2 fallos (HPC+Fan)'},
        'FD004': {'op_conditions': 6, 'fault_modes': 2, 'description': '6 cond, 2 fallos'},
    }

    def __init__(self, data_path: str):
        """Inicializa el loader y valida que la carpeta de datos exista.

        Falla de inmediato si la ruta no existe, en lugar de dejar que el error
        aparezca más tarde al leer un archivo concreto, y prepara las cachés de
        sub-datasets, tablas de RUL y escaladores para que cargas y
        normalizaciones repetidas no se recalculen.

        param data_path: Carpeta con los archivos train_FDxxx.txt, test_FDxxx.txt
            y RUL_FDxxx.txt del dataset C-MAPSS.
        return: None; deja el loader listo, o lanza FileNotFoundError si la ruta
            no existe.
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Carpeta C-MAPSS no encontrada: {self.data_path}")
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.rul_data: Dict[str, pd.DataFrame] = {}
        self.scalers: Dict[str, object] = {}
        logger.info(f"CMAPSSLoader inicializado en {self.data_path}")

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Carga en memoria los cuatro sub-datasets FD001–FD004.

        Es necesario para los análisis comparativos del TFG, que contrastan el
        comportamiento de la degradación entre número de condiciones operativas y
        de modos de fallo: sin cargarlos todos no puede evaluarse si el patrón
        observado en FD001 se generaliza al resto.

        return: Diccionario {subset: DataFrame} con los cuatro sub-datasets ya
            etiquetados con RUL por ciclo.
        """
        for subset in ['FD001', 'FD002', 'FD003', 'FD004']:
            self.datasets[subset] = self._load_subset(subset)
        logger.info("Cuatro sub-datasets C-MAPSS cargados")
        return self.datasets

    def load_subset(self, subset: str = 'FD001') -> pd.DataFrame:
        """Carga un único sub-dataset, reutilizando la caché si ya se leyó.

        Es la vía habitual cuando solo interesa un escenario concreto (FD001 en la
        mayor parte del TFG). La caché evita releer y reprocesar del disco un
        archivo de decenas de miles de filas cada vez que un notebook lo solicita.

        param subset: Identificador del sub-dataset ('FD001'…'FD004').
        return: DataFrame del sub-dataset con las 26 columnas nombradas, la
            columna RUL por ciclo y la columna subset.
        """
        if subset not in self.datasets:
            self.datasets[subset] = self._load_subset(subset)
        return self.datasets[subset]

    def _load_subset(self, subset: str) -> pd.DataFrame:
        """Lee de disco los archivos de un sub-dataset y calcula el RUL por ciclo.

        Asigna nombres a las 26 columnas del formato C-MAPSS, que llega sin
        cabecera, y deriva la vida remanente restando el ciclo actual al último
        ciclo de cada motor. Ese etiquetado es imprescindible porque el archivo
        RUL_FDxxx.txt solo aporta la vida remanente del último ciclo del conjunto
        de test, no la de cada fila de entrenamiento.

        param subset: Identificador del sub-dataset ('FD001'…'FD004').
        return: DataFrame con las columnas nombradas, la RUL calculada por ciclo y
            una columna subset con el identificador de origen.
        """
        train_file = self.data_path / f'train_{subset}.txt'
        df = pd.read_csv(train_file, sep=r'\s+', header=None)
        # C-MAPSS tiene 26 columnas separadas por espacios; el iloc[:, :26]
        # es salvaguarda contra columnas extra vacías al final de cada línea,
        # habitual cuando el archivo termina con espacios en blanco.
        if df.shape[1] < 26:
            raise ValueError(
                f"Archivo {train_file.name} tiene {df.shape[1]} columnas, "
                f"esperadas 26. Posible corrupción del archivo.")
        df = df.iloc[:, :26]
        df.columns = self.COLUMN_NAMES

        rul_file = self.data_path / f'RUL_{subset}.txt'
        self.rul_data[subset] = pd.read_csv(rul_file, sep=r'\s+', header=None)

        # RUL por ciclo = ciclos restantes hasta fallo
        max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
        max_cycles.columns = ['unit_id', 'max_cycle']
        df = df.merge(max_cycles, on='unit_id')
        df['RUL'] = df['max_cycle'] - df['cycle']
        df.drop('max_cycle', axis=1, inplace=True)
        df['subset'] = subset

        n_units = df['unit_id'].nunique()
        logger.info(
            f"  {subset}: {n_units} motores, {len(df)} filas, "
            f"RUL medio = {df['RUL'].mean():.0f} ciclos"
        )
        return df

    def compute_health_index(self, df: pd.DataFrame,
                             max_rul_clip: int = 125) -> pd.DataFrame:
        """Calcula el health index HI ∈ [0, 1] a partir del RUL.

        Aplica clipping en `max_rul_clip` ciclos siguiendo la práctica
        estándar en la literatura (Heimes 2008, Zheng et al. 2017), que
        satura HI = 1 durante la fase de operación normal y transiciona
        linealmente a HI = 0 conforme el motor se aproxima al fallo.

        El clipping es necesario porque al principio de la vida del motor los
        sensores apenas cambian: pedirle al modelo que distinga una RUL de 300 de
        una de 250 ciclos introduce ruido de etiqueta y penaliza la precisión justo
        en la zona crítica, que es la cercana al fallo.

        param df: DataFrame con la columna RUL calculada por ciclo.
        param max_rul_clip: Umbral de saturación de la RUL, en ciclos.
        return: Copia del DataFrame con las columnas RUL_clipped y HI añadidas.
        """
        df = df.copy()
        df['RUL_clipped'] = df['RUL'].clip(upper=max_rul_clip)
        df['HI'] = df['RUL_clipped'] / max_rul_clip
        return df

    def normalize(self, df: pd.DataFrame, method: str = 'minmax',
                  sensors: Optional[list] = None) -> pd.DataFrame:
        """Normaliza los sensores informativos con MinMax o StandardScaler.

        MinMax mapea cada sensor al rango [0, 1] y es preferible cuando
        la distribución no es gaussiana; StandardScaler produce media 0
        y varianza 1 y es preferible con arquitecturas que asumen
        entradas centradas (redes densas con activaciones tanh).

        Es necesario porque los sensores de C-MAPSS conviven en escalas de órdenes
        de magnitud muy distintos (presiones de decenas frente a revoluciones de
        miles): sin normalizar, el descenso de gradiente queda dominado por las
        variables de mayor rango. El escalador ajustado se guarda en self.scalers
        y puede reutilizarse externamente (scaler.transform(new_data)) para
        aplicar la misma transformación a datos nuevos (típico caso train/test).
        Llamar a normalize() de nuevo con el mismo method sobrescribe el scaler
        anterior en self.scalers.

        param df: DataFrame con las columnas de sensores a normalizar.
        param method: 'minmax' para MinMaxScaler o cualquier otro valor para
            StandardScaler.
        param sensors: Lista explícita de columnas a normalizar; si es None se
            usan los sensores informativos presentes en el DataFrame.
        return: Copia del DataFrame con las columnas de sensores reescaladas.
        """
        df = df.copy()
        if sensors is None:
            sensors = [s for s in self.INFORMATIVE_SENSORS if s in df.columns]
        scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
        df[sensors] = scaler.fit_transform(df[sensors])
        self.scalers[method] = scaler
        logger.info(f"Normalización '{method}' aplicada a {len(sensors)} sensores")
        return df

    def feature_engineering(self, df: pd.DataFrame,
                            window_sizes: Optional[list] = None) -> pd.DataFrame:
        """Genera features dinámicas por sensor.

        Para cada sensor informativo produce:
            - Delta respecto al valor inicial del motor.
            - Tasa de cambio ciclo a ciclo.
            - Media móvil y desviación estándar sobre las ventanas
              indicadas en `window_sizes`.

        Estas features capturan la tendencia degradativa que los sensores
        crudos no revelan por sí solos, mejorando la señal disponible
        para modelos de RUL basados en LSTM o GRU. Es necesario porque el valor
        instantáneo de un sensor depende sobre todo del punto de operación; lo que
        informa del deterioro es su deriva respecto al motor sano y la variabilidad
        acumulada en las últimas ventanas.

        param df: DataFrame con los sensores y la columna unit_id, que agrupa por
            motor para que las ventanas no crucen la frontera entre motores.
        param window_sizes: Tamaños de ventana, en ciclos, de las medias y
            desviaciones móviles.
        return: Copia del DataFrame con las columnas {sensor}_delta,
            {sensor}_rate, {sensor}_rmean_{w} y {sensor}_rstd_{w} añadidas.
        """
        # Ventanas por defecto: [5, 10, 20] ciclos, cubren tendencias a
        # corto (ruido de sensores), medio (dinámica de operación) y
        # largo (degradación progresiva).
        if window_sizes is None:
            window_sizes = [5, 10, 20]

        # Ordenar por (unit_id, cycle) garantiza que las tasas de cambio
        # (df.diff) y ventanas móviles se calculan sobre secuencias
        # temporalmente coherentes, aunque el DataFrame llegue desordenado.
        df = df.copy().sort_values(['unit_id', 'cycle']).reset_index(drop=True)
        sensors = [s for s in self.INFORMATIVE_SENSORS if s in df.columns]

        for sensor in sensors:
            first_values = df.groupby('unit_id')[sensor].transform('first')
            df[f'{sensor}_delta'] = df[sensor] - first_values
            df[f'{sensor}_rate'] = df.groupby('unit_id')[sensor].diff().fillna(0)

            for w in window_sizes:
                rolling = df.groupby('unit_id')[sensor].rolling(
                    window=w, min_periods=1)
                df[f'{sensor}_rmean_{w}'] = rolling.mean().reset_index(
                    level=0, drop=True)
                df[f'{sensor}_rstd_{w}'] = rolling.std().fillna(0).reset_index(
                    level=0, drop=True)

        feature_suffixes = ('_delta', '_rate', '_rmean_', '_rstd_')
        n_new = sum(1 for c in df.columns if any(s in c for s in feature_suffixes))
        logger.info(f"Feature engineering: {n_new} features generadas")
        return df

    def remove_constant_sensors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina del DataFrame los sensores con varianza cuasi-nula.

        Los sensores de CONSTANT_SENSORS mantienen prácticamente el mismo valor
        durante toda la vida del motor, de modo que no aportan información sobre la
        degradación pero sí dimensionalidad y coste de entrenamiento. Retirarlos
        antes de normalizar evita además dividir por rangos casi nulos.

        param df: DataFrame del que retirar los sensores constantes.
        return: Copia del DataFrame sin las columnas de sensores constantes; las
            que no estuvieran presentes se ignoran sin error.
        """
        df = df.copy()
        to_drop = [s for s in self.CONSTANT_SENSORS if s in df.columns]
        df.drop(columns=to_drop, inplace=True, errors='ignore')
        logger.info(f"Eliminados {len(to_drop)} sensores constantes")
        return df

    def summary(self) -> None:
        """Registra en el log un resumen tabular de los sub-datasets cargados.

        Tabula número de motores, de filas y estadísticos de la vida útil de cada
        sub-dataset. Es necesario para documentar en la memoria las cifras exactas
        del corpus usado y para detectar de un vistazo una carga incompleta o una
        ruta de datos equivocada.

        return: None; el resumen se emite por el logger, y si no hay sub-datasets
            cargados se registra un aviso y se sale sin más.
        """
        if not self.datasets:
            logger.warning("No hay sub-datasets cargados. Ejecute load_all() primero.")
            return

        logger.info("Resumen NASA C-MAPSS (Saxena et al., PHM 2008)")
        logger.info(f"  {'Subset':<8} {'Motores':>8} {'Filas':>8} "
                    f"{'RUL medio':>10} {'RUL σ':>8}  Descripción")

        total_units = 0
        total_rows = 0
        for subset, info in self.SUBSETS.items():
            if subset not in self.datasets:
                continue
            df = self.datasets[subset]
            n_units = df['unit_id'].nunique()
            n_rows = len(df)
            rul_mean = df.groupby('unit_id')['RUL'].max().mean()
            rul_std = df.groupby('unit_id')['RUL'].max().std()
            total_units += n_units
            total_rows += n_rows
            logger.info(f"  {subset:<8} {n_units:>8} {n_rows:>8} "
                        f"{rul_mean:>10.0f} {rul_std:>8.0f}  {info['description']}")

        logger.info(f"  {'TOTAL':<8} {total_units:>8} {total_rows:>8}")