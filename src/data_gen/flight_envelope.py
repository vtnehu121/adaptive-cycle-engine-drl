"""
Última Fecha de Modificación: 08/Aug/2026
Descripción flight_envelope.py: Muestreador estratificado del envelope de vuelo
del motor ACE. Genera tuplas (altitude, mach, tra, bpr_ts, bleed_fraction)
repartidas entre las cinco fases operativas de una misión de caza multirrol
(takeoff, climb, cruise, combat, descent) con pesos editorialmente calibrados
para asegurar cobertura suficiente del envelope operativo durante el
entrenamiento del gemelo digital: mayor densidad en fases de alta variabilidad
(cruise, combat) y menor en fases transitorias cortas (takeoff, descent). Es el
primer eslabón del pipeline de datos: define el dominio sobre el que se generan
el corpus sintético y el gemelo digital.

Cada fase define un hiperrectángulo en el espacio de parámetros; dentro de él el
muestreo es uniforme e independiente en cada dimensión. Los rangos de
bleed_fraction reflejan la demanda auxiliar típica de la aeronave en cada fase:
máxima en cruise (presurización y aire acondicionado estables) y reducida en
combat, donde el motor prioriza empuje sobre servicios auxiliares.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FlightEnvelope:
    """Muestreador estratificado del envelope de vuelo por fase de misión.

    Encapsula los rangos operativos de cada fase y el reparto de muestras entre
    ellas. Es necesario porque un muestreo uniforme sobre todo el envelope
    sobrerrepresentaría regiones irrelevantes (p. ej. combate supersónico) y
    dejaría sin cubrir los puntos de operación donde el motor pasa realmente la
    mayor parte de la misión, sesgando el entrenamiento posterior.
    """

    # Rangos operativos y peso relativo de cada fase de misión.
    # Los pesos suman 1.0 y determinan el número de muestras por fase.
    # Rangos derivados del envelope operativo típico de un caza multirrol
    # de 4ª generación (F/A-18E/F Super Hornet, F-16 Fighting Falcon), con
    # empuje seco de la clase 20-25 klbf y afterburner de 30-35 klbf.
    REGIMES: Dict[str, Dict] = {
        'takeoff': {
            # Aceleración pista→despegue: pie de pista a ~2000 ft AGL,
            # baja velocidad de rotación (V2 ≈ 150-180 KIAS ≈ Mach 0.25).
            # TRA muy alto (90-100%) para maximizar empuje.
            'altitude': (0, 2000),
            'mach': (0.1, 0.35),
            'tra': (90, 100),
            'bpr_ts': (0.1, 0.3),
            'bleed_fraction': (0.02, 0.04),
            'weight': 0.15,
        },
        'climb': {
            # Ascenso subsonic hasta altitud de crucero. Mach 0.4-0.7
            # típico de perfil de ascenso continuo. TRA MCT (Max Continuous).
            'altitude': (2000, 25000),
            'mach': (0.4, 0.7),
            'tra': (80, 95),
            'bpr_ts': (0.2, 0.4),
            'bleed_fraction': (0.03, 0.06),
            'weight': 0.20,
        },
        'cruise': {
            # Crucero subsónico económico a alta altitud. Máximo bleed
            # (presurización + AA) y bypass alto para minimizar SFC.
            'altitude': (25000, 42000),
            'mach': (0.75, 0.90),
            'tra': (70, 85),
            'bpr_ts': (0.3, 0.5),
            'bleed_fraction': (0.04, 0.08),
            'weight': 0.25,
        },
        'combat': {
            # Maniobra táctica: Mach 0.8-1.8 (transonic→supersonic).
            # AB activo (TRA 95-100%), bypass mínimo (más empuje al núcleo),
            # bleed reducido (todo el aire disponible al motor).
            'altitude': (5000, 25000),
            'mach': (0.8, 1.8),
            'tra': (95, 100),
            'bpr_ts': (0.05, 0.2),
            'bleed_fraction': (0.01, 0.03),
            'weight': 0.25,
        },
        'descent': {
            # Descenso a idle o baja potencia. TRA muy reducido (30-60%),
            # bypass alto para eficiencia, bleed moderado.
            'altitude': (5000, 20000),
            'mach': (0.3, 0.6),
            'tra': (30, 60),
            'bpr_ts': (0.3, 0.5),
            'bleed_fraction': (0.03, 0.05),
            'weight': 0.15,
        },
    }

    def __init__(self, n_samples: int = 5000, seed: int = 42):
        """Inicializa el muestreador fijando tamaño de corpus y semilla.

        Crea el generador aleatorio a partir de una semilla explícita en lugar de
        usar el estado global de NumPy, que es lo que permite regenerar el corpus
        sintético del TFG bit a bit y que los resultados sean reproducibles.

        param n_samples: Número total de muestras a repartir entre las cinco fases.
        param seed: Semilla del generador aleatorio para muestreo determinista.
        return: None; deja el objeto listo para invocar generate().
        """
        self.n_samples = n_samples
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self) -> pd.DataFrame:
        """Genera el DataFrame completo de condiciones de vuelo del corpus.

        Recorre las cinco fases de REGIMES, muestrea uniformemente dentro del
        hiperrectángulo de cada una y concatena el resultado. Es necesario porque
        el resto del pipeline (simulador TMATS, gemelo digital y entorno de RL)
        parte de este conjunto de puntos de operación. El número de muestras por
        fase es ``n_samples * weight``; el truncamiento entero de ese reparto se
        compensa replicando filas al azar para devolver exactamente n_samples.

        return: DataFrame con columnas altitude, mach, tra, bpr_ts,
            bleed_fraction y regime, con exactamente n_samples filas.
        """
        samples = []
        for regime_name, params in self.REGIMES.items():
            n = int(self.n_samples * params['weight'])
            samples.append(pd.DataFrame({
                'altitude':       self.rng.uniform(*params['altitude'], n),
                'mach':           self.rng.uniform(*params['mach'], n),
                'tra':            self.rng.uniform(*params['tra'], n),
                'bpr_ts':         self.rng.uniform(*params['bpr_ts'], n),
                'bleed_fraction': self.rng.uniform(*params['bleed_fraction'], n),
                'regime':         regime_name,
            }))

        df = pd.concat(samples, ignore_index=True)

        # Compensar pérdida por truncamiento del reparto proporcional.
        # Genera el random_state para pandas.sample desde el propio rng
        # de NumPy, de modo que el reparto de filas extra queda encadenado
        # al mismo estado aleatorio que el muestreo uniforme y no depende
        # de una segunda semilla implícita.
        deficit = self.n_samples - len(df)
        if deficit > 0:
            sample_seed = int(self.rng.integers(0, 2**32 - 1))
            extra = df.sample(n=deficit, replace=True, random_state=sample_seed)
            df = pd.concat([df, extra], ignore_index=True)

        return df.iloc[:self.n_samples].reset_index(drop=True)

    def summary(self, df: pd.DataFrame) -> None:
        """Registra en el log un resumen estadístico del muestreo generado.

        Vuelca el reparto real de filas por fase y los descriptivos de cada
        variable continua. Es necesario para verificar que los pesos declarados
        en REGIMES se han materializado en el corpus y que los rangos no se han
        degradado tras la compensación por truncamiento, sin tener que abrir el
        CSV resultante.

        param df: DataFrame de condiciones de vuelo devuelto por generate().
        return: None; el resumen se emite por el logger del módulo.
        """
        logger.info(f"Muestreo: {len(df)} filas | seed = {self.seed}")

        counts = df['regime'].value_counts()
        for regime, n in counts.items():
            pct = 100.0 * n / len(df)
            logger.info(f"  {regime:<8} {n:>5} ({pct:5.1f}%)")

        cols = ['altitude', 'mach', 'tra', 'bpr_ts', 'bleed_fraction']
        stats = df[cols].describe().round(3)
        for line in stats.to_string().split('\n'):
            logger.info(f"  {line}")