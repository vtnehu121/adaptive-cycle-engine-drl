"""
Última Fecha de Modificación: 08/Aug/2026
Descripción degradation.py: Inyección estocástica de degradación sobre el corpus
sintético ACE. Adapta el protocolo NASA C-MAPSS (Saxena,
Goebel, Simon; PHM 2008) a la arquitectura de tres flujos del motor, combinando
dos modelos de desgaste según el rol de cada componente en el paper original:

    1. Desgaste natural de fondo (Table 3 del paper): valores deterministas
       calibrados para Fan, LPC, HPT y LPT. Representa el envejecimiento
       gradual común a todos los motores del dataset.

    2. Degradación exponencial estocástica adaptada de la ec. 6 del paper:
       modelo h(t) = 1 - exp(-a·t^b) con parámetros a ∈ [0.001, 0.003] y
       b ∈ [1.4, 1.6] muestreados por motor, aplicando t normalizado en
       [0,1] en lugar de ciclos absolutos (ver _compute_hpc_exponential
       para la justificación de la adaptación). Reservado para el HPC
       como fault mode principal, consistente con la sección VI-1 del
       paper.

Añade al corpus los deltas de eficiencia y flujo por componente, las variables
termodinámicas degradadas y un health index normalizado. Es la fuente de
etiquetas para el monitor de salud (RNN) y de los estados degradados que ve el
agente de control.

Referencia:
    Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). "Damage
    propagation modeling for aircraft engine run-to-failure simulation."
    International Conference on Prognostics and Health Management (PHM),
    IEEE. DOI: 10.1109/PHM.2008.4711414
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DegradationInjector:
    """Inyecta degradación estocástica sobre el corpus sintético ACE.

    Reúne en una sola clase las tablas de desgaste de C-MAPSS, el modelo
    exponencial del HPC y el mapeo hacia las variables del ACE. Es necesaria
    porque el simulador termodinámico produce únicamente motores nominales: sin
    esta capa no existirían trayectorias de deterioro con las que entrenar la RNN
    de pronóstico ni evaluar el control adaptativo frente a un motor envejecido.
    """

    # Table 3 de Saxena et al. (PHM 2008): desgaste natural de fondo por
    # componente en % (Initial / 3000 ciclos / 6000 ciclos). Los valores
    # están tabulados a los tres instantes discretos del paper; la
    # interpolación intermedia se hace en _interpolate_natural_wear
    # anclada en Initial y c6000. Excluye HPC porque este es el fault
    # mode del dataset y se modela por separado con la ec. 6 exponencial
    # estocástica.
    WEAR_TABLE = {
        'fan_eff':  {'initial': -0.18, 'c3000': -1.50, 'c6000': -2.85},
        'fan_flow': {'initial': -0.26, 'c3000': -2.04, 'c6000': -3.65},
        'lpc_eff':  {'initial': -0.62, 'c3000': -1.46, 'c6000': -2.61},
        'lpc_flow': {'initial': -1.01, 'c3000': -2.08, 'c6000': -4.00},
        'hpt_eff':  {'initial': -0.48, 'c3000': -2.63, 'c6000': -3.81},
        'hpt_flow': {'initial': +0.08, 'c3000': +1.76, 'c6000': +2.57},
        'lpt_eff':  {'initial': -0.10, 'c3000': -0.54, 'c6000': -1.08},
        'lpt_flow': {'initial': +0.08, 'c3000': +0.26, 'c6000': +0.42},
    }

    # Parámetros del modelo exponencial estocástico h(t) = 1 - exp(-a·t^b)
    # aplicado al HPC como fault mode principal (ec. 6 y ec. 10 del paper).
    HPC_EXP_MODEL = {
        'a_eff_range': (0.001, 0.003),
        'a_flow_range': (0.001, 0.003),
        'b_range': (1.4, 1.6),
    }

    # Mapeo entre componentes C-MAPSS y actuadores/estaciones del ACE.
    # 'affected_vars' son variables termodinámicas del corpus modificadas
    # multiplicativamente por (1 + delta * multiplier). Solo se mapean los
    # 4 componentes que la RNN predice a nivel agregado (fan, hpc, hpt, lpt);
    # las variantes de flujo (fan_flow, hpt_flow, etc.) y el lpc completo se
    # calculan como columnas para trazabilidad con Saxena Table 3 pero no se
    # propagan porque no aportan información nueva a la predicción agregada.
    ACE_MAPPING = {
        'fan_eff': {
            'ace_actuator': 'VFGV',
            'affected_vars': ['Nf'],
            'multiplier': 1.0,
        },
        'hpc_eff': {
            'ace_actuator': 'VGV-T',
            'affected_vars': ['T30', 'P30'],
            'multiplier': [2.0, 3.0],
        },
        'hpt_eff': {
            'ace_actuator': 'Sangrado',
            'affected_vars': ['T50'],
            'multiplier': 1.5,
        },
        # LPT es pasivo (sin actuador dedicado, a diferencia de fan, hpc
        # y hpt), pero su degradación afecta a la velocidad del eje de
        # alta (Nc) por su acople mecánico.
        'lpt_eff': {
            'ace_actuator': None,
            'affected_vars': ['Nc'],
            'multiplier': 1.0,
        },
    }

    def __init__(self, seed: int = 42, max_life_cycles: int = 6000):
        """Inicializa el inyector con su semilla y el horizonte de vida útil.

        max_life_cycles fija la escala temporal sobre la que se interpola el
        desgaste natural y se evalúa el modelo exponencial, por lo que debe
        coincidir con el horizonte usado al etiquetar la RUL; la semilla explícita
        garantiza que el corpus degradado sea reproducible entre ejecuciones.

        param seed: Semilla del generador aleatorio para reproducibilidad.
        param max_life_cycles: Vida útil máxima en ciclos usada como escala temporal.
        return: None; deja el inyector listo para invocar inject().
        """
        self.seed = seed
        self.max_life_cycles = max_life_cycles
        self.rng = np.random.default_rng(seed)
        logger.info(f"DegradationInjector inicializado "
                    f"(seed={seed}, max_cycles={max_life_cycles})")

    def inject(self, df: pd.DataFrame,
               noise_sigma: float = 0.001) -> pd.DataFrame:
        """Inyecta degradación estocástica sobre el corpus y lo devuelve ampliado.

        Asigna un ciclo de vida a cada muestra, calcula el desgaste natural de los
        ocho componentes de la Table 3, superpone el modelo exponencial del HPC,
        propaga un subconjunto de deltas a las variables termodinámicas y deriva
        el health index. Es el método central del módulo: convierte un corpus de
        motores nominales en un corpus con historial de deterioro, requisito para
        entrenar el pronóstico de RUL y para evaluar el control sobre motores
        envejecidos.

        Propagación selectiva a variables termodinámicas (ACE_MAPPING):
            - fan_eff → Nf     (velocidad eje baja)
            - hpc_eff → T30, P30 (salida compresor alta)
            - hpt_eff → T50    (salida turbina alta)
            - lpt_eff → Nc     (velocidad eje alta)
        Los deltas restantes (fan_flow, lpc_eff/flow, hpt_flow, lpt_flow,
        hpc_flow) se calculan y se guardan como columnas por trazabilidad con
        el paper de Saxena (Table 3 completa), pero no se propagan porque la
        RNN de pronóstico predice degradación agregada por componente, no por
        modo (eficiencia vs flujo) por separado.

        Añade al DataFrame las columnas life_cycle, {componente}_delta,
        {variable}_degraded y HI.

        param df: Corpus sintético generado por TMATSSimulator.batch.
        param noise_sigma: Desviación típica del ruido gaussiano por componente,
            que refleja la variabilidad de fabricación y ensamblaje entre
            motores del corpus. El valor por defecto 0.001 (0.1%) es dos
            órdenes de magnitud inferior a los deltas máximos de la Table 3
            (~1-4%), garantizando que el ruido no domine la señal de
            degradación pero introduzca dispersión realista.
        return: Copia del corpus con las columnas de degradación añadidas; el
            DataFrame original no se modifica.
        """
        df = df.copy()
        n = len(df)
        logger.info(f"Inyectando degradación en {n} muestras")

        # Asignar ciclo de vida uniforme por muestra
        df['life_cycle'] = self.rng.integers(0, self.max_life_cycles, n)
        t_normalized = df['life_cycle'].values / self.max_life_cycles

        # Parámetro b del modelo exponencial (compartido por HPC)
        b_values = self.rng.uniform(*self.HPC_EXP_MODEL['b_range'], n)

        # 1) Desgaste natural de fondo (interpolación de Table 3)
        for component in self.WEAR_TABLE.keys():
            delta = self._interpolate_natural_wear(component, t_normalized, b_values)
            delta += self.rng.normal(0, noise_sigma, n)
            df[f'{component}_delta'] = delta

        # 2) Degradación HPC estocástica (modelo exponencial ec. 6 Saxena)
        df['hpc_eff_delta'] = self._compute_hpc_exponential(
            'a_eff_range', t_normalized, b_values, noise_sigma, n)
        df['hpc_flow_delta'] = self._compute_hpc_exponential(
            'a_flow_range', t_normalized, b_values, noise_sigma, n)

        # 3) Aplicar deltas a variables termodinámicas
        for component, mapping in self.ACE_MAPPING.items():
            delta_col = f'{component}_delta'
            if delta_col not in df.columns:
                continue
            delta = df[delta_col].values
            affected = mapping['affected_vars']
            multipliers = mapping['multiplier']
            if not isinstance(multipliers, list):
                multipliers = [multipliers] * len(affected)
            for var, mult in zip(affected, multipliers):
                if var in df.columns:
                    df[f'{var}_degraded'] = df[var] * (1 + delta * mult)

        # 4) Health index normalizado como proxy lineal del ciclo de vida
        # restante: HI=1.0 en motor nuevo (t=0), HI=0.0 en fin de vida
        # (t=max_life_cycles). Simplificación deliberada respecto a un HI
        # basado en los deltas acumulados: el RNN de pronóstico aprende
        # directamente la RUL sin necesidad de que el HI represente
        # exactamente la salud instantánea.
        df['HI'] = (1 - t_normalized).clip(0, 1)

        logger.info(f"Degradación inyectada: {len(self.WEAR_TABLE)} componentes "
                    f"deterministas + HPC estocástico, HI medio={df['HI'].mean():.3f}")
        return df

    def _interpolate_natural_wear(self, component: str,
                                  t_norm: np.ndarray,
                                  b_values: np.ndarray) -> np.ndarray:
        """Interpola el desgaste natural de un componente entre Initial y c6000.

        Aplica un perfil monotónico t^b (inspirado en la curvatura exponencial
        de Saxena) anclado en los dos extremos publicados: initial en t=0 y
        c6000 en t=1. El punto intermedio c3000 de la Table 3 se conserva en
        WEAR_TABLE por trazabilidad con el paper y se emite en get_wear_table(),
        pero no se impone como restricción de la curva: con b ∈ [1.4, 1.6] la
        trayectoria pasa exactamente por los extremos y se desvía de c3000 en
        un 20-30% (por ejemplo, fan_eff a t=0.5 da ≈ -1.12% frente al -1.50%
        tabulado). La desviación se considera aceptable para el desgaste
        natural de fondo, dado que el fault mode dominante (HPC) se modela
        por separado con la ec. 6 completa.

        param component: Clave del componente en WEAR_TABLE (p. ej. 'fan_eff').
        param t_norm: Ciclos de vida normalizados a [0, 1] de cada muestra.
        param b_values: Exponentes b del modelo, muestreados por motor.
        return: Array de deltas de degradación en tanto por uno, negativos para
            pérdidas de eficiencia o flujo y positivos donde la Table 3 lo indica.
        """
        wear = self.WEAR_TABLE[component]
        initial = wear['initial'] / 100.0
        final = wear['c6000'] / 100.0
        return initial + (final - initial) * (t_norm ** b_values)

    def _compute_hpc_exponential(self, a_range_key: str,
                                 t_norm: np.ndarray,
                                 b_values: np.ndarray,
                                 noise_sigma: float,
                                 n: int) -> np.ndarray:
        """Aplica el modelo exponencial estocástico h(t) = 1 - exp(-a·t^b) al HPC.

        Sigue la ecuación 6 de Saxena et al. (PHM 2008) muestreando el parámetro
        `a` por motor dentro del rango de la ecuación 10. Se trata aparte del
        resto de componentes porque el HPC es el fault mode dominante del dataset:
        su dispersión entre motores es la que genera vidas útiles distintas y, por
        tanto, un problema de pronóstico no trivial.

        param a_range_key: Clave de HPC_EXP_MODEL con el rango de `a` a muestrear
            ('a_eff_range' para eficiencia, 'a_flow_range' para flujo).
        param t_norm: Ciclos de vida normalizados a [0, 1] de cada muestra.
        param b_values: Exponentes b del modelo, compartidos con el desgaste natural.
        param noise_sigma: Desviación típica del ruido gaussiano añadido al delta.
        param n: Número de muestras del corpus, usado para dimensionar el muestreo.
        return: Array de deltas negativos en tanto por uno con la pérdida de
            eficiencia o flujo del HPC en cada muestra.
        """
        a_lo, a_hi = self.HPC_EXP_MODEL[a_range_key]
        a_values = self.rng.uniform(a_lo, a_hi, n)
        # Modelo exponencial adaptado de la ec. 6 de Saxena (h(t) = 1-exp(-a·t^b)).
        # Se usa t normalizado en [0,1] en lugar del t en ciclos absolutos del
        # paper, con los parámetros a ∈ [0.001, 0.003] recalibrados para este
        # esquema (el rango original del paper asumía t en ciclos y valores de
        # a mucho menores). La normalización mantiene los deltas del HPC en el
        # mismo orden de magnitud (~0.2% máximo con a=0.003, b=1.6) que el
        # desgaste natural tabulado en la Table 3, garantizando propagación
        # multiplicativa estable a través de las variables termodinámicas y
        # coherencia con el corpus del RNN de pronóstico.
        # Signo negativo: pérdida de eficiencia/flujo
        delta = -(1 - np.exp(-a_values * t_norm ** b_values))
        delta += self.rng.normal(0, noise_sigma, n)
        return delta

    def summary(self, df: pd.DataFrame) -> None:
        """Registra en el log un resumen tabular de la degradación inyectada.

        Muestra el rango de ciclos de vida, el de health index y el delta medio y
        máximo de cada componente. Es necesario como comprobación rápida de que la
        magnitud del deterioro se mantiene en el orden publicado por C-MAPSS
        (unidades de porcentaje) y de que ningún componente ha quedado sin inyectar.

        param df: Corpus ya degradado devuelto por inject().
        return: None; el resumen se emite por el logger del módulo.
        """
        logger.info(f"Resumen degradación: {len(df)} muestras")
        logger.info(f"  Ciclo de vida: [{df['life_cycle'].min()}, "
                    f"{df['life_cycle'].max()}]")
        logger.info(f"  Health index:  [{df['HI'].min():.3f}, "
                    f"{df['HI'].max():.3f}]")

        components = list(self.WEAR_TABLE.keys()) + ['hpc_eff', 'hpc_flow']
        logger.info(f"  {'Componente':<12} {'Δ medio [%]':>12} {'Δ máx [%]':>12}")
        for component in components:
            col = f'{component}_delta'
            if col in df.columns:
                mean_pct = df[col].mean() * 100
                max_pct = df[col].abs().max() * 100
                logger.info(f"  {component:<12} {mean_pct:>12.4f} {max_pct:>12.4f}")

    def get_wear_table(self) -> pd.DataFrame:
        """Devuelve la Table 3 de Saxena et al. en formato DataFrame.

        Expone la tabla de desgaste natural codificada en WEAR_TABLE como una
        estructura tabular. Es necesario para reproducir la tabla en la memoria del
        TFG y para poder contrastar los valores realmente usados en el código con
        los publicados en el paper, sin duplicarlos a mano en otro sitio.

        return: DataFrame con una fila por componente y las columnas Component,
            Initial (%), 3000 cycles (%) y 6000 cycles (%).
        """
        rows = []
        for comp, values in self.WEAR_TABLE.items():
            rows.append({
                'Component': comp,
                'Initial (%)': values['initial'],
                '3000 cycles (%)': values['c3000'],
                '6000 cycles (%)': values['c6000'],
            })
        return pd.DataFrame(rows)