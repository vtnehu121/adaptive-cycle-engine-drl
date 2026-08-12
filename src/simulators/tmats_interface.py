"""
Última Fecha de Modificación: 08/Aug/2026
Descripción tmats_interface.py: Interfaz Python-MATLAB para el simulador T-MATS
del motor ACE. Encapsula la comunicación con el modelo
Simulink ACE_3stream_brayton.slx a través del MATLAB Engine API, traduciendo una
condición de vuelo (altitud, Mach, TRA, reparto de tercer flujo y sangrado) en
las variables termodinámicas del ciclo y en las magnitudes derivadas de empuje y
consumo. Es la fuente de verdad física de la que se deriva todo el corpus
sintético del TFG (5,000 muestras que alimentan el gemelo digital PINN).

Cada instancia mantiene una sesión de MATLAB persistente durante todo el batch
de simulaciones, minimizando el overhead de arranque. Las condiciones de vuelo
se inyectan directamente en el workspace de MATLAB mediante eng.workspace[...] y
los resultados se extraen con eng.eval(...), preservando la representación
binaria de doble precisión sin conversión textual intermedia.

Uso típico:
    sim = TMATSSimulator(models_path='models/')
    sim.start()
    result = sim.run(altitude=35000, mach=0.85, tra=100,
                     bpr_ts=0.3, bleed_fraction=0.05)
    sim.stop()

Referencia:
    Chapman, J.W., Lavelle, T.M., May, R.D., Litt, J.S., & Guo, T.H. (2014).
    "Toolbox for the Modeling and Analysis of Thermodynamic Systems (T-MATS)
    User's Guide." NASA/TM-2014-216638.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matlab.engine
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TMATSSimulator:
    """Interfaz Python-MATLAB para ejecutar simulaciones del modelo ACE.

    Mantiene una única sesión de MATLAB viva durante todo un batch. Es necesaria
    porque arrancar el Engine y cargar T-MATS cuesta decenas de segundos: abrir una
    sesión por simulación haría inviables los miles de puntos de operación del
    corpus. Además concentra en un solo sitio la conversión entre el vocabulario
    del proyecto (condiciones de vuelo) y el del modelo Simulink (estructuras MWS).
    """

    # Fracciones de bleed fraccional del HPC (refrigeración HPT/LPT).
    # Coherente con MWS.HPC.Fbld = [0.055, 0.035] en setup_HPC.m.
    FBLD_HPT = 0.055
    FBLD_LPT = 0.035

    # Estaciones termodinámicas expuestas como estructuras en el workspace
    # de MATLAB (cada una con campos W, ht, Tt, Pt, FAR)
    STATIONS = ['S2', 'S15', 'S21', 'S25', 'S3', 'S4', 'S45', 'S5']

    def __init__(self, models_path: str):
        """Inicializa el simulador y comprueba que el modelo Simulink existe.

        No arranca MATLAB todavía: solo valida la carpeta y el archivo .slx. Esta
        validación temprana es necesaria porque un error de ruta detectado tras
        levantar el Engine costaría el arranque completo de MATLAB antes de fallar.

        param models_path: Carpeta 'models/' con ACE_3stream_brayton.slx y el
            orquestador init_ACE.m.
        return: None; deja el objeto listo para start(), o lanza FileNotFoundError
            si falta la carpeta o el modelo Simulink.
        """
        self.models_path = Path(models_path)
        self.eng: Optional[matlab.engine.MatlabEngine] = None
        self.simulation_count = 0

        if not self.models_path.exists():
            raise FileNotFoundError(f"Carpeta models/ no encontrada: {self.models_path}")

        model_file = self.models_path / 'ACE_3stream_brayton.slx'
        if not model_file.exists():
            raise FileNotFoundError(f"Modelo Simulink no encontrado: {model_file}")

        logger.info(f"TMATSSimulator inicializado sobre {self.models_path}")

    def start(self) -> None:
        """Arranca MATLAB y carga el modelo ACE.

        Ejecuta init_ACE.m (que configura el path de T-MATS y carga los
        mapas del JT9D en el Model WorkSpace) y a continuación carga el
        modelo Simulink en memoria.

        Se separa del constructor porque concentra todo el coste de arranque: así
        el ciclo de vida de la sesión de MATLAB queda bajo control explícito del
        llamante y puede reutilizarse para miles de simulaciones consecutivas.

        return: None; deja self.eng operativo, o lanza RuntimeError si ya había
            una sesión de MATLAB iniciada.
        """
        if self.eng is not None:
            raise RuntimeError("MATLAB ya está iniciado.")

        logger.info("Iniciando MATLAB Engine...")
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(str(self.models_path), nargout=0)
        self.eng.run('init_ACE.m', nargout=0)
        self.eng.load_system('ACE_3stream_brayton', nargout=0)
        logger.info("MATLAB + T-MATS + modelo ACE cargados")

    def stop(self) -> None:
        """Cierra el modelo Simulink y detiene la sesión de MATLAB.

        Es necesario porque el proceso de MATLAB no muere al terminar el script de
        Python: sin esta llamada quedarían sesiones huérfanas consumiendo memoria y
        licencia. El cierre del modelo se envuelve en try/except para que un
        Simulink ya descargado no impida detener el Engine.

        return: None; deja self.eng a None, y no hace nada si ya estaba detenido.
        """
        if self.eng is None:
            return
        try:
            self.eng.close_system('ACE_3stream_brayton', 0, nargout=0)
        except matlab.engine.MatlabExecutionError:
            # Sistema ya cerrado o modelo no cargado.
            pass
        self.eng.quit()
        self.eng = None
        logger.info("MATLAB detenido")

    def run(self, altitude: float = 35000, mach: float = 0.85,
            tra: float = 100.0, bpr_ts: float = 0.3,
            bleed_fraction: float = 0.05) -> Dict[str, float]:
        """Ejecuta una simulación transitoria del motor ACE en un punto de operación.

        Inyecta la condición de vuelo en el workspace de MATLAB, fija la atmósfera
        ISA y el estado inicial del solver, lanza el modelo Simulink y devuelve el
        estado termodinámico convergido. Es el método elemental sobre el que se
        construye todo lo demás: cada fila del corpus sintético y cada paso del
        entorno de RL acaban traduciéndose en una llamada a run().

        param altitude: Altitud de vuelo en pies.
        param mach: Número de Mach de vuelo.
        param tra: Throttle Resolver Angle en % (20–100).
        param bpr_ts: Fracción de tercer flujo derivada por el splitter (0.05–0.60).
        param bleed_fraction: Fracción de sangrado customer del HPC (0.01–0.10),
            el actuador de aire auxiliar que controla el agente DRL.
        return: Diccionario con las condiciones de entrada, las variables
            termodinámicas de las estaciones y las derivadas (empuje, SFC, EPR,
            BPR); lanza RuntimeError si MATLAB no está iniciado.
        """
        if self.eng is None:
            raise RuntimeError("MATLAB no iniciado. Llame a start() primero.")

        self.simulation_count += 1

        # Inyectar condiciones de vuelo en el workspace
        self.eng.workspace['sim_altitude'] = float(altitude)
        self.eng.workspace['sim_mach'] = float(mach)
        self.eng.workspace['sim_tra'] = float(tra)
        self.eng.workspace['sim_bpr_ts'] = float(bpr_ts)
        self.eng.workspace['sim_bleed_fraction'] = float(bleed_fraction)

        # Configurar condiciones atmosféricas ISA y NRIC
        self._set_atmospheric_conditions(altitude, mach, tra, bpr_ts)

        # Ejecutar simulación transitoria
        self.eng.eval("sim('ACE_3stream_brayton');", nargout=0)

        # Extraer resultados del workspace
        return self._extract_results(altitude, mach, tra, bpr_ts, bleed_fraction)

    def _set_atmospheric_conditions(self, altitude: float, mach: float,
                                    tra: float, bpr_ts: float) -> None:
        """Configura condiciones atmosféricas ISA y estado inicial del solver.

        Calcula temperatura y presión ambiente según la Atmósfera Estándar
        Internacional (ICAO 1993), aplica las relaciones isentrópicas de
        remanso para gas caloríficamente perfecto (γ = 1.4) y actualiza
        el vector NRIC (Newton-Raphson Initial Conditions) del solver.

        Es necesario por dos motivos: el modelo T-MATS espera condiciones de
        remanso en la entrada, no altitud y Mach; y el solver Newton-Raphson
        multivariable solo converge si arranca cerca de la solución, por lo que las
        semillas del NRIC deben escalarse con el punto de operación pedido.

        param altitude: Altitud de vuelo en pies, que fija la atmósfera ISA.
        param mach: Número de Mach, que fija el salto estático-remanso.
        param tra: Throttle Resolver Angle en %, mapeado a demanda de FAR y a
            velocidades de eje.
        param bpr_ts: Fracción de tercer flujo, usada para sembrar el reparto de
            caudal en el NRIC.
        return: None; el efecto es la escritura de MWS.Input.* en el workspace.
        """
        # Atmósfera ISA (°R, psia)
        if altitude <= 36089:
            T_amb = 518.67 - 0.00356616 * altitude
            P_amb = 14.696 * (T_amb / 518.67) ** 5.2561
        else:
            T_amb = 389.97
            P_amb = 14.696 * 0.22336 * np.exp(-0.0000481 * (altitude - 36089))

        # Condiciones de remanso (gas caloríficamente perfecto, γ = 1.4)
        gamma = 1.4
        stagnation_factor = 1 + (gamma - 1) / 2 * mach ** 2
        T_total = T_amb * stagnation_factor
        P_total = P_amb * stagnation_factor ** (gamma / (gamma - 1))

        # Inyectar condiciones atmosféricas
        self.eng.eval(f"MWS.Input.Tt = {T_total};", nargout=0)
        self.eng.eval(f"MWS.Input.Pt = {P_total};", nargout=0)
        self.eng.eval(f"MWS.Input.Pamb = {P_amb};", nargout=0)

        # Demanda de fuel-air ratio: mapeo lineal TRA → FAR
        # TRA=20% → FAR=0.010 (idle), TRA=100% → FAR=0.030 (WOT)
        far_demand = 0.010 + (tra - 20) / 80 * 0.020
        self.eng.eval(f"MWS.Input.FARin = {far_demand};", nargout=0)

        # Flujo másico corregido por densidad relativa a sea level ISA
        rho_ratio = (P_total / 14.696) / (T_total / 518.67)
        W_ref = 674.22  # flujo de referencia del JT9D en punto de diseño
        W_adjusted = W_ref * rho_ratio
        self.eng.eval(f"MWS.Input.W = {W_adjusted};", nargout=0)

        # Velocidades de eje escaladas con TRA
        lp_shaft = 3200 + (tra - 20) / 80 * 1500
        hp_shaft = 6500 + (tra - 20) / 80 * 2500
        self.eng.eval(f"MWS.Input.LP_Shaft = {lp_shaft};", nargout=0)
        self.eng.eval(f"MWS.Input.HP_Shaft = {hp_shaft};", nargout=0)

        # Vector NRIC del solver Newton-Raphson multivariable (9 semillas).
        # Orden: [W_inlet, PR_LPC, W_bypass_split, PR_HPC, PR_HPT, PR_LPT,
        #         EPR_global, Nf, Nc]. Los ratios (2.37, 1.77, 2.04, 2.67,
        #         4.89) son estimaciones del punto de diseño del JT9D
        # heredadas del ejemplo T-MATS oficial (Chapman et al. 2014); el
        # solver converge por Newton-Raphson en pocas iteraciones si las
        # semillas están dentro del ±30% del valor correcto. El reparto
        # de bypass se escala con bpr_ts para acompañar la fase de vuelo.
        nric = (f"[{W_adjusted}, 2.37, {5.0 * (1 - bpr_ts)}, 1.77, 2.04, "
                f"2.67, 4.89, {lp_shaft}, {hp_shaft}]")
        self.eng.eval(f"MWS.Input.NRIC = {nric};", nargout=0)

    def _extract_results(self, altitude: float, mach: float, tra: float,
                         bpr_ts: float, bleed_fraction: float) -> Dict[str, float]:
        """Extrae del workspace las variables termodinámicas y derivadas.

        Lee las estructuras de estación (S2, S15, S21, S25, S3, S4, S45,
        S5) generadas por el logging del modelo Simulink y las convierte
        en un diccionario con las 26 columnas del corpus + metadatos.

        W31, W32 se calculan a partir del fractional bleed del HPC
        (refrigeración estructural HPT y LPT). W_bleed refleja el
        customer bleed real capturado en S3_CustBld, controlado por el
        actuador DRL sim_bleed_fraction.

        Se aísla del método run() porque concentra la traducción al esquema de
        columnas de C-MAPSS, que es lo que permite comparar el corpus ACE con el
        dataset de NASA. Cada lectura se protege con try/except para que una
        variable no registrada por el modelo produzca un NaN puntual en lugar de
        invalidar la simulación completa.

        param altitude: Altitud de vuelo en pies, replicada en el resultado.
        param mach: Número de Mach de vuelo, replicado en el resultado.
        param tra: Throttle Resolver Angle en %, replicado en el resultado.
        param bpr_ts: Fracción de tercer flujo, replicada en el resultado.
        param bleed_fraction: Fracción de sangrado pedida, replicada en el resultado.
        return: Diccionario con condiciones de entrada, marca temporal, variables
            de estación y magnitudes derivadas; los campos no disponibles quedan
            como NaN.
        """
        result: Dict[str, float] = {
            'altitude': altitude,
            'mach': mach,
            'tra': tra,
            'bpr_ts': bpr_ts,
            'bleed_fraction': bleed_fraction,
            'timestamp': datetime.now().isoformat(),
        }

        # Variables directas de las estaciones termodinámicas
        stations_map = {
            'S2':  {'T2': 'Tt', 'P2': 'Pt', 'W_inlet': 'W'},
            'S15': {'T24': 'Tt', 'P15': 'Pt'},
            'S3':  {'T30': 'Tt', 'P30': 'Pt'},
            'S4':  {'T4': 'Tt'},
            'S5':  {'T50': 'Tt'},
        }
        for station, fields in stations_map.items():
            for name, field in fields.items():
                try:
                    result[name] = float(self.eng.eval(
                        f"{station}.{field}.Data(end)", nargout=1))
                except (matlab.engine.MatlabExecutionError, ValueError) as e:
                    # MATLAB no encuentra la variable o el valor no es
                    # convertible a float → NaN. Otros errores (memoria,
                    # KeyboardInterrupt) se propagan para no ocultarlos.
                    logger.debug(f"Variable {station}.{field} no disponible: {e}")
                    result[name] = float('nan')

        # Flujo másico entrando al HPC (referencia para bleeds fraccionales)
        try:
            W_S25 = float(self.eng.eval("S25.W.Data(end)", nargout=1))
        except (matlab.engine.MatlabExecutionError, ValueError):
            W_S25 = float('nan')
    
        # Variables derivadas
        try:
            result['epr'] = result.get('P30', 0) / max(result.get('P2', 1), 0.01)
            result['Ps30'] = result.get('P30', 0) * 0.95  # aprox estática

            result['BPR'] = float(self.eng.eval(
                "S15.W.Data(end) / S21.W.Data(end)", nargout=1))
            result['farB'] = float(self.eng.eval("S4.FAR.Data(end)", nargout=1))

            # Velocidades de eje
            result['Nf'] = float(self.eng.eval("FData.Nmech.Data(end)", nargout=1))
            result['Nc'] = float(self.eng.eval("HPCData.Nmech.Data(end)", nargout=1))

            # Velocidades corregidas ISA
            theta2 = result.get('T2', 518.67) / 518.67
            result['NRf'] = result['Nf'] / np.sqrt(theta2)
            result['NRc'] = result['Nc'] / np.sqrt(theta2)

            # Ratio combustible/presión estática
            wf = float(self.eng.eval(
                "S4.FAR.Data(end) * S4.W.Data(end)", nargout=1))
            result['phi'] = wf / max(result.get('Ps30', 1), 0.01)

            # Entalpía del bleed fraccional al HPT
            result['htBleed'] = float(self.eng.eval("S3.ht.Data(end)", nargout=1))

            # Bleeds fraccionales (refrigeración estructural HPT/LPT)
            result['W31'] = self.FBLD_HPT * W_S25
            result['W32'] = self.FBLD_LPT * W_S25

            # Customer bleed en lb/s 
            try:
                fraction = float(self.eng.eval(
                    "S3_CustBld.Data(end,1)", nargout=1))
                result['W_bleed'] = fraction * W_S25
            except (matlab.engine.MatlabExecutionError, ValueError):
                result['W_bleed'] = float('nan')

            # Demandas de velocidad
            result['Nf_dmd'] = result['Nf']
            result['PCNfR_dmd'] = result['NRf']

            # Empuje total del motor
            try:
                fg_core = float(self.eng.eval("Fg_core.Data(end)", nargout=1))
                fg_byp = float(self.eng.eval("Fg_byp.Data(end)", nargout=1))
                result['thrust_approx'] = fg_core + fg_byp
            except (matlab.engine.MatlabExecutionError, ValueError):
                result['thrust_approx'] = float('nan')

            # SFC en lb/h·lbf
            if result['thrust_approx'] > 0:
                result['sfc'] = wf * 3600 / result['thrust_approx']
            else:
                result['sfc'] = float('nan')

            # Consumo de combustible (lb/s)
            result['wf'] = wf

        except (matlab.engine.MatlabExecutionError, ValueError,
                ZeroDivisionError, KeyError) as e:
            # Fallo típico: variable no registrada por el modelo, o
            # división por cero cuando thrust=0. Se registra qué variables
            # ya se habían calculado para diagnóstico posterior.
            calculated = [k for k in result.keys() if k not in
                          {'altitude', 'mach', 'tra', 'bpr_ts',
                           'bleed_fraction', 'timestamp'}]
            logger.warning(
                f"Error calculando variables derivadas: {e}. "
                f"Calculadas parcialmente: {calculated}")
        return result

    def batch(self, conditions_df: pd.DataFrame,
              save_path: Optional[str] = None) -> pd.DataFrame:
        """Ejecuta un conjunto de simulaciones sobre un DataFrame de condiciones.

        Cada fila del DataFrame debe contener las columnas altitude, mach,
        tra, bpr_ts y bleed_fraction. Las simulaciones que fallan por
        divergencia del solver se registran y se descartan sin interrumpir
        el resto del batch.

        Es necesario porque generar el corpus supone miles de ejecuciones
        encadenadas sobre la misma sesión de MATLAB: tolerar los fallos aislados
        del Newton-Raphson y volcar el resultado a CSV evita perder horas de
        cómputo por un puñado de puntos de operación no convergidos.

        param conditions_df: DataFrame de condiciones de vuelo, una por fila.
        param save_path: Ruta CSV donde volcar los resultados; si es None no se
            escribe nada en disco.
        return: DataFrame con una fila por simulación convergida, en el mismo orden
            de entrada y sin las filas fallidas; lanza RuntimeError si MATLAB no
            está iniciado.
        """
        if self.eng is None:
            raise RuntimeError("MATLAB no iniciado.")

        logger.info(f"Iniciando batch de {len(conditions_df)} simulaciones")

        results = []
        failed = 0
        for i, (idx, row) in enumerate(conditions_df.iterrows()):
            try:
                output = self.run(
                    altitude=row['altitude'],
                    mach=row['mach'],
                    tra=row.get('tra', 100.0),
                    bpr_ts=row.get('bpr_ts', 0.3),
                    bleed_fraction=row.get('bleed_fraction', 0.05),
                )
                results.append(output)
            except (matlab.engine.MatlabExecutionError, RuntimeError) as e:
                logger.warning(f"Simulación idx={idx} (paso {i}) falló: {e}")
                failed += 1

            if (i + 1) % 100 == 0:
                logger.info(f"  Progreso: {i + 1}/{len(conditions_df)}")

        results_df = pd.DataFrame(results)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(save_path, index=False)
            logger.info(f"Resultados guardados: {save_path}")

        logger.info(f"Batch completado: {len(results)} exitosas, {failed} fallidas")
        return results_df