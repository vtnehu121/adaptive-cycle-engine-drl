"""
Última Fecha de Modificación: 08/Aug/2026
Descripción ace_env.py: Entorno Gymnasium para el control por aprendizaje por
refuerzo profundo del motor ACE. Envuelve el
gemelo digital PINN y expone al agente un espacio de cuatro actuadores continuos
y una observación de 19 variables normalizadas, junto con la secuencia de fases
de misión, la comprobación de restricciones termomecánicas y la recompensa
multiobjetivo con pesos dependientes de la fase. Es la formulación como CMDP
sobre la que se entrenan y evalúan los agentes.

Espacio de acción (4 actuadores continuos normalizados en [-1, 1]):
    - VFGV:  Ángulo Variable Fan Guide Vanes → [-20°, +20°]
    - VGV-T: Ángulo Variable Guide Vanes Turbina → [-15°, +15°]
    - AB:    Afterburner → [0, 1] (fracción)
    - Bleed: Sangrado de aire → [0.01, 0.10] (fracción)

Espacio de observación (19 variables, composición ordenada):
    - 10 sensores termodinámicos: T2, T4, T30, T50, P2, P30, Nf, Nc, thrust, sfc
    - 4 posiciones actuadores: vfgv_angle, vgvt_angle, ab_fraction, bleed_fraction
    - 4 condiciones de vuelo: altitude, mach, tra, mission_phase_encoded
    - 1 índice de salud: health_index
"""

import logging
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


class ACEEnv(gym.Env):
    """
    Entorno Gymnasium para control del motor ACE de tercer flujo.

    El agente controla 4 actuadores para optimizar simultáneamente
    empuje, SFC, potencia eléctrica y refrigeración, respetando
    las restricciones termomecánicas (T4, surge margins).

    Es necesario porque los algoritmos de RL de la librería usada solo saben hablar
    la interfaz de Gymnasium: esta clase traduce entre esa interfaz y el dominio del
    motor, y es además donde vive la decisión de diseño central del TFG, que es cómo
    se compensan entre sí empuje, consumo, potencia extraída y refrigeración según la
    fase de vuelo sin violar los límites de seguridad.
    """

    metadata = {'render_modes': ['human']}

    # Regímenes de misión con condiciones típicas
    MISSION_PHASES = {
        'takeoff':  {'altitude': 0,     'mach': 0.30, 'tra': 100, 'duration': 20},
        'climb':    {'altitude': 20000, 'mach': 0.60, 'tra': 90,  'duration': 40},
        'cruise':   {'altitude': 35000, 'mach': 0.85, 'tra': 80,  'duration': 100},
        'combat':   {'altitude': 15000, 'mach': 1.60, 'tra': 100, 'duration': 30},
        'descent':  {'altitude': 10000, 'mach': 0.50, 'tra': 40,  'duration': 30},
    }

    # Restricciones de seguridad (hard constraints)
    SAFETY_LIMITS = {
        'T4_max': 3200,          # °R — límite turbina
        'T4_warning': 3000,      # °R — umbral advertencia
        'SmFan_min': 8.0,        # % — margen stall fan
        'SmHPC_min': 5.0,        # % — margen stall HPC
        'Nf_max': 6000,          # rpm
        'Nc_max': 12000,         # rpm
    }

    def __init__(self, pinn_model=None, mission_profile: str = 'mixed',
                 max_steps: int = 200, degradation_rate: float = 0.0,
                 reward_weights=None):
        """Define los espacios de acción y observación e inicializa el estado interno.

        Las acciones se declaran normalizadas en [-1, 1] y se traducen después a las
        unidades físicas de cada actuador; esto es necesario porque los algoritmos de
        política continua asumen acciones simétricas y acotadas, y mezclar grados con
        fracciones en el espacio nativo desequilibraría la exploración.

        param pinn_model: Gemelo digital PINN entrenado; si es None se usa el modelo
            simplificado interno, pensado solo para pruebas.
        param mission_profile: Perfil de misión: 'mixed', 'cruise', 'combat' o 'full'.
        param max_steps: Número máximo de pasos por episodio.
        param degradation_rate: Deterioro aplicado por paso; 0 desactiva la degradación.
        param reward_weights: Pesos de recompensa por fase; si es None se usan los de
            la tabla por defecto del TFG.
        return: None; deja el entorno listo para reset().
        """
        super().__init__()

        self.pinn_model = pinn_model
        self.mission_profile = mission_profile
        self.max_steps = max_steps
        self.degradation_rate = degradation_rate
        self.reward_weights = reward_weights

        # Espacio de acción: 4 actuadores continuos normalizados a [-1, 1]
        # Se desnormalizan internamente a los rangos reales
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(4,), dtype=np.float32
        )

        # Espacio de observación: 19 variables
        # [T2, T4, T30, T50, P2, P30, Nf, Nc, thrust, sfc,
        #  vfgv_angle, vgvt_angle, ab_fraction, bleed_fraction,
        #  altitude, mach, tra, mission_phase_encoded, health_index]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(19,), dtype=np.float32
        )

        # Estado interno
        self.current_step = 0
        self.current_phase_idx = 0
        self.mission_sequence = []
        self.actuator_state = np.zeros(4, dtype=np.float32)
        self.health_index = 1.0
        self.episode_reward = 0.0
        self.episode_violations = 0

        # Historial del episodio
        self.history = []

        logger.info(f"ACEEnv inicializado: profile={mission_profile}, "
                    f"max_steps={max_steps}")

    def _build_mission_sequence(self) -> list:
        """Construye la secuencia de fases de misión que recorrerá el episodio.

        Los perfiles fijos aíslan un régimen concreto para estudiarlo, el perfil
        'full' reproduce una misión realista completa y 'mixed' encadena fases y
        duraciones al azar. Este último es el que se usa para entrenar, porque
        expone al agente a transiciones variadas y evita que aprenda la secuencia de
        memoria en lugar de aprender a reaccionar a la fase actual.

        return: Lista de tuplas (nombre de fase, duración en pasos) cuya suma cubre
            al menos max_steps.
        """
        if self.mission_profile == 'cruise':
            return [('cruise', self.max_steps)]
        elif self.mission_profile == 'combat':
            return [('combat', self.max_steps)]
        elif self.mission_profile == 'full':
            return [
                ('takeoff', 20), ('climb', 40), ('cruise', 60),
                ('combat', 30), ('cruise', 30), ('descent', 20),
            ]
        else:  # mixed
            # Usa el generador del entorno (self.np_random, inicializado por
            # super().reset(seed=seed)) en lugar de np.random global, para que
            # la secuencia de fases sea reproducible cuando el usuario fija
            # una semilla en env.reset(seed=42).
            phases = list(self.MISSION_PHASES.keys())
            seq = []
            remaining = self.max_steps
            while remaining > 0:
                phase = phases[self.np_random.integers(len(phases))]
                duration = min(int(self.np_random.integers(10, 50)), remaining)
                seq.append((phase, duration))
                remaining -= duration
            return seq

    def _get_current_conditions(self) -> Dict[str, float]:
        """Obtiene las condiciones de vuelo correspondientes al paso actual.

        Recorre la secuencia acumulando duraciones hasta localizar la fase en curso.
        Es necesario porque la condición de vuelo no la elige el agente: es el
        contexto exógeno que define el problema en cada instante. Se devuelve una
        copia para que ningún llamante altere la tabla de fases compartida.

        return: Diccionario con altitude, mach, tra y duration de la fase actual; si
            el paso excede la secuencia se devuelve la fase de crucero.
        """
        total = 0
        for phase, duration in self.mission_sequence:
            total += duration
            if self.current_step < total:
                return self.MISSION_PHASES[phase].copy()
        return self.MISSION_PHASES['cruise'].copy()

    def _get_current_phase_name(self) -> str:
        """Obtiene el nombre de la fase de misión activa en el paso actual.

        Se necesita por separado de las condiciones porque el nombre de la fase
        interviene en dos sitios más: selecciona los pesos de la recompensa y se
        codifica dentro de la observación, de modo que el agente sepa en qué régimen
        se encuentra y pueda condicionar su política.

        return: Nombre de la fase actual, o 'cruise' si el paso excede la secuencia.
        """
        total = 0
        for phase, duration in self.mission_sequence:
            total += duration
            if self.current_step < total:
                return phase
        return 'cruise'

    def _denormalize_actions(self, actions: np.ndarray) -> Dict[str, float]:
        """
        Desnormaliza acciones de [-1, 1] a rangos reales de actuadores.

        Acciones:
            [0] VFGV:  [-1, 1] → [-20°, +20°]
            [1] VGV-T: [-1, 1] → [-15°, +15°]
            [2] AB:    [-1, 1] → [0, 1]
            [3] Bleed: [-1, 1] → [0.01, 0.10]

        Es necesario porque la política emite un vector homogéneo y adimensional
        mientras que el motor exige ángulos y fracciones en unidades físicas con
        rangos muy distintos. Al ser el mapeo afín y monótono, cualquier acción
        válida de la política cae dentro del rango admisible del actuador, lo que
        elimina la necesidad de recortes posteriores.

        param actions: Vector de cuatro acciones normalizadas en [-1, 1].
        return: Diccionario con vfgv_angle, vgvt_angle, ab_fraction y
            bleed_fraction en unidades físicas.
        """
        return {
            'vfgv_angle': float(actions[0] * 20.0),
            'vgvt_angle': float(actions[1] * 15.0),
            'ab_fraction': float((actions[2] + 1.0) / 2.0),
            'bleed_fraction': float(0.01 + (actions[3] + 1.0) / 2.0 * 0.09),
        }

    def _query_pinn(self, conditions: Dict, actuators: Dict) -> Dict[str, float]:
        """
        Consulta al Gemelo Digital PINN.

        Combina condiciones de vuelo + actuadores para obtener
        la respuesta termodinámica del motor.

        El ángulo VFGV se traduce aquí a la fracción de tercer flujo bpr_ts, que es
        la variable que realmente entiende el gemelo digital. Concentrar en este
        método la consulta al modelo y la aplicación de la degradación permite que el
        resto del entorno sea independiente de si detrás hay un PINN entrenado o el
        modelo simplificado de pruebas.

        param conditions: Condiciones de vuelo de la fase actual.
        param actuators: Posiciones de los actuadores ya en unidades físicas.
        return: Diccionario con el estado termodinámico del motor, con el empuje y
            el consumo ya penalizados por la degradación acumulada.
        """
        # BPR del tercer flujo controlado por VFGV
        bpr_ts = 0.3 + actuators['vfgv_angle'] / 20.0 * 0.2

        if self.pinn_model is not None:
            result = self.pinn_model.predict(
                altitude=conditions['altitude'],
                mach=conditions['mach'],
                tra=conditions['tra'],
                bpr_ts=bpr_ts
            )
        else:
            # Modelo simplificado si no hay PINN (para testing)
            result = self._simplified_engine(conditions, actuators, bpr_ts)

        # Aplicar degradación
        if self.degradation_rate > 0:
            deg = 1.0 - self.degradation_rate * self.current_step
            result['thrust_approx'] *= max(deg, 0.8)
            result['sfc'] /= max(deg, 0.8)

        return result

    def _simplified_engine(self, conditions: Dict, actuators: Dict,
                           bpr_ts: float) -> Dict[str, float]:
        """Modelo termodinámico simplificado, usado solo cuando no hay PINN.

        Aplica relaciones algebraicas groseras entre condición de vuelo, actuadores y
        estado del motor. No pretende ser fiel: existe para que los tests unitarios y
        las pruebas de integración del entorno puedan ejecutarse sin cargar un
        checkpoint del gemelo digital ni depender de una GPU.

        param conditions: Condiciones de vuelo de la fase actual.
        param actuators: Posiciones de los actuadores en unidades físicas.
        param bpr_ts: Fracción de tercer flujo derivada del ángulo VFGV.
        return: Diccionario con las mismas claves de estado que devuelve el PINN,
            para que ambos caminos sean intercambiables.
        """
        alt = conditions['altitude']
        mach = conditions['mach']
        tra = conditions['tra']

        T2 = 518.67 - 0.00357 * alt
        T4 = T2 * 5.0 * (tra / 100) * (1 + 0.1 * actuators['ab_fraction'])
        T30 = T2 * 2.5
        T50 = T4 * 0.6
        P2 = 14.696 * (T2 / 518.67) ** 5.256
        P30 = P2 * 25
        Nf = 3000 + 2000 * (tra / 100)
        Nc = Nf * 2.0
        thrust = 50000 * (tra / 100) * (1 + 0.5 * actuators['ab_fraction'])
        sfc = 0.5 + 0.3 * actuators['ab_fraction']

        return {
            'thrust_approx': thrust, 'sfc': sfc,
            'T2': T2, 'T4': T4, 'T30': T30, 'T50': T50,
            'P2': P2, 'P30': P30, 'Nf': Nf, 'Nc': Nc
        }

    def _compute_observation(self, engine_state: Dict,
                             actuators: Dict, conditions: Dict) -> np.ndarray:
        """Construye el vector de observación normalizado que recibe el agente.

        Reúne estado del motor, posición de actuadores, condición de vuelo, fase
        codificada y health index, dividiendo cada magnitud por su orden esperado.
        La normalización es imprescindible: sin ella conviven en el mismo vector
        temperaturas de miles de grados y fracciones menores que uno, lo que satura
        las activaciones de la red de política y bloquea el aprendizaje.

        param engine_state: Estado termodinámico devuelto por el gemelo digital.
        param actuators: Posiciones de los actuadores en unidades físicas.
        param conditions: Condiciones de vuelo de la fase actual.
        return: Array de 19 float32 aproximadamente en [0, 1] con la observación,
            usando valores por defecto razonables si falta alguna variable.
        """
        phase_map = {'takeoff': 0, 'climb': 1, 'cruise': 2,
                     'combat': 3, 'descent': 4}
        phase = self._get_current_phase_name()

        # Normalización vital para evitar saturación del gradiente (Vanishing Gradient)
        obs = np.array([
            engine_state.get('T2', 500) / 1000.0,
            engine_state.get('T4', 2500) / 4000.0,
            engine_state.get('T30', 1400) / 2000.0,
            engine_state.get('T50', 1300) / 2000.0,
            engine_state.get('P2', 14) / 50.0,
            engine_state.get('P30', 300) / 500.0,
            engine_state.get('Nf', 4000) / 10000.0,
            engine_state.get('Nc', 8000) / 20000.0,
            engine_state.get('thrust_approx', 20000) / 100000.0,
            engine_state.get('sfc', 0.5) / 2.0,
            actuators['vfgv_angle'] / 20.0,
            actuators['vgvt_angle'] / 15.0,
            actuators['ab_fraction'],
            actuators['bleed_fraction'] / 0.10,
            conditions['altitude'] / 50000.0,
            conditions['mach'] / 2.0,
            conditions['tra'] / 100.0,
            phase_map.get(phase, 2) / 4.0,
            self.health_index,
        ], dtype=np.float32)

        return obs

    def _check_safety(self, engine_state: Dict) -> Tuple[bool, Dict[str, bool]]:
        """
        Verifica restricciones de seguridad termomecánica.

        Restricciones:
            1. T₄ < T₄_max (temperatura combustor)
            2. Nf < Nf_max (velocidad fan)
            3. Nc < Nc_max (velocidad core)
            4. SmFan > SmFan_min (margen surge fan — proxy)
            5. SmHPC > SmHPC_min (margen surge HPC — proxy)

        Surge margins estimados como proxy:
            SmFan ≈ (Nf_max - Nf) / Nf_max × 100
            SmHPC ≈ (Nc_max - Nc) / Nc_max × 100

        Es necesario porque el problema está formulado como CMDP: la seguridad no es
        un objetivo más que se pueda compensar con empuje, sino una restricción que
        se contabiliza aparte. Los márgenes de bombeo se aproximan a partir de las
        velocidades de eje porque el gemelo digital no expone los mapas de
        compresor de los que saldrían los valores exactos.

        param engine_state: Estado termodinámico devuelto por el gemelo digital.
        return: Tupla (booleano de operación segura, diccionario con cada violación
            individual y los márgenes SmFan y SmHPC calculados).
        """
        Nf = engine_state.get('Nf', 0)
        Nc = engine_state.get('Nc', 0)

        # Surge margin proxy: aproximación TOSCA basada en la distancia
        # relativa a la velocidad máxima del eje. NO coincide con el SmFan
        # aeronáutico exacto (que se calcularía del mapa del compresor,
        # midiendo la distancia entre línea de trabajo y línea de bombeo
        # en el diagrama presión-flujo), pero el gemelo digital PINN no
        # expone los mapas de compresor. Sirve como aproximación cualitativa
        # para el logging de safety y como término de penalización proxy en
        # la recompensa (ver _compute_reward).
        sm_fan = max(0, (self.SAFETY_LIMITS['Nf_max'] - Nf) / self.SAFETY_LIMITS['Nf_max'] * 100)
        sm_hpc = max(0, (self.SAFETY_LIMITS['Nc_max'] - Nc) / self.SAFETY_LIMITS['Nc_max'] * 100)

        violations = {
            'T4_exceeded': engine_state.get('T4', 0) > self.SAFETY_LIMITS['T4_max'],
            'Nf_exceeded': Nf > self.SAFETY_LIMITS['Nf_max'],
            'Nc_exceeded': Nc > self.SAFETY_LIMITS['Nc_max'],
            'SmFan_low': sm_fan < self.SAFETY_LIMITS['SmFan_min'],
            'SmHPC_low': sm_hpc < self.SAFETY_LIMITS['SmHPC_min'],
        }

        # Guardar márgenes en estado para logging
        violations['SmFan'] = sm_fan
        violations['SmHPC'] = sm_hpc

        safe = not any(v for k, v in violations.items()
                       if k.endswith('exceeded') or k.endswith('low'))
        return safe, violations

    def reset(self, seed=None, options=None):
        """Reinicia el entorno y devuelve la observación inicial del episodio.

        Genera una nueva secuencia de misión y sortea un health index inicial en
        [0.7, 1.0]. Ese sorteo es deliberado: obliga al agente a enfrentarse desde el
        primer paso a motores con distinto grado de deterioro, que es la condición
        para que la política resultante sea adaptativa y no esté ajustada a un motor
        nuevo.

        param seed: Semilla propagada al generador del entorno base de Gymnasium.
        param options: Opciones adicionales de la interfaz Gymnasium; no se usan.
        return: Tupla (observación inicial de 19 componentes, diccionario info con
            la fase de partida y el health index sorteado).
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.actuator_state = np.zeros(4, dtype=np.float32)
        # Usa self.np_random (inicializado por super().reset arriba) para que
        # el health_index inicial sea reproducible con la semilla del entorno.
        self.health_index = 1.0 - float(self.np_random.uniform(0, 0.3))
        self.episode_reward = 0.0
        self.episode_violations = 0
        self.history = []
        self.mission_sequence = self._build_mission_sequence()

        conditions = self._get_current_conditions()
        actuators = self._denormalize_actions(self.actuator_state)
        engine_state = self._query_pinn(conditions, actuators)
        obs = self._compute_observation(engine_state, actuators, conditions)

        info = {'phase': self._get_current_phase_name(),
                'health_index': self.health_index}

        return obs, info

    def step(self, action: np.ndarray):
        """
        Ejecuta un paso de control en el entorno y devuelve la transición resultante.

        Desnormaliza la acción, consulta el gemelo digital, comprueba las
        restricciones de seguridad, calcula la recompensa y avanza la degradación.
        Es el corazón de la interacción agente-motor. El episodio nunca termina por
        violación de seguridad, solo se trunca al agotar los pasos: cortar el
        episodio ante una violación daría al agente el incentivo perverso de
        provocarla para escapar de un estado de baja recompensa.

        param action: Vector de cuatro acciones normalizadas en [-1, 1].
        return: Tupla (observación, recompensa, terminated siempre False, truncated
            al alcanzar max_steps, info con fase, salud, empuje, SFC, T4, estado de
            seguridad y violaciones acumuladas).
        """
        self.current_step += 1

        # Desnormalizar acciones
        actuators = self._denormalize_actions(action)
        self.actuator_state = action.copy()

        # Obtener condiciones de vuelo
        conditions = self._get_current_conditions()

        # Consultar PINN
        engine_state = self._query_pinn(conditions, actuators)

        # Verificar seguridad
        safe, violations = self._check_safety(engine_state)
        if not safe:
            self.episode_violations += 1

        # Calcular recompensa (delegado a reward.py)
        phase = self._get_current_phase_name()
        reward = self._compute_reward(engine_state, actuators, phase, safe)
        self.episode_reward += reward

        # Degradación progresiva
        if self.degradation_rate > 0:
            self.health_index = max(0, self.health_index - self.degradation_rate)

        # Construir observación
        obs = self._compute_observation(engine_state, actuators, conditions)

        # Condiciones de terminación: nunca terminated=True porque cortar
        # el episodio ante violación de seguridad crearía incentivo perverso
        # (ver docstring del método). Solo se trunca al agotar max_steps.
        terminated = False
        truncated = self.current_step >= self.max_steps

        # Info
        info = {
            'phase': phase,
            'health_index': self.health_index,
            'thrust': engine_state.get('thrust_approx', 0),
            'sfc': engine_state.get('sfc', 0),
            'T4': engine_state.get('T4', 0),
            'safe': safe,
            'violations': violations,
            'episode_violations': self.episode_violations,
            'SmFan': violations.get('SmFan', 0),
            'SmHPC': violations.get('SmHPC', 0),
        }

        self.history.append(info)

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, engine_state: Dict, actuators: Dict,
                        phase: str, safe: bool) -> float:
        """
        Calcula recompensa multiobjetivo con pesos dinámicos por fase.

        Cuatro objetivos operativos:
            1. Empuje: maximizar en combate, mantener en crucero
            2. SFC: minimizar siempre (eficiencia)
            3. Potencia eléctrica: proxy via sangrado de aire (bleed)
            4. Refrigeración: proxy via tercer flujo + T4

        Restricción de seguridad Safe-RL:
            - Barrera logarítmica en T4 cerca del límite
            - Penalización fuerte por violación

        Los pesos dependen de la fase porque el compromiso entre objetivos cambia por
        completo a lo largo de la misión: en combate manda el empuje y en crucero el
        consumo. La barrera logarítmica sobre T4 es necesaria para que la señal de
        peligro aparezca de forma progresiva al acercarse al límite, en lugar de
        surgir de golpe al cruzarlo, cuando ya no hay margen para corregir.

        param engine_state: Estado termodinámico devuelto por el gemelo digital.
        param actuators: Posiciones de los actuadores en unidades físicas.
        param phase: Nombre de la fase de misión actual, que selecciona los pesos.
        param safe: Resultado de la comprobación de restricciones de este paso.
        return: Recompensa escalar del paso, suma ponderada de empuje, eficiencia,
            potencia, refrigeración y seguridad, más el bono por paso seguro o la
            penalización proporcional al exceso de T4.
        """
        # Pesos dinámicos por fase de misión (5 fases × 5 componentes)
        default_weights = {
            'takeoff':  {'thrust': 0.45, 'sfc': 0.10, 'power': 0.05, 'cooling': 0.10, 'safety': 0.30},
            'climb':    {'thrust': 0.30, 'sfc': 0.25, 'power': 0.05, 'cooling': 0.10, 'safety': 0.30},
            'cruise':   {'thrust': 0.10, 'sfc': 0.40, 'power': 0.15, 'cooling': 0.10, 'safety': 0.25},
            'combat':   {'thrust': 0.45, 'sfc': 0.05, 'power': 0.05, 'cooling': 0.15, 'safety': 0.30},
            'descent':  {'thrust': 0.05, 'sfc': 0.30, 'power': 0.20, 'cooling': 0.15, 'safety': 0.30},
        }

        # Si el usuario ha pasado reward_weights personalizados, usarlos
        # Si no, usar los pesos por defecto (Tabla 5.5 del Word)
        weights = self.reward_weights if self.reward_weights is not None else default_weights

        w = weights.get(phase, weights['cruise'])

        # Componente 1: Empuje normalizado [0, 1]
        # Mayor empuje → mayor recompensa. Normalizado a 100,000 lbf que
        # es aproximadamente el empuje máximo con afterburner activo.
        thrust = engine_state.get('thrust_approx', 0)
        r_thrust = min(thrust / 100000, 1.0)

        # Componente 2: Eficiencia - mapeo LINEAL decreciente con SFC.
        # Da recompensa 1.0 cuando SFC=0 y 0.0 cuando SFC≥2.0 lbm/(lbf·h),
        # que es aproximadamente el peor caso con afterburner activo.
        sfc = engine_state.get('sfc', 1.0)
        r_sfc = max(0, 1.0 - sfc / 2.0)

        # Componente 3: Potencia eléctrica (proxy via bleed) [0, 1]
        # Más bleed → más aire extraído → más potencia para generadores
        bleed = actuators['bleed_fraction']
        r_power = bleed / 0.10  # Normalizado: bleed_max = 0.10

        # Componente 4: Refrigeración (proxy via tercer flujo + T4) [0, 1]
        # Más bpr_ts (tercer flujo) → mejor refrigeración del avión
        # Menos T4 → menos estrés térmico
        T4 = engine_state.get('T4', 0)
        T4_max = self.SAFETY_LIMITS['T4_max']
        bpr_ts = 0.3 + actuators['vfgv_angle'] / 20.0 * 0.2
        r_cooling = 0.5 * bpr_ts / 0.5 + 0.5 * max(0, 1.0 - T4 / T4_max)

        # Componente 5: Penalización Safe-RL (barrera logarítmica T4)
        T4_warn = self.SAFETY_LIMITS['T4_warning']

        if T4 > T4_max:
            r_safety = -10.0  # Penalización severa
        elif T4 > T4_warn:
            # Barrera logarítmica
            margin = (T4_max - T4) / (T4_max - T4_warn)
            r_safety = np.clip(np.log(max(margin, 1e-6)), -5.0, 0.0)
        else:
            r_safety = 0.0

        # Surge margin proxy penalty: reutiliza la misma aproximación tosca
        # de _check_safety (distancia relativa a velocidad máxima del eje).
        # La penalización crece linealmente al descender por debajo del
        # mínimo, en lugar de aplicar barrera logarítmica como en T4,
        # porque el proxy no es preciso lo suficiente como para justificar
        # una barrera diferenciable.
        Nf = engine_state.get('Nf', 0)
        Nc = engine_state.get('Nc', 0)
        sm_fan = max(0, (self.SAFETY_LIMITS['Nf_max'] - Nf) / self.SAFETY_LIMITS['Nf_max'] * 100)
        sm_hpc = max(0, (self.SAFETY_LIMITS['Nc_max'] - Nc) / self.SAFETY_LIMITS['Nc_max'] * 100)

        if sm_fan < self.SAFETY_LIMITS['SmFan_min']:
            r_safety -= 2.0 * (self.SAFETY_LIMITS['SmFan_min'] - sm_fan) / self.SAFETY_LIMITS['SmFan_min']
        if sm_hpc < self.SAFETY_LIMITS['SmHPC_min']:
            r_safety -= 2.0 * (self.SAFETY_LIMITS['SmHPC_min'] - sm_hpc) / self.SAFETY_LIMITS['SmHPC_min']

        # Recompensa total (4 objetivos + safety)
        reward = (w['thrust'] * r_thrust +
                  w['sfc'] * r_sfc +
                  w['power'] * r_power +
                  w['cooling'] * r_cooling +
                  w['safety'] * r_safety)

        # Penalización adicional si no es seguro
        # Penalización PROPORCIONAL (no termina episodio)
        if not safe:
            exceso_t4 = max(0, T4 - T4_max)
            reward -= (exceso_t4 * 0.1) + 5.0
        else:
            reward += 1.0  # Survival bonus: +1 por step seguro

        return float(reward)


# Registrar entorno en Gymnasium
gym.register(
    id='ACE-v0',
    entry_point='src.environments.ace_env:ACEEnv',
    max_episode_steps=200,
)


if __name__ == '__main__':
    # Test sin PINN (modelo simplificado)
    env = ACEEnv(pinn_model=None, mission_profile='full', max_steps=200)

    obs, info = env.reset()
    print(f"Observación inicial: {obs.shape}")
    print(f"Fase: {info['phase']}, HI: {info['health_index']:.3f}")

    total_reward = 0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}: F={info['thrust']:.0f} lbf, "
                  f"SFC={info['sfc']:.3f}, T4={info['T4']:.0f}°R, "
                  f"Phase={info['phase']}, R={reward:.3f}")

        if terminated or truncated:
            break

    print(f"\nEpisodio completado: {step+1} pasos")
    print(f"  Reward total: {total_reward:.2f}")
    print(f"  Violaciones: {env.episode_violations}")