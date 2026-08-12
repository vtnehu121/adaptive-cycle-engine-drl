"""
Última Fecha de Modificación: 08/Aug/2026
Descripción fadec_baseline.py: Controlador FADEC (Full Authority Digital Engine
Control) convencional que sirve de baseline frente a los agentes de aprendizaje 
por refuerzo. Implementa gain scheduling con lookup tables por fase de misión, 
aplicando posiciones fijas de actuadores predefinidas sin adaptar la respuesta 
a la degradación del motor ni a condiciones operativas cambiantes: 
esta rigidez es precisamente la limitación que el agente DRL pretende
superar. El controlador emula la interfaz de Stable-Baselines3
(`predict(obs, deterministic=True)`) para permitir su evaluación en pipelines
compartidos con los agentes SAC/TD3. Incluye ruido gaussiano de actuación que
emula la imprecisión mecánica real y un método de evaluación con métricas
agregadas por fase de misión.

Referencia:
    Jaw, L.C., Mattingly, J.D. (2009). Aircraft Engine Controls:
    Design, System Analysis, and Health Monitoring. AIAA Education
    Series, capítulos 5-6.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Configuración del scheduling

# Ruido gaussiano aditivo sobre la posición nominal del actuador para
# emular la imprecisión mecánica del sistema de actuación real
# (histéresis, holguras, cuantización D/A). El valor 0.10 (10%) refleja
# tolerancias representativas de FADECs comerciales; permite además
# introducir estocasticidad en la evaluación del baseline.
ACTUATOR_NOISE_STD = 0.10

# Codificación entera de las fases de misión, alineada con la variable
# mission_phase_encoded del ACEEnv. OBS_PHASE_INDEX = 17 corresponde a la
# posición 18ª (índice 17) del vector de observación de 19 dimensiones
# que devuelve ACEEnv._get_observation():
#   [T2, T4, T30, T50, P2, P30, Nf, Nc, thrust, sfc,
#    vfgv_angle, vgvt_angle, ab_fraction, bleed_fraction,
#    altitude, mach, tra, mission_phase_encoded, health_index]
PHASE_ENCODING = {0: 'takeoff', 1: 'climb', 2: 'cruise',
                  3: 'combat', 4: 'descent'}

DEFAULT_PHASE = 'cruise'
OBS_PHASE_INDEX = 17


class FADECBaseline:
    """Baseline FADEC con gain scheduling y lookup tables por fase.

    Cada fase de misión mapea a un vector de cuatro actuadores
    normalizados en [-1, 1] compatibles con el espacio de acciones del
    ACEEnv: VFGV (Variable Fan Guide Vanes), VGV-T (Variable Guide
    Vanes de turbina), afterburner y bleed. Las posiciones se han
    calibrado para representar valores subóptimos pero seguros que
    reflejen el comportamiento típico de un FADEC industrial no
    adaptativo, dejando margen de mejora al agente DRL en la métrica
    de reward por fase.
    """

    # Lookup table: posiciones de actuadores por fase (valores en [-1, 1]).
    # Calibradas heurísticamente contra el gemelo digital (PINN) para
    # representar consignas subóptimas pero seguras típicas de un FADEC
    # industrial no adaptativo. Justificación por fase:
    #   - takeoff: AB alto (empuje máximo), bleed bajo (todo el aire al núcleo)
    #   - climb:   AB moderado, VFGV positivo (mayor bypass = mejor SFC)
    #   - cruise:  AB desactivado (economía), configuración de máxima eficiencia
    #   - combat:  AB máximo, VFGV negativo (menos bypass = más empuje)
    #   - descent: AB desactivado, bleed positivo (permite sangrado de aire)
    # El óptimo teórico dependería de la degradación instantánea, que este
    # baseline por diseño ignora.
    SCHEDULE = {
        'takeoff': {'vfgv':  0.1, 'vgvt':  0.1, 'ab':  0.6, 'bleed': -0.5},
        'climb':   {'vfgv':  0.3, 'vgvt':  0.1, 'ab':  0.2, 'bleed': -0.4},
        'cruise':  {'vfgv':  0.4, 'vgvt': -0.1, 'ab': -0.8, 'bleed': -0.2},
        'combat':  {'vfgv': -0.3, 'vgvt':  0.3, 'ab':  0.8, 'bleed': -0.7},
        'descent': {'vfgv':  0.5, 'vgvt': -0.2, 'ab': -0.8, 'bleed':  0.3},
    }

    def __init__(self, actuator_noise: float = ACTUATOR_NOISE_STD,
                 seed: Optional[int] = None):
        """Configura el nivel de ruido de actuación e inicializa el historial.

        El ruido se parametriza en lugar de fijarse porque permite comprobar hasta
        qué punto la ventaja del agente DRL depende de la imprecisión atribuida al
        baseline: con ruido nulo la comparación queda en scheduling puro.

        param actuator_noise: Desviación típica del ruido gaussiano aplicado a la
            posición nominal de cada actuador.
        param seed: Semilla del generador de ruido para reproducibilidad. Si es
            None se usa aleatoriedad no reproducible. En la comparativa oficial
            del TFG se recomienda seed=42 para que las métricas del baseline
            sean idénticas entre ejecuciones.
        return: None; deja el controlador listo para predict() o evaluate().
        """
        self.actuator_noise = actuator_noise
        self.episode_actions = []
        # Generador propio de NumPy para que el ruido de actuación sea
        # reproducible sin interferir con el estado global de np.random.
        self.rng = np.random.default_rng(seed)
        logger.info(
            f"FADECBaseline inicializado: gain scheduling con "
            f"{len(self.SCHEDULE)} fases, ruido de actuadores "
            f"σ = ±{actuator_noise * 100:.0f}%")

    def predict(self, obs: np.ndarray, phase: Optional[str] = None,
                deterministic: bool = True) -> Tuple[np.ndarray, None]:
        """Predice la acción del FADEC para la observación indicada.

        Interfaz compatible con Stable-Baselines3:
            action, _ = model.predict(obs, deterministic=True)

        Consulta la tabla de la fase activa y le suma el ruido de actuación. Respetar
        la firma de Stable-Baselines3 es lo que permite evaluar el FADEC y los
        agentes entrenados con exactamente el mismo bucle de evaluación, condición
        necesaria para que la comparación de la memoria sea limpia. Obsérvese que la
        acción se registra en el historial antes de añadir el ruido, de modo que
        queda constancia de la orden nominal del scheduling.

        param obs: Vector de observación del entorno, de 19 variables.
        param phase: Fase de misión explícita; si es None se infiere de la posición
            OBS_PHASE_INDEX de la observación.
        param deterministic: Ignorado; el FADEC es determinista por construcción y su
            única fuente de aleatoriedad es el ruido de actuadores.
        return: Tupla (vector de cuatro acciones recortado a [-1, 1], None), donde el
            segundo elemento existe solo por compatibilidad con la interfaz SB3.
        """
        if phase is None:
            phase_idx = (int(obs[OBS_PHASE_INDEX])
                         if len(obs) > OBS_PHASE_INDEX else 2)
            phase = PHASE_ENCODING.get(phase_idx, DEFAULT_PHASE)

        schedule = self.SCHEDULE.get(phase, self.SCHEDULE[DEFAULT_PHASE])

        action = np.array([
            schedule['vfgv'],
            schedule['vgvt'],
            schedule['ab'],
            schedule['bleed'],
        ], dtype=np.float32)

        self.episode_actions.append(action.copy())

        # Ruido gaussiano por imprecisión mecánica del sistema de actuación.
        # Usa self.rng para reproducibilidad opcional (fijable en __init__).
        action = action + self.rng.normal(0, self.actuator_noise, size=4)
        return np.clip(action, -1.0, 1.0), None

    def evaluate(self, env, n_episodes: int = 50) -> Dict:
        """Evalúa el FADEC sobre `n_episodes` en el entorno indicado.

        Devuelve reward medio, violaciones por episodio, tasa de
        seguridad y métricas agregadas por fase.

        Es necesario porque la comparación con el DRL no puede reducirse al reward
        total: el baseline puede perder en recompensa y aun así ser preferible si
        viola menos restricciones, o ganar en una fase y perder en otra. El desglose
        por fase es lo que permite localizar dónde aporta realmente el agente
        aprendido.

        param env: Entorno ACEEnv sobre el que ejecutar los episodios.
        param n_episodes: Número de episodios completos a evaluar.
        return: Diccionario con reward medio y su desviación, violaciones medias por
            episodio, tasa de seguridad y métricas de empuje, SFC y reward por fase.
        """
        rewards = []
        violations_total = 0
        total_steps = 0  # Contador de pasos REALES para safety_rate correcto
        phase_metrics = {p: {'thrust': [], 'sfc': [], 'reward': []}
                         for p in self.SCHEDULE}

        for _ in range(n_episodes):
            obs, info = env.reset()
            ep_reward = 0.0
            done = False

            while not done:
                phase = info.get('phase', DEFAULT_PHASE)
                action, _ = self.predict(obs, phase=phase)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated
                total_steps += 1

                p = info.get('phase', DEFAULT_PHASE)
                phase_metrics[p]['thrust'].append(info.get('thrust', 0))
                phase_metrics[p]['sfc'].append(info.get('sfc', 0))
                phase_metrics[p]['reward'].append(reward)

            rewards.append(ep_reward)
            violations_total += info.get('episode_violations', 0)

        phase_summary = {
            p: {
                'mean_thrust': float(np.mean(m['thrust'])),
                'mean_sfc':    float(np.mean(m['sfc'])),
                'mean_reward': float(np.mean(m['reward'])),
            }
            for p, m in phase_metrics.items() if m['thrust']
        }

        results = {
            'controller':      'FADEC',
            'n_episodes':      n_episodes,
            'mean_reward':     float(np.mean(rewards)),
            'std_reward':      float(np.std(rewards)),
            'mean_violations': violations_total / n_episodes,
            'safety_rate':     1 - violations_total / max(total_steps, 1),           
            'phase_metrics':   phase_summary,
        }

        logger.info(
            f"FADECBaseline evaluado sobre {n_episodes} episodios:")
        logger.info(f"  Reward:     {results['mean_reward']:.2f} "
                    f"± {results['std_reward']:.2f}")
        logger.info(f"  Violaciones/ep: {results['mean_violations']:.1f}")
        logger.info(f"  Tasa seguridad: {results['safety_rate']:.1%}")
        for p, m in phase_summary.items():
            logger.info(f"  {p:>10}: thrust = {m['mean_thrust']:.0f} lbf, "
                        f"SFC = {m['mean_sfc']:.3f}")

        return results

    def reset(self) -> None:
        """Vacía el historial de acciones acumulado durante el episodio.

        Debe llamarse entre episodios cuando se quiera analizar la secuencia de
        órdenes de un episodio concreto; sin ello el historial crece indefinidamente
        y mezcla acciones de ejecuciones distintas.

        return: None; deja episode_actions como lista vacía.
        """
        self.episode_actions = []