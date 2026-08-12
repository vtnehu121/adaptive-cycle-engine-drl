"""
Última Fecha de Modificación: 08/Aug/2026
Descripción safe_rl.py: Capa de seguridad Safe-RL para el control del motor ACE.
Define SafetyLayer, que convierte las restricciones termomecánicas duras en 
penalizaciones de barrera logarítmica continuas y diferenciables sobre 
T₄, Nf y Nc, y ActionFilter, que limita la velocidad de cambio de los actuadores
antes de ejecutar la acción. Ambos mecanismos empujan la política lejos de la 
región insegura sin recurrir a discontinuidades que el optimizador no podría aprovechar.

Formulación:

Formulación:
    La barrera logarítmica transforma un hard constraint (h(x) ≥ 0) en
    un soft penalty continuo y diferenciable:

        B(x) = -log(h(x))       si h(x) > 0
        B(x) = -PENALTY_HARD    si h(x) ≤ 0

    Se sustituye el +∞ teórico por una penalización finita grande
    (PENALTY_HARD = 10.0, más -50.0 adicional al detectar violación)
    para evitar que un solo paso inseguro invalide todo el episodio
    en el aprendizaje. Esto permite que el gradiente del optimizador
    "empuje" la política lejos de las regiones inseguras de forma suave.

Restricciones implementadas actualmente:
    1. T₄ < T₄_max (temperatura máxima del combustor)
    2. Nf < Nf_max (velocidad máxima del fan)
    3. Nc < Nc_max (velocidad máxima del core)
    4. thrust ≥ 0 (empuje no negativo)

Referencias externas verificadas:
    Ames, A. D., Xu, X., Grizzle, J. W., & Tabuada, P. (2017).
        "Control barrier function based quadratic programs for safety
        critical systems." IEEE Trans. Autom. Control, 62(8), 3861-3876.
"""

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SafetyLayer:
    """
    Capa de seguridad basada en barreras logarítmicas.

    Evalúa el estado del motor y genera penalizaciones que se suman
    a la función de recompensa del agente DRL. Las penalizaciones
    crecen exponencialmente a medida que el estado se acerca a los
    límites de seguridad, forzando al agente a aprender políticas
    que mantienen márgenes de operación seguros.

    Es necesaria porque una penalización escalón, activada solo al cruzar el límite,
    no ofrece gradiente alguno mientras el estado se aproxima al peligro: el agente
    únicamente aprendería después de haber violado la restricción. La barrera hace
    que el coste crezca ya dentro de la zona de advertencia, cuando todavía hay
    margen de maniobra para corregir.
    """

    def __init__(self,
                 T4_max: float = 3200,
                 T4_warning: float = 3000,
                 Nf_max: float = 6000,
                 Nc_max: float = 12000,
                 barrier_weight: float = 1.0):
        """Fija los límites de seguridad y los contadores de violaciones.

        Los umbrales de advertencia se separan de los límites duros porque son ellos
        los que determinan la anchura de la zona en la que actúa la barrera: cuanto
        más pronto empiece a penalizar, más conservadora resultará la política.

        param T4_max: Temperatura máxima admisible del combustor, en °R.
        param T4_warning: Umbral de T4 a partir del cual empieza a penalizarse.
        param Nf_max: Velocidad máxima del eje de baja, en rpm.
        param Nc_max: Velocidad máxima del eje de alta, en rpm.
        param barrier_weight: Peso global aplicado a la penalización resultante.
        return: None; deja la capa lista con los contadores a cero.
        """
        self.limits = {
            'T4_max': T4_max,
            'T4_warning': T4_warning,
            'Nf_max': Nf_max,
            'Nc_max': Nc_max,
        }
        self.barrier_weight = barrier_weight
        self.violation_count = 0
        self.total_evaluations = 0

        logger.info(f"SafetyLayer inicializada: T4_max={T4_max}°R, "
                    f"Nf_max={Nf_max} rpm, Nc_max={Nc_max} rpm")

    def evaluate(self, state: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Evalúa el estado del motor y acumula la penalización de seguridad.

        Aplica una barrera superior a T4, Nf y Nc, castiga el empuje negativo y añade
        una penalización severa si alguna restricción llega a violarse. Devolver el
        desglose junto con el total es necesario para poder atribuir después qué
        restricción concreta está limitando la política, en lugar de observar solo un
        número agregado. El método lleva además la cuenta de violaciones del episodio.

        param state: Estado del motor, con al menos las claves T4, Nf, Nc y
            thrust_approx.
        return: Tupla (penalización total ya ponderada por barrier_weight, siempre
            menor o igual que cero; diccionario con el detalle por restricción, el
            margen de T4, el indicador de violación y la tasa acumulada).
        """
        self.total_evaluations += 1
        details = {}
        total_penalty = 0.0

        # 1. Barrera T₄ (temperatura combustor)
        T4 = state.get('T4', 0)
        p_t4, d_t4 = self._barrier_upper(
            value=T4,
            limit=self.limits['T4_max'],
            warning=self.limits['T4_warning'],
            name='T4'
        )
        total_penalty += p_t4
        details['T4_penalty'] = p_t4
        details['T4_margin'] = d_t4

        # 2. Barrera Nf (velocidad fan)
        Nf = state.get('Nf', 0)
        p_nf, d_nf = self._barrier_upper(
            value=Nf,
            limit=self.limits['Nf_max'],
            warning=self.limits['Nf_max'] * 0.9,
            name='Nf'
        )
        total_penalty += p_nf
        details['Nf_penalty'] = p_nf

        # 3. Barrera Nc (velocidad core)
        Nc = state.get('Nc', 0)
        p_nc, d_nc = self._barrier_upper(
            value=Nc,
            limit=self.limits['Nc_max'],
            warning=self.limits['Nc_max'] * 0.9,
            name='Nc'
        )
        total_penalty += p_nc
        details['Nc_penalty'] = p_nc

        # 4. Variables positivas (thrust, SFC)
        thrust = state.get('thrust_approx', 0)
        if thrust < 0:
            total_penalty -= 5.0
            details['thrust_negative'] = True

        # Verificar violación
        violated = (T4 > self.limits['T4_max'] or
                    Nf > self.limits['Nf_max'] or
                    Nc > self.limits['Nc_max'])

        if violated:
            self.violation_count += 1
            total_penalty -= 50.0  # Penalización severa

        details['total_penalty'] = total_penalty * self.barrier_weight
        details['violated'] = violated
        details['violation_rate'] = self.violation_count / max(self.total_evaluations, 1)

        return total_penalty * self.barrier_weight, details

    def _barrier_upper(self, value: float, limit: float,
                       warning: float, name: str) -> Tuple[float, float]:
        """
        Barrera logarítmica para restricción superior: value < limit.

        B(x) = -log((limit - x) / (limit - warning))  si x > warning
        B(x) = 0                                        si x ≤ warning

        Concentra en un único sitio la forma de la barrera, de modo que T4, Nf y Nc
        compartan exactamente el mismo perfil de penalización y sus contribuciones
        sean comparables entre sí. Una vez superado el límite se devuelve un valor
        finito y no infinito, porque un infinito propagado a la recompensa haría
        inutilizable el episodio completo para el aprendizaje.

        param value: Valor actual de la variable evaluada.
        param limit: Límite superior duro que no debe alcanzarse.
        param warning: Umbral a partir del cual comienza la penalización.
        param name: Nombre de la variable, empleado para trazas.
        return: Tupla (penalización menor o igual que cero, margen normalizado en
            [0, 1] donde 1 indica operación holgada y 0 el límite alcanzado).
        """
        if value >= limit:
            return -10.0, 0.0

        margin = (limit - value) / (limit - warning) if limit != warning else 1.0

        if value > warning:
            penalty = -np.log(max(margin, 1e-8))
            return -penalty, max(margin, 0)
        else:
            return 0.0, 1.0

    def get_statistics(self) -> Dict[str, float]:
        """Devuelve las estadísticas de seguridad acumuladas desde el último reset.

        La tasa de violaciones es la métrica que se contrasta con el baseline FADEC en
        la memoria: sin ella el reward medio por sí solo no dice si el agente está
        ganando rendimiento a costa de operar más cerca del límite.

        return: Diccionario con total_evaluations, violations, violation_rate y
            safety_rate; los cocientes se protegen frente a cero evaluaciones.
        """
        return {
            'total_evaluations': self.total_evaluations,
            'violations': self.violation_count,
            'violation_rate': self.violation_count / max(self.total_evaluations, 1),
            'safety_rate': 1 - self.violation_count / max(self.total_evaluations, 1),
        }

    def reset(self):
        """Pone a cero los contadores de evaluaciones y violaciones.

        Debe invocarse al comenzar cada episodio para que la tasa de violaciones se
        refiera a ese episodio; de lo contrario los contadores se acumulan y la tasa
        queda promediada sobre toda la ejecución, ocultando episodios malos.

        return: None; deja violation_count y total_evaluations a cero.
        """
        self.violation_count = 0
        self.total_evaluations = 0


class ActionFilter:
    """
    Filtro de acciones pre-ejecución.

    Proyecta acciones inseguras hacia la región segura más cercana
    del espacio de control. Implementación simplificada de CBF
    (Control Barrier Functions) sin resolver QP completo.

    La formulación formal como proyección ortogonal mediante QP se
    desarrolla en el el apartado de safe-RL de la memoria.

    Es necesario porque la política puede pedir saltos instantáneos entre extremos
    del rango de un actuador, algo que ninguna geometría variable real puede seguir:
    sin este filtro, las trayectorias aprendidas serían físicamente irrealizables.
    """

    def __init__(self, vfgv_rate_max: float = 2.0, vgvt_rate_max: float = 1.5):
        """Fija las velocidades máximas de actuación e inicializa el estado previo.

        Guardar la acción anterior es imprescindible: el límite se aplica sobre el
        incremento respecto al paso previo, de modo que el filtro tiene memoria y no
        puede evaluarse mirando solo la acción actual.

        param vfgv_rate_max: Variación máxima del VFGV por paso, en grados.
        param vgvt_rate_max: Variación máxima del VGV-T por paso, en grados.
        return: None; deja el filtro con la acción previa a cero.
        """
        self.vfgv_rate_max = vfgv_rate_max
        self.vgvt_rate_max = vgvt_rate_max
        self.prev_action = np.zeros(4)

    def filter(self, action: np.ndarray) -> np.ndarray:
        """
        Filtra acción para respetar límites de rate.

        Limita la velocidad de cambio de los actuadores para simular
        la inercia mecánica real del sistema de actuación.
        G(s) = 1/(τs + 1) discretizado como rate limiter.

        Solo se limitan las geometrías variables VFGV y VGV-T, que son las que tienen
        inercia mecánica apreciable; el afterburner y el sangrado responden con
        rapidez suficiente como para no necesitar restricción de velocidad.

        param action: Vector de cuatro acciones normalizadas propuesto por la política.
        return: Vector filtrado del mismo tamaño, recortado a [-1, 1] y compatible con
            la velocidad máxima de cada actuador.
        """
        filtered = action.copy()

        # Rate limiting en VFGV (acción[0])
        delta_vfgv = filtered[0] - self.prev_action[0]
        max_delta_vfgv = self.vfgv_rate_max / 20.0  # Normalizado
        filtered[0] = self.prev_action[0] + np.clip(
            delta_vfgv, -max_delta_vfgv, max_delta_vfgv)

        # Rate limiting en VGV-T (acción[1])
        delta_vgvt = filtered[1] - self.prev_action[1]
        max_delta_vgvt = self.vgvt_rate_max / 15.0
        filtered[1] = self.prev_action[1] + np.clip(
            delta_vgvt, -max_delta_vgvt, max_delta_vgvt)

        # Clamp general
        filtered = np.clip(filtered, -1.0, 1.0)

        self.prev_action = filtered.copy()
        return filtered

    def reset(self):
        """Devuelve el filtro a su estado inicial, con los actuadores centrados.

        Necesario entre episodios: si se conservase la última acción del episodio
        anterior, el primer paso del nuevo arrancaría desde una posición de actuador
        que no se corresponde con el estado real del motor recién reiniciado.

        return: None; deja prev_action como vector de ceros.
        """
        self.prev_action = np.zeros(4)


if __name__ == '__main__':
    # Test de la capa de seguridad
    safety = SafetyLayer()

    # Estado seguro
    state_safe = {'T4': 2500, 'Nf': 4000, 'Nc': 8000, 'thrust_approx': 20000}
    penalty, details = safety.evaluate(state_safe)
    print(f"Estado seguro:   penalty={penalty:.3f}, violated={details['violated']}")

    # Estado en zona de advertencia
    state_warn = {'T4': 3100, 'Nf': 4000, 'Nc': 8000, 'thrust_approx': 20000}
    penalty, details = safety.evaluate(state_warn)
    print(f"Zona advertencia: penalty={penalty:.3f}, T4_margin={details['T4_margin']:.3f}")

    # Estado peligroso
    state_danger = {'T4': 3250, 'Nf': 4000, 'Nc': 8000, 'thrust_approx': 20000}
    penalty, details = safety.evaluate(state_danger)
    print(f"Estado peligroso: penalty={penalty:.3f}, violated={details['violated']}")

    # Estadísticas
    stats = safety.get_statistics()
    print(f"\nEstadísticas: {stats}")

    # Test del filtro de acciones
    print("\n--- Action Filter ---")
    af = ActionFilter()
    for i in range(5):
        raw = np.array([1.0, -1.0, 0.5, 0.0])
        filtered = af.filter(raw)
        print(f"  Step {i}: raw={raw[:2]} → filtered={filtered[:2]}")