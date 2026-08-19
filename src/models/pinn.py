"""
Última Fecha de Modificación: 19/Aug/2026
Descripción pinn.py: Gemelo digital del motor de Ciclo Adaptativo (ACE) basado en
una Physics-Informed Neural Network (PINN) del ciclo Brayton de tres flujos.
Sustituye al simulador T-MATS en el bucle de control, que exige inferencia de
milisegundos. La arquitectura combina un backbone residual (LayerNorm + GELU)
con tres constraint layers que calculan directamente las variables deterministas
(farB, P2, SFC) sin predecirlas, dedicando toda la capacidad de la red a las
variables termodinámicas no triviales. 

La pérdida combinada añade a la MSE sobre datos seis residuos PDE del
ciclo Brayton evaluados también sobre puntos de colocación aleatorios
(Raissi et al., 2019) y un currículo progresivo del peso físico
(Wang et al., 2021).

Residuos físicos implementados:
    PDE 1 — Relación isentrópica del compresor:
            T30/T2 = (P30/P2)^((γ-1)/γ), γ = 1.4
    PDE 2 — Balance energético del shaft:
            cp_cold·(T30-T2) ≈ cp_hot·(T4-T50)
    PDE 3 — Ecuación de empuje (momentum de la tobera):
            F ∝ √(2·cp·(T4-T50))
    PDE 4 — Eficiencia de combustión:
            T4 = T30 + η_b·farB·PCI/(cp·(1+farB))
    PDE 5 — Conservación de masa (flujo choked):
            W ∝ P2/√T2
    PDE 6 — Cierre del reparto de masa entre las tres vías:
            f_core + f_bypass + f_TS = 1

Referencias:
    Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019).
        Physics-informed neural networks. J. Comput. Phys., 378, 686-707.
    Wang, S., Teng, Y., Perdikaris, P. (2021). Understanding and
        mitigating gradient pathologies in PINNs. SIAM J. Sci. Comput.,
        43(5), A3055-A3081.
    Cuomo, S. et al. (2022). Scientific Machine Learning through
        Physics-Informed Neural Networks. arXiv:2201.05624.
"""

import logging
import time
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Coeficientes de la pérdida física
# =============================================================================
# Los pesos de las seis pérdidas físicas y de los regularizadores por
# monotonía se calibraron empíricamente mediante grid search grueso
# durante el entrenamiento del checkpoint activo (pinn.pt). El ajuste
# empírico es la práctica estándar en la literatura de PINNs, dado que
# no existen valores óptimos analíticos (Raissi et al., 2019; Wang et
# al., 2021; Cuomo et al., 2022). La priorización refleja la jerarquía
# termodinámica del ciclo Brayton: las relaciones fundamentales
# (isentrópica, balance de energía, combustión) reciben pesos mayores
# que las restricciones cinemáticas y los regularizadores.
PDE_WEIGHTS = {
    'isentropic': 3.0,   # PDE 1: relación termodinámica fundamental
    'energy':     2.0,   # PDE 2: balance de potencia shaft
    'thrust':     1.0,   # PDE 3: momentum de la tobera
    'combustion': 2.0,   # PDE 4: eficiencia del combustor
    'mass':       1.5,   # PDE 5: conservación de masa
    'split':      1.5,   # PDE 6: cierre del reparto de masa entre las tres vías
}

REGULARIZER_WEIGHTS = {
    'monotony':   0.5,   # T2 < T30 < T4 > T50 (compresión, combustión, expansión)
    'pressure':   0.5,   # P30 > P2 (compresión)
    'positivity': 0.3,   # thrust, sfc, Nf, Nc, W_inlet > 0
    'shaft':      0.3,   # Nc > Nf (HP > LP)
    'opr':        0.2,   # OPR = P30/P2 ∈ [3, 50]
}


# =============================================================================
# Constantes termodinámicas
# =============================================================================
GAMMA_COLD = 1.4                          # γ aire frío
GAMMA_HOT = 1.33                          # γ gases calientes
GAMMA_RATIO_COLD = (GAMMA_COLD - 1) / GAMMA_COLD    # 0.2857
GAMMA_RATIO_HOT = (GAMMA_HOT - 1) / GAMMA_HOT       # 0.2481
CP_COLD = 0.24                            # BTU/(lbm·°R) aire frío
CP_HOT = 0.276                            # BTU/(lbm·°R) gases calientes
ETA_BURNER = 0.995                        # eficiencia combustor (calibración del checkpoint activo)
PCI_JP8 = 18400.0                         # BTU/lbm poder calorífico JP-8

# Presupuesto de latencia del lazo de control, en ms. Es el umbral del nivel A de
# DO-178C (fallo de efecto catastrófico), que es el criterio adoptado en la memoria
# y el que aplica el perfilado SWaP-C de scripts/plotting/plot_section6.py.
LATENCY_TARGET_MS = 5.0

class ResidualBlock(nn.Module):
    """Bloque residual con LayerNorm + GELU + dropout, backbone del PINN.

    La conexión residual es necesaria porque la pérdida física obliga a apilar
    varios bloques para capturar las no linealidades del ciclo, y en esa
    profundidad una pila densa simple sufre desvanecimiento del gradiente; la
    normalización previa además estabiliza el entrenamiento cuando los términos
    físicos entran en juego con pesos crecientes.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        """Construye el bloque residual con su pila de capas interna.

        param dim: Dimensión de entrada y salida del bloque, idéntica para que la
            suma residual sea válida sin proyección adicional.
        param dropout: Probabilidad de dropout aplicada dos veces dentro del bloque.
        return: None; deja el módulo listo para su uso en forward().
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica la transformación del bloque y la suma al atajo residual.

        param x: Tensor de activaciones de entrada, de dimensión (batch, dim).
        return: Tensor de la misma forma con el residuo sumado a la entrada.
        """
        return x + self.block(x)


class BraytonPINN(nn.Module):
    """Physics-Informed Neural Network para el ciclo Brayton del ACE.

    Entradas (4): altitude, mach, tra, bpr_ts
    Salidas (14): thrust, sfc, T2, T4, T30, T50, P2, P30, Nf, Nc, farB,
                  W_inlet, BPR, wf

    Tres salidas son deterministas y se calculan mediante constraint
    layers sin ser predichas por la red:

        farB = 0.010 + (tra - 20) / 80 · 0.020    (scheduling FADEC)
        P2   = ISA_total_pressure(altitude, mach) (termodinámica ISA)
        SFC  = wf · 3600 / thrust                 (definición)

    La red predice las once variables termodinámicas restantes y las
    tres deterministas se ensamblan en el forward.

    Calcular esas tres en lugar de aprenderlas es necesario por dos razones: son
    identidades exactas, de modo que aprenderlas solo introduciría error, y libera
    capacidad de la red para las variables realmente no triviales, que son las que
    dominan el error del gemelo digital.
    """

    INPUT_FEATURES = ['altitude', 'mach', 'tra', 'bpr_ts']

    # Salidas predichas por la red (11 variables).
    INTERNAL_FEATURES = ['thrust_approx', 'T2', 'T4', 'T30', 'T50',
                         'P30', 'Nf', 'Nc', 'W_inlet', 'BPR', 'wf']

    # Salidas totales del modelo (14 variables, incluyen las 3 calculadas).
    OUTPUT_FEATURES = ['thrust_approx', 'sfc', 'T2', 'T4', 'T30', 'T50',
                       'P2', 'P30', 'Nf', 'Nc', 'farB',
                       'W_inlet', 'BPR', 'wf']

    def __init__(self, hidden_dim: int = 224, n_layers: int = 8,
                 dropout: float = 0.1):
        """Construye el backbone residual y registra los buffers de normalización.

        Las estadísticas de normalización se registran como buffers, no como
        variables sueltas, para que viajen dentro del state_dict: así un checkpoint
        cargado en otra máquina reproduce exactamente la misma escala de entradas y
        salidas sin necesidad de acompañarlo del corpus de entrenamiento.

        param hidden_dim: Dimensión oculta de los bloques residuales.
        param n_layers: Número de bloques residuales apilados en el backbone.
        param dropout: Probabilidad de dropout en cada bloque residual.
        return: None; deja la red construida e inicializada.

        Nota: los defaults (224/8) coinciden con la arquitectura del checkpoint
        activo del TFG (pinn.pt, 838,443 parámetros). Al cargar un checkpoint
        con arquitectura distinta, el hidden_dim y n_layers deben pasarse
        explícitamente antes de load_state_dict().
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_inputs = len(self.INPUT_FEATURES)
        self.n_internal = len(self.INTERNAL_FEATURES)
        self.n_outputs = len(self.OUTPUT_FEATURES)

        self.input_proj = nn.Sequential(
            nn.Linear(self.n_inputs, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.n_internal),
        )

        self.register_buffer('input_mean', torch.zeros(self.n_inputs))
        self.register_buffer('input_std', torch.ones(self.n_inputs))
        self.register_buffer('internal_mean', torch.zeros(self.n_internal))
        self.register_buffer('internal_std', torch.ones(self.n_internal))

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"BraytonPINN inicializado: {n_params:,} parámetros, "
                    f"{n_layers} bloques residuales, dim = {hidden_dim}")
        logger.info("  Salidas: 11 predichas + 3 computadas por "
                    "constraint layers (farB, P2, SFC)")

    def _init_weights(self):
        """Inicializa con Xavier normal todas las capas lineales de la red.

        Es necesario porque la pérdida física evalúa cocientes y potencias de las
        salidas: una inicialización con varianza mal escalada produce predicciones
        iniciales extremas que hacen explotar los residuos PDE antes de que el
        currículo llegue siquiera a activarlos. Los sesgos se ponen a cero.

        return: None; los pesos del modelo se modifican in situ.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_normalization(self, input_mean: torch.Tensor,
                          input_std: torch.Tensor,
                          internal_mean: torch.Tensor,
                          internal_std: torch.Tensor) -> None:
        """Fija los tensores de normalización a partir de estadísticas del corpus.

        Debe llamarse antes de entrenar o de cargar datos nuevos, porque el forward
        estandariza las entradas y desestandariza las salidas con estos buffers. Las
        desviaciones típicas se acotan por abajo para que una variable constante en
        el corpus no provoque una división por cero.

        param input_mean: Medias de las cuatro variables de entrada.
        param input_std: Desviaciones típicas de las entradas, acotadas a 1e-8.
        param internal_mean: Medias de las once salidas predichas por la red.
        param internal_std: Desviaciones típicas de esas salidas, acotadas a 1e-8.
        return: None; actualiza los buffers registrados del modelo.
        """
        self.input_mean = input_mean
        self.input_std = input_std.clamp(min=1e-8)
        self.internal_mean = internal_mean
        self.internal_std = internal_std.clamp(min=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predice el estado termodinámico completo a partir de la condición de vuelo.

        Normaliza la entrada, la propaga por el backbone residual para obtener las
        once variables aprendidas y ensambla con ellas las tres calculadas por las
        constraint layers. El ensamblado se hace aquí, y no en el llamante, para
        que la pérdida física y el resto del sistema vean siempre el vector de 14
        salidas en el mismo orden, con o sin gradientes.

        param x: Tensor (batch, 4) con altitude, mach, tra y bpr_ts sin normalizar.
        return: Tensor (batch, 14) con las variables de OUTPUT_FEATURES en unidades
            físicas, incluidas thrust y wf acotadas por abajo para estabilidad.
        """
        # Inputs raw para las constraint layers.
        altitude = x[:, 0]
        mach = x[:, 1]
        tra = x[:, 2]

        # Backbone: predice 11 variables internas normalizadas.
        x_norm = (x - self.input_mean) / self.input_std
        h = self.input_proj(x_norm)
        for block in self.residual_blocks:
            h = block(h)
        y_norm = self.output_proj(h)
        y_int = y_norm * self.internal_std + self.internal_mean

        thrust = y_int[:, 0].clamp(min=100)
        wf = y_int[:, 10].clamp(min=0.01)

        # Constraint layer 1: farB (scheduling FADEC).
        farB_computed = 0.010 + (tra - 20) / 80 * 0.020

        # Constraint layer 2: P2 (presión total ISA con corrección de Mach).
        T_amb = 518.67 - 0.00356 * altitude
        P_amb = 14.696 * (T_amb / 518.67).clamp(min=0.01) ** 5.256
        P2_computed = P_amb * (1 + 0.2 * mach ** 2) ** 3.5

        # Constraint layer 3: SFC (definición operativa).
        sfc_computed = (wf * 3600) / thrust

        # Ensamblado final (14 columnas).
        return torch.cat([
            thrust.unsqueeze(1),                # 0  thrust
            sfc_computed.unsqueeze(1),          # 1  SFC
            y_int[:, 1:5],                      # 2-5  T2, T4, T30, T50
            P2_computed.unsqueeze(1),           # 6  P2
            y_int[:, 5:6],                      # 7  P30
            y_int[:, 6:8],                      # 8-9  Nf, Nc
            farB_computed.unsqueeze(1),         # 10 farB
            y_int[:, 8:11],                     # 11-13  W_inlet, BPR, wf
        ], dim=1)

    def predict(self, altitude: float, mach: float, tra: float,
                bpr_ts: float) -> Dict[str, float]:
        """Predice el estado del motor en un único punto operativo, en modo eval.

        Envoltorio de conveniencia sobre forward() que acepta escalares y devuelve
        un diccionario con nombres. Es necesario porque conmuta explícitamente a
        modo evaluación y desactiva el grafo de gradientes: sin ello el dropout
        seguiría activo y cada consulta puntual daría un resultado distinto.

        param altitude: Altitud de vuelo en pies.
        param mach: Número de Mach de vuelo.
        param tra: Throttle Resolver Angle en % (20–100).
        param bpr_ts: Fracción de tercer flujo del splitter.
        return: Diccionario {nombre de OUTPUT_FEATURES: valor} con las 14 variables
            del estado termodinámico predicho.
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor([[altitude, mach, tra, bpr_ts]],
                             dtype=torch.float32)
            y = self.forward(x)
        return {name: y[0, i].item()
                for i, name in enumerate(self.OUTPUT_FEATURES)}


class BraytonPhysicsLoss(nn.Module):
    """Pérdida física con seis residuos PDE del ciclo Brayton.

    Los residuos se calculan como diferencias relativas cuadráticas y
    se combinan con los coeficientes de `PDE_WEIGHTS` y
    `REGULARIZER_WEIGHTS` definidos a nivel de módulo.

    Nota metodológica: los pesos son heurísticos y forman parte del
    hiperparámetro efectivo del entrenamiento. Cambiarlos requiere
    reentrenar el modelo para preservar coherencia con el checkpoint
    activo.

    Es lo que convierte la red en una PINN y no en un simple regresor: penalizar
    la violación de las leyes del ciclo obliga al modelo a extrapolar de forma
    físicamente coherente en las zonas del envelope con pocos datos.
    """

    def forward(self, y_pred: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Evalúa los residuos físicos y los regularizadores sobre una predicción.

        Desempaqueta las 14 salidas, calcula los seis residuos PDE del ciclo Brayton
        y los regularizadores de monotonía, presión, positividad, eje y OPR, y los
        combina con los pesos de módulo. No usa las etiquetas del corpus: es
        precisamente por eso que puede aplicarse también sobre puntos de colocación
        donde no existen datos. Los clamps evitan raíces y divisiones sobre valores
        no físicos durante las primeras épocas.

        param y_pred: Tensor (batch, 14) de salidas del modelo en unidades físicas.
        return: Tupla (pérdida física total escalar, diccionario con el valor
            desagregado de cada residuo y regularizador para el logging).
        """
        # Desempaquetado con clamps de estabilidad numérica.
        thrust = y_pred[:, 0]
        sfc = y_pred[:, 1]
        T2 = y_pred[:, 2].clamp(min=100)
        T4 = y_pred[:, 3].clamp(min=100)
        T30 = y_pred[:, 4].clamp(min=100)
        T50 = y_pred[:, 5].clamp(min=100)
        P2 = y_pred[:, 6].clamp(min=1.0)
        P30 = y_pred[:, 7].clamp(min=1.0)
        Nf = y_pred[:, 8]
        Nc = y_pred[:, 9]
        farB = y_pred[:, 10].clamp(min=1e-4, max=0.08)
        W_inlet = y_pred[:, 11].clamp(min=1.0)
        BPR = y_pred[:, 12].clamp(min=0.01)

        # PDE 1: relación isentrópica del compresor.
        T_ratio = T30 / T2
        P_ratio = P30 / P2
        T_ratio_isentropic = P_ratio.pow(GAMMA_RATIO_COLD)
        residual_isentropic = (T_ratio - T_ratio_isentropic) / T_ratio_isentropic
        loss_isentropic = residual_isentropic.pow(2).mean()

        # PDE 2: balance energético compresor-turbina (shaft power).
        delta_compressor = T30 - T2
        delta_turbine = (T4 - T50).clamp(min=1)
        energy_ratio = delta_compressor / delta_turbine
        target_ratio = CP_COLD / CP_HOT
        residual_energy = (energy_ratio - target_ratio) / target_ratio
        loss_energy = residual_energy.pow(2).mean()

        # PDE 3: momentum de la tobera (proxy del empuje).
        thrust_proxy = (T4 - T50).clamp(min=1).sqrt()
        thrust_norm = thrust / thrust.abs().mean().clamp(min=1)
        proxy_norm = thrust_proxy / thrust_proxy.mean().clamp(min=1e-6)
        loss_thrust = (thrust_norm - proxy_norm).pow(2).mean()

        # PDE 4: eficiencia de combustión (JP-8, η_b = 0.995).
        T4_theoretical = (T30 + ETA_BURNER * farB * PCI_JP8
                          / (CP_COLD * (1 + farB)))
        residual_combustion = (T4 - T4_theoretical) / T4_theoretical.clamp(min=100)
        loss_combustion = residual_combustion.pow(2).mean()

        # PDE 5: conservación de masa (flujo choked, W ∝ P/√T).
        mass_proxy = P2 / T2.sqrt()
        W_norm = W_inlet / W_inlet.mean().clamp(min=1)
        proxy_norm = mass_proxy / mass_proxy.mean().clamp(min=1e-6)
        loss_mass = (W_norm - proxy_norm).pow(2).mean()

        # PDE 6: cierre del reparto de masa entre las tres vías.
        # Las tres fracciones se construyen a partir del BPR predicho y del
        # reparto nominal al tercer flujo, de modo que su suma es la unidad de
        # forma exacta: el término evalúa el cierre y no aporta gradiente. Actúa
        # como comprobación estructural de que la parametrización del reparto es
        # consistente, no como restricción activa sobre la salida de la red.
        REFERENCE_BPR_TS = 0.3
        fraction_core = (1 - REFERENCE_BPR_TS) / (1 + BPR)
        fraction_bypass = (1 - REFERENCE_BPR_TS) * BPR / (1 + BPR)
        fraction_third = REFERENCE_BPR_TS
        total_fractions = fraction_core + fraction_bypass + fraction_third
        loss_split = (total_fractions - 1.0).pow(2).mean()

        # Regularizadores de monotonía y rangos operativos.
        loss_compress = torch.relu(T2 - T30).pow(2).mean()
        loss_combust_ineq = torch.relu(T30 - T4).pow(2).mean()
        loss_expand = torch.relu(T50 - T4).pow(2).mean()
        loss_pressure = torch.relu(P2 - P30).pow(2).mean()
        loss_positivity = (torch.relu(-thrust).pow(2).mean() +
                           torch.relu(-sfc).pow(2).mean() +
                           torch.relu(-Nf).pow(2).mean() +
                           torch.relu(-Nc).pow(2).mean() +
                           torch.relu(-W_inlet).pow(2).mean())
        loss_shaft = torch.relu(Nf - Nc).pow(2).mean()

        opr = P30 / P2
        loss_opr = (torch.relu(3.0 - opr).pow(2).mean() +
                    torch.relu(opr - 50.0).pow(2).mean())

        loss_monotony = loss_compress + loss_combust_ineq + loss_expand

        loss = (PDE_WEIGHTS['isentropic'] * loss_isentropic +
                PDE_WEIGHTS['energy']     * loss_energy +
                PDE_WEIGHTS['thrust']     * loss_thrust +
                PDE_WEIGHTS['combustion'] * loss_combustion +
                PDE_WEIGHTS['mass']       * loss_mass +
                PDE_WEIGHTS['split']      * loss_split +
                REGULARIZER_WEIGHTS['monotony']   * loss_monotony +
                REGULARIZER_WEIGHTS['pressure']   * loss_pressure +
                REGULARIZER_WEIGHTS['positivity'] * loss_positivity +
                REGULARIZER_WEIGHTS['shaft']      * loss_shaft +
                REGULARIZER_WEIGHTS['opr']        * loss_opr)

        details = {
            'pde_isentropic': loss_isentropic.item(),
            'pde_energy':     loss_energy.item(),
            'pde_thrust':     loss_thrust.item(),
            'pde_combustion': loss_combustion.item(),
            'pde_mass':       loss_mass.item(),
            'pde_split':      loss_split.item(),
            'monotony':       loss_monotony.item(),
            'pressure':       loss_pressure.item(),
            'positivity':     loss_positivity.item(),
        }

        return loss, details


class CollocationSampler:
    """Generador de puntos de colocación en el dominio operativo.

    Muestrea uniformemente puntos donde no hay datos de entrenamiento
    para evaluar la pérdida física, permitiendo que las restricciones
    PDE se cumplan globalmente en el dominio y no solo sobre el
    corpus. Este mecanismo distingue una PINN de una red con
    regularización termodinámica post-hoc.
    """
    # Dominio de colocación INTENCIONADAMENTE MÁS AMPLIO que el envelope
    # de entrenamiento (flight_envelope.py: 0-42000 ft, 0-2.0 mach, 20-100 tra,
    # 0-0.5 bpr_ts). Muestrear más allá del corpus fuerza al PINN a ser
    # físicamente coherente en condiciones OOD, que es lo que se evalúa en
    # las pruebas Out-of-Distribution.
    DEFAULT_DOMAIN = {
        'altitude': (0, 50000),
        'mach':     (0.0, 2.0),
        'tra':      (20, 100),
        'bpr_ts':   (0.0, 0.6),
    }

    def __init__(self, n_points: int = 64, domain: dict = None):
        """Configura el número de puntos por batch y los límites del dominio.

        param n_points: Puntos de colocación generados en cada llamada a sample().
        param domain: Diccionario {variable: (mín, máx)} con los límites del
            envelope; si es None se usa DEFAULT_DOMAIN.
        return: None; deja el muestreador listo para sample().
        """
        self.n_points = n_points
        self.domain = domain or self.DEFAULT_DOMAIN

    def sample(self) -> torch.Tensor:
        """Genera un batch de puntos aleatorios uniformes dentro del dominio.

        Se remuestrea en cada iteración en lugar de fijar una rejilla estática para
        que, a lo largo del entrenamiento, la restricción física acabe cubriendo el
        dominio de forma densa sin multiplicar el coste por paso.

        return: Tensor (n_points, 4) con puntos de colocación en unidades físicas,
            en el mismo orden de columnas que espera el modelo.
        """
        points = torch.zeros(self.n_points, 4)
        for i, (_, (lo, hi)) in enumerate(self.domain.items()):
            points[:, i] = torch.rand(self.n_points) * (hi - lo) + lo
        return points


class PINNLoss(nn.Module):
    """Pérdida combinada L = ω₁·L_data + ω₂·L_physics + ω₃·L_colocación.

    Aplica currículo progresivo lineal sobre ω₂ y ω₃ durante los
    primeros `warmup_epochs` para evitar los gradient pathologies
    identificados por Wang et al. (2021) cuando las pérdidas físicas
    entran de golpe con peso alto.
    """

    def __init__(self, omega1_init: float = 1.0,
                 omega2_init: float = 0.01, omega2_final: float = 1.0,
                omega3_init: float = 0.001, omega3_final: float = 0.5,
                warmup_epochs: int = 50):
        """Configura el currículo de pesos y construye los términos de la pérdida.

        Los pesos físicos arrancan en valores muy pequeños porque al inicio del
        entrenamiento las predicciones son arbitrarias y sus residuos PDE dominarían
        por completo el gradiente, impidiendo que la red aprenda siquiera a ajustar
        los datos.

        param omega1_init: Peso fijo del término de ajuste a datos.
        param omega2_init: Peso inicial de la pérdida física sobre el batch de datos.
        param omega2_final: Peso de la pérdida física al terminar el warmup.
        param omega3_init: Peso inicial de la pérdida sobre puntos de colocación.
        param omega3_final: Peso de la pérdida de colocación al terminar el warmup.
        param warmup_epochs: Épocas durante las que ω₂ y ω₃ crecen linealmente.
        return: None; deja el criterio listo para usarse como función de pérdida.
        """
        super().__init__()

        self.omega1 = omega1_init
        self.omega2_init = omega2_init
        self.omega2_final = omega2_final
        self.omega3_init = omega3_init
        self.omega3_final = omega3_final
        self.warmup_epochs = warmup_epochs
        self.omega2 = omega2_init
        self.omega3 = omega3_init

        self.data_loss_fn = nn.MSELoss()
        self.physics_loss_fn = BraytonPhysicsLoss()
        self.collocation_sampler = CollocationSampler(n_points=64)

    def update_weights(self, epoch: int) -> None:
        """Actualiza los pesos ω₂ y ω₃ según el currículo lineal de warmup.

        Debe invocarse una vez al comienzo de cada época. Es lo que materializa la
        mitigación de gradient pathologies de Wang et al. (2021): introducir la
        física de forma gradual en lugar de a peso pleno desde la primera época.
        Pasado el warmup los pesos quedan fijos en su valor final.

        param epoch: Índice de la época actual, empezando en cero.
        return: None; actualiza self.omega2 y self.omega3 in situ.
        """
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            self.omega2 = (self.omega2_init
                           + (self.omega2_final - self.omega2_init) * progress)
            self.omega3 = (self.omega3_init
                           + (self.omega3_final - self.omega3_init) * progress)
        else:
            self.omega2 = self.omega2_final
            self.omega3 = self.omega3_final

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                model: BraytonPINN = None) -> Tuple[torch.Tensor, Dict]:
        """Combina el error sobre datos, la pérdida física y la de colocación.

        El término de colocación exige volver a llamar al modelo sobre puntos sin
        etiqueta, por eso se recibe el propio modelo como argumento opcional: en
        validación se omite y la pérdida se reduce a datos más física sobre el
        batch, que es lo comparable entre épocas.

        param y_pred: Tensor (batch, 14) con las salidas predichas por el modelo.
        param y_true: Tensor (batch, 14) con las salidas de referencia del corpus.
        param model: Modelo con el que evaluar los puntos de colocación; si es None
            ese término vale cero.
        return: Tupla (pérdida total escalar, diccionario con el desglose de cada
            término, los pesos vigentes y los residuos físicos individuales).
        """
        loss_data = self.data_loss_fn(y_pred, y_true)
        loss_physics, physics_details = self.physics_loss_fn(y_pred)

        loss_collocation = torch.tensor(0.0)
        if model is not None and self.omega3 > 0:
            collocation_x = self.collocation_sampler.sample().to(y_pred.device)
            collocation_pred = model(collocation_x)
            loss_collocation, _ = self.physics_loss_fn(collocation_pred)

        loss_total = (self.omega1 * loss_data
                      + self.omega2 * loss_physics
                      + self.omega3 * loss_collocation)

        details = {
            'total':       loss_total.item(),
            'data':        loss_data.item(),
            'physics':     loss_physics.item(),
            'collocation': loss_collocation.item(),
            'omega1':      self.omega1,
            'omega2':      self.omega2,
            'omega3':      self.omega3,
            **{f'phys_{k}': v for k, v in physics_details.items()},
        }

        return loss_total, details


class ACEDigitalTwin:
    """Wrapper de alto nivel para entrenar y evaluar el PINN del ACE.

    Agrupa modelo, criterio, optimizador y planificador junto con el bucle de
    entrenamiento, la validación por variable y la medición de latencia. Es
    necesario para que los scripts de entrenamiento y de evaluación compartan
    exactamente la misma receta: cualquier divergencia entre ellos invalidaría la
    comparación entre el checkpoint activo y las variantes de la ablación.
    """

    DEFAULT_CHECKPOINT_DIR = Path('checkpoints/digital_twin')
    DEFAULT_CHECKPOINT_NAME = 'pinn.pt'

    def __init__(self, hidden_dim: int = 224, n_layers: int = 8,
                 lr: float = 1e-3, device: str = 'auto'):
        """Construye el modelo, el criterio y el optimizador sobre el dispositivo.

        Fija además las semillas de PyTorch y NumPy, requisito para que el TFG pueda
        reproducir los checkpoints publicados. El planificador de tasa de aprendizaje
        es un coseno hasta 200 épocas, alineado con el presupuesto por defecto de fit().

        param hidden_dim: Dimensión oculta de los bloques residuales del PINN.
        param n_layers: Número de bloques residuales apilados.
        param lr: Tasa de aprendizaje inicial del optimizador AdamW.
        param device: 'auto' para elegir CUDA si está disponible, o un dispositivo
            concreto de PyTorch.
        return: None; deja el wrapper listo para prepare_data() y fit().
        """
        
        if device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Semilla fija ANTES de instanciar el modelo, para que la
        # inicialización de pesos (Xavier normal) sea reproducible.
        torch.manual_seed(42)
        np.random.seed(42)

        self.model = BraytonPINN(
            hidden_dim=hidden_dim, n_layers=n_layers).to(self.device)
        self.criterion = PINNLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-6)
        self.train_history = []

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"ACEDigitalTwin inicializado en {self.device}")
        logger.info(f"  Modelo: {n_params:,} parámetros")
        logger.info(f"  Constraint layers: farB, P2, SFC (identidades exactas)")
        logger.info(f"  Pérdida física: 6 PDEs + regularizadores")
        logger.info(f"  Colocación: 64 puntos por batch")

    def prepare_data(self, df, train_ratio: float = 0.8) -> tuple:
        """Prepara los DataLoaders de train y validación y fija la normalización.

        Calcula las estadísticas de entrada y de las variables internas sobre el
        corpus completo, las inyecta en el modelo y reparte las filas en dos
        subconjuntos con una permutación de semilla fija. La partición determinista
        es necesaria para que las métricas de validación de distintas ejecuciones y
        de las variantes ablacionadas se refieran siempre a las mismas muestras.

        param df: Corpus con las columnas de entrada y de salida del PINN.
        param train_ratio: Fracción de filas destinadas a entrenamiento.
        return: Tupla (train_loader, val_loader) con lotes de 32 y 64 muestras
            respectivamente; el de entrenamiento se baraja en cada época.
        """
        X = df[BraytonPINN.INPUT_FEATURES].values.astype(np.float32)

        y_cols = [c for c in BraytonPINN.OUTPUT_FEATURES if c in df.columns]
        y = df[y_cols].values.astype(np.float32)

        internal_cols = [c for c in BraytonPINN.INTERNAL_FEATURES
                         if c in df.columns]
        y_internal = df[internal_cols].values.astype(np.float32)

        X_mean, X_std = X.mean(axis=0), X.std(axis=0)
        int_mean, int_std = y_internal.mean(axis=0), y_internal.std(axis=0)

        self.model.set_normalization(
            torch.tensor(X_mean), torch.tensor(X_std),
            torch.tensor(int_mean), torch.tensor(int_std),
        )

        n_train = int(len(X) * train_ratio)
        indices = np.random.RandomState(42).permutation(len(X))
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        train_ds = torch.utils.data.TensorDataset(
            torch.tensor(X[train_idx]), torch.tensor(y[train_idx]))
        val_ds = torch.utils.data.TensorDataset(
            torch.tensor(X[val_idx]), torch.tensor(y[val_idx]))

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=64, shuffle=False)

        logger.info(f"  Datos: {n_train} train | {len(X) - n_train} val")
        return train_loader, val_loader

    def train_epoch(self, train_loader, epoch: int) -> dict:
        """Entrena el modelo durante una época completa y promedia sus métricas.

        Actualiza primero los pesos del currículo, recorre el DataLoader aplicando
        recorte de norma de gradiente y avanza el planificador al terminar. El
        recorte es necesario porque los residuos físicos pueden producir gradientes
        de magnitud muy dispar entre lotes y desestabilizar el descenso.

        param train_loader: DataLoader con los lotes de entrenamiento.
        param epoch: Índice de la época actual, usado por el currículo de pesos.
        return: Diccionario con el valor medio en la época de cada término de la
            pérdida y de los pesos del currículo.
        """
        self.model.train()
        self.criterion.update_weights(epoch)
        epoch_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss, details = self.criterion(y_pred, y_batch, model=self.model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            epoch_losses.append(details)

        self.scheduler.step()
        return {k: float(np.mean([d[k] for d in epoch_losses]))
                for k in epoch_losses[0]}

    @torch.no_grad()
    def validate(self, val_loader) -> dict:
        """Evalúa el modelo sobre validación y calcula métricas por variable.

        Acumula todas las predicciones antes de medir para que MAPE y RMSE se
        calculen sobre el conjunto completo y no como media de medias por lote. El
        desglose por variable es necesario porque el error del gemelo digital es muy
        desigual entre magnitudes y la memoria del TFG reporta cada una por separado.

        param val_loader: DataLoader con los lotes de validación.
        return: Diccionario con {variable}_mape y {variable}_rmse de las 14 salidas,
            la pérdida total y los residuos físicos agregados.
        """
        self.model.eval()
        all_preds, all_targets = [], []

        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(self.device)
            y_pred = self.model(X_batch)
            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch)

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        metrics = {}
        for i, name in enumerate(BraytonPINN.OUTPUT_FEATURES):
            if i < preds.shape[1]:
                mape = (torch.abs(preds[:, i] - targets[:, i]) /
                        targets[:, i].clamp(min=1e-8)).mean() * 100
                rmse = torch.sqrt(((preds[:, i] - targets[:, i]) ** 2).mean())
                metrics[f'{name}_mape'] = mape.item()
                metrics[f'{name}_rmse'] = rmse.item()

        val_loss, val_details = self.criterion(preds, targets)
        metrics['total'] = val_loss.item()
        for k, v in val_details.items():
            if 'phys_' in k:
                metrics[k] = v

        return metrics

    def fit(self, train_loader, val_loader, epochs: int = 1200,
            patience: int = 200, checkpoint_path: str = None) -> list:
        """Ejecuta el entrenamiento completo con early stopping y guarda el mejor modelo.

        Alterna entrenamiento y validación, registra el historial, guarda el
        checkpoint cada vez que mejora y detiene el bucle tras `patience` épocas sin
        mejora. La métrica de selección es el MAPE medio excluyendo sfc, farB y P2,
        porque esas tres las calculan las constraint layers de forma exacta e
        incluirlas diluiría artificialmente el error de las variables aprendidas.

        param train_loader: DataLoader con los lotes de entrenamiento.
        param val_loader: DataLoader con los lotes de validación.
        param epochs: Número máximo de épocas a ejecutar.
        param patience: Épocas sin mejora del MAPE medio antes de parar.
        param checkpoint_path: Ruta del checkpoint; si es None se usa la ruta por
            defecto de la clase.
        return: Lista de diccionarios con el historial por época, ya volcado a
            training_history_pinn.csv junto al checkpoint; el modelo queda con los
            pesos del mejor época cargados.
        """
        if checkpoint_path is None:
            checkpoint_path = str(
                self.DEFAULT_CHECKPOINT_DIR / self.DEFAULT_CHECKPOINT_NAME)

        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

        best_val_mape = float('inf')
        patience_counter = 0

        logger.info(f"Entrenamiento PINN: {epochs} epochs, "
                    f"patience = {patience}")
        logger.info(f"  Currículo: ω₂ {self.criterion.omega2_init}"
                    f" → {self.criterion.omega2_final}, "
                    f"ω₃ {self.criterion.omega3_init}"
                    f" → {self.criterion.omega3_final} "
                    f"en {self.criterion.warmup_epochs} epochs")

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            self.train_history.append({
                'epoch':             epoch,
                'train_loss':        train_metrics['total'],
                'val_loss':          val_metrics['total'],
                'train_data':        train_metrics['data'],
                'train_physics':     train_metrics['physics'],
                'train_collocation': train_metrics['collocation'],
                'omega2':            train_metrics['omega2'],
                'omega3':            train_metrics['omega3'],
                **{f'val_{k}': v for k, v in val_metrics.items()
                   if 'mape' in k or 'rmse' in k},
            })

            # Métrica de selección: MAPE medio excluyendo variables
            # calculadas exactamente por las constraint layers.
            excluded = {'sfc_mape', 'farB_mape', 'P2_mape'}
            mape_keys = [k for k in val_metrics
                         if k.endswith('_mape') and k not in excluded]
            avg_mape = (np.mean([val_metrics[k] for k in mape_keys])
                        if mape_keys else float('inf'))

            if avg_mape < best_val_mape:
                best_val_mape = avg_mape
                patience_counter = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                thrust_mape = val_metrics.get('thrust_approx_mape', float('nan'))
                sfc_mape = val_metrics.get('sfc_mape', float('nan'))
                pde_isen = val_metrics.get('phys_pde_isentropic', float('nan'))
                logger.info(
                    f"  Epoch {epoch + 1:>3}/{epochs} | "
                    f"Train {train_metrics['total']:.2f} | "
                    f"Val {val_metrics['total']:.2f} | "
                    f"ω₂ {train_metrics['omega2']:.3f} | "
                    f"ω₃ {train_metrics['omega3']:.3f} | "
                    f"Thrust {thrust_mape:.2f}% | "
                    f"SFC {sfc_mape:.2f}% | "
                    f"PDE_isen {pde_isen:.4f}")

            if patience_counter >= patience:
                logger.info(f"  Early stopping en epoch {epoch + 1}")
                break

        history_path = Path(checkpoint_path).parent / 'training_history_pinn.csv'
        pd.DataFrame(self.train_history).to_csv(history_path, index=False)
        logger.info(f"Historial guardado en {history_path}")

        self.model.load_state_dict(
            torch.load(checkpoint_path, weights_only=False))
        logger.info(f"Mejor modelo cargado: MAPE medio = {best_val_mape:.2f}%")

        return self.train_history

    def measure_latency(self, n_warmup: int = 100,
                        n_runs: int = 1000) -> dict:
        """Mide la latencia de inferencia del gemelo digital en milisegundos.

        Ejecuta primero un warmup y después cronometra n_runs inferencias unitarias,
        reportando media, desviación y percentiles. Es necesario porque el PINN solo
        sustituye a T-MATS si cumple el presupuesto temporal del lazo de control: el
        criterio que se verifica es P99 < LATENCY_TARGET_MS (5 ms, nivel A de
        DO-178C), no la media, ya que es la cola de la distribución la que compromete
        el tiempo real.

        param n_warmup: Inferencias previas descartadas, para excluir el coste de
            compilación de kernels y de asignación de memoria.
        param n_runs: Inferencias cronometradas sobre las que se calculan los
            estadísticos.
        return: Diccionario con mean_ms, std_ms, p50_ms, p95_ms, p99_ms, max_ms y el
            booleano meets_latency_target.
        """
        self.model.eval()
        x = torch.randn(1, 4).to(self.device)

        for _ in range(n_warmup):
            with torch.no_grad():
                self.model(x)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                self.model(x)
            times.append((time.perf_counter() - t0) * 1000)

        times_sorted = sorted(times)
        results = {
            'mean_ms': float(np.mean(times)),
            'std_ms':  float(np.std(times)),
            'p50_ms':  times_sorted[n_runs // 2],
            'p95_ms':  times_sorted[int(n_runs * 0.95)],
            'p99_ms':  times_sorted[int(n_runs * 0.99)],
            'max_ms':  max(times),
        }
        results['meets_latency_target'] = results['p99_ms'] < LATENCY_TARGET_MS

        logger.info(f"Latencia PINN ({self.device}):")
        logger.info(f"  Media: {results['mean_ms']:.3f} ms")
        logger.info(f"  P95:   {results['p95_ms']:.3f} ms")
        logger.info(f"  P99:   {results['p99_ms']:.3f} ms")
        logger.info(f"  Cumple objetivo < {LATENCY_TARGET_MS:.0f} ms P99 (DO-178C nivel A): "
                    f"{'sí' if results['meets_latency_target'] else 'no'}")

        return results

    def get_training_summary(self) -> str:
        """Genera un resumen textual legible del entrenamiento realizado.

        Recorre el historial acumulado, localiza la mejor época por MAPE medio y
        lista las métricas por variable de la última. Es necesario para dejar
        constancia del resultado del entrenamiento en el log de la ejecución, sin
        obligar a abrir el CSV de historial.

        return: Cadena multilínea con épocas completadas, mejor época y MAPE por
            variable, o un aviso si todavía no se ha entrenado.
        """
        if not self.train_history:
            return "No hay historial de entrenamiento."

        def avg_mape(entry: dict) -> float:
            """Promedia todas las métricas MAPE de una entrada del historial.

            Sirve como clave de ordenación para localizar la mejor época; devuelve
            infinito cuando la entrada no contiene ninguna métrica MAPE, para que
            esas épocas nunca resulten seleccionadas como mejores.

            param entry: Diccionario de métricas de una época del historial.
            return: MAPE medio de la época, o infinito si no hay métricas MAPE.
            """
            mapes = [v for k, v in entry.items() if 'mape' in k]
            return float(np.mean(mapes)) if mapes else float('inf')

        best = min(self.train_history, key=avg_mape)
        last = self.train_history[-1]

        lines = [
            f"Epochs completadas: {len(self.train_history)}",
            f"Mejor epoch:        {best['epoch'] + 1}",
            f"Mejor MAPE medio:   {avg_mape(best):.2f}%",
            "Métricas por variable (última epoch):",
        ]
        for key, val in last.items():
            if 'mape' in key:
                name = key.replace('val_', '').replace('_mape', '')
                lines.append(f"  {name:<20} MAPE = {val:.2f}%")

        return '\n'.join(lines)