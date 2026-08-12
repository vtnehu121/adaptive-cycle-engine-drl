"""
Última Fecha de Modificación: 08/Aug/2026
Descripción pinn_ablated.py: Variantes ablacionadas del BraytonPINN empleadas en el
estudio empírico de la Sección 6 del TFG. Define la arquitectura sin constraint
layers (BraytonPINNNoCL), que predice directamente las 14 salidas, y la pérdida
solo-datos (PINNLossDataOnly), que anula los términos físico y de colocación. Ambas
piezas permiten cuantificar por separado cuánto aporta cada ingrediente del gemelo
digital manteniendo idéntico el resto del montaje experimental.

Se implementan dos configuraciones ablated frente al modelo completo:

    Configuración B — Sin Constraint Layers:
        Implementada como `BraytonPINNNoCL` en este módulo. La arquitectura
        predice directamente las 14 variables de salida, incluyendo las
        tres que en el modelo completo se obtenían por identidades exactas
        (farB, P2, SFC). Evalúa la contribución de los constraint layers
        a la fidelidad del gemelo digital.

    Configuración C — Sin PDEs:
        Reutiliza la arquitectura completa `BraytonPINN` (definida en
        pinn.py, con sus constraint layers) pero se entrena con
        `PINNLossDataOnly` (ω₂ = ω₃ = 0, pérdida física desactivada). En
        train_pinn_ablation.py se le denomina "BraytonPINNNoPhysics" como
        etiqueta conceptual del checkpoint pinn_no_physics.pt. Evalúa la
        contribución del término físico a la fidelidad del gemelo digital.

Ambos modelos se entrenan sobre el mismo corpus, con la misma
configuración de optimización y con los mismos splits reproducibles
que el modelo completo, garantizando un ablation justo.
"""

import logging
from typing import Dict

import torch
import torch.nn as nn

from src.models.pinn import (
    BraytonPINN,
    ResidualBlock,
)

logger = logging.getLogger(__name__)


class BraytonPINNNoCL(nn.Module):
    """Variante ablated del BraytonPINN sin constraint layers.

    La red predice directamente las 14 variables de salida
    (`OUTPUT_FEATURES`). La arquitectura del backbone (input
    projection, bloques residuales, output projection) permanece
    idéntica al modelo completo para que el ablation aísle
    exclusivamente el efecto de las constraint layers.

    Notas de diseño:
        - El `output_proj` termina en 14 neuronas en lugar de 11.
        - Los buffers de normalización interna cubren las 14 variables
          (no solo las 11 predichas del modelo completo).
    """

    INPUT_FEATURES = BraytonPINN.INPUT_FEATURES
    OUTPUT_FEATURES = BraytonPINN.OUTPUT_FEATURES

    def __init__(self, hidden_dim: int = 224, n_layers: int = 8,
                 dropout: float = 0.1):
        """Construye el backbone con una cabeza de 14 salidas en lugar de 11.

        Reutiliza los mismos hiperparámetros por defecto que el modelo completo
        (BraytonPINN 224/8 = 838,443 params, esta variante = 838,818 params por
        las 3 neuronas extra en output_proj). Cualquier diferencia de capacidad
        haría que el ablation midiese el tamaño de la red y no la ausencia de
        las constraint layers.

        param hidden_dim: Dimensión oculta de los bloques residuales.
        param n_layers: Número de bloques residuales apilados.
        param dropout: Probabilidad de dropout en cada bloque residual.
        return: None; deja la variante construida e inicializada con Xavier.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_inputs = len(self.INPUT_FEATURES)
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
            nn.Linear(hidden_dim // 2, self.n_outputs),
        )

        self.register_buffer('input_mean', torch.zeros(self.n_inputs))
        self.register_buffer('input_std', torch.ones(self.n_inputs))
        self.register_buffer('output_mean', torch.zeros(self.n_outputs))
        self.register_buffer('output_std', torch.ones(self.n_outputs))

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"BraytonPINNNoCL inicializado: {n_params:,} "
                    f"parámetros, arquitectura sin constraint layers")

    def _init_weights(self) -> None:
        """Inicializa con Xavier normal las capas lineales de la variante.

        Replica exactamente el esquema de inicialización del modelo completo, para
        que las dos configuraciones del ablation partan del mismo régimen de
        varianza y la comparación no quede contaminada por el arranque.

        return: None; los pesos del modelo se modifican in situ.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_normalization(self, input_mean: torch.Tensor,
                          input_std: torch.Tensor,
                          output_mean: torch.Tensor,
                          output_std: torch.Tensor) -> None:
        """Fija los tensores de normalización a partir de estadísticas del corpus.

        A diferencia del modelo completo, aquí las estadísticas de salida cubren las
        14 variables, porque ninguna se calcula por identidad: todas atraviesan el
        camino aprendido y necesitan su propia escala.

        param input_mean: Medias de las cuatro variables de entrada.
        param input_std: Desviaciones típicas de las entradas, acotadas a 1e-8.
        param output_mean: Medias de las 14 salidas del modelo.
        param output_std: Desviaciones típicas de las 14 salidas, acotadas a 1e-8.
        return: None; actualiza los buffers registrados del modelo.
        """
        self.input_mean = input_mean
        self.input_std = input_std.clamp(min=1e-8)
        self.output_mean = output_mean
        self.output_std = output_std.clamp(min=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predice las 14 salidas directamente, sin ensamblar identidades exactas.

        El contraste con el forward del modelo completo es justamente el objeto del
        ablation: aquí farB, P2 y SFC salen de la red y por tanto arrastran error de
        aproximación, en lugar de calcularse de forma cerrada.

        param x: Tensor (batch, 4) con altitude, mach, tra y bpr_ts sin normalizar.
        return: Tensor (batch, 14) con las variables de OUTPUT_FEATURES ya
            desnormalizadas a unidades físicas.
        """
        x_norm = (x - self.input_mean) / self.input_std
        h = self.input_proj(x_norm)
        for block in self.residual_blocks:
            h = block(h)
        y_norm = self.output_proj(h)
        return y_norm * self.output_std + self.output_mean

    def predict(self, altitude: float, mach: float, tra: float,
                bpr_ts: float) -> Dict[str, float]:
        """Predice el estado del motor en un único punto operativo, en modo eval.

        Mantiene la misma firma y el mismo formato de salida que el predict() del
        modelo completo, condición necesaria para que los scripts de evaluación
        puedan recorrer ambas variantes con el mismo código.

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


class PINNLossDataOnly(nn.Module):
    """Pérdida ablated que desactiva el término físico y de colocación.

    Se usa junto con la arquitectura completa `BraytonPINN` (definida en
    pinn.py) para la configuración C del ablation, entrenando con únicamente
    L_data (MSE contra el corpus): se preserva la arquitectura del modelo
    completo pero se elimina la información física del gradiente. El
    argumento `model` se acepta para mantener la misma firma que `PINNLoss`,
    aunque no se utilice.

    Los atributos `omega2_init`, `omega3_init`, `omega2_final`,
    `omega3_final` y `warmup_epochs` se mantienen a cero para preservar
    la compatibilidad con `ACEDigitalTwin.fit()`, que los consulta para
    el logging del currículo.
    """

    def __init__(self):
        """Construye la pérdida solo-datos y neutraliza los pesos del currículo.

        Fijar los omegas a cero en lugar de eliminarlos es lo que permite reutilizar
        ACEDigitalTwin.fit() sin tocarlo: el bucle de entrenamiento sigue leyendo los
        mismos atributos, pero el término físico nunca contribuye al gradiente.

        return: None; deja el criterio listo para usarse como función de pérdida.
        """
        super().__init__()
        self.data_loss_fn = nn.MSELoss()

        # Compatibilidad con ACEDigitalTwin.fit(): estos atributos se
        # consultan en el bloque de logging del currículo. Al ser
        # constantemente cero, el currículo queda desactivado sin
        # necesidad de modificar el ACEDigitalTwin.
        self.omega1 = 1.0
        self.omega2 = 0.0
        self.omega3 = 0.0
        self.omega2_init = 0.0
        self.omega3_init = 0.0
        self.omega2_final = 0.0
        self.omega3_final = 0.0
        self.warmup_epochs = 0

    def update_weights(self, epoch: int) -> None:
        """No hace nada: sin física no hay currículo de pesos que actualizar.

        Existe únicamente para respetar la interfaz de PINNLoss, que el bucle de
        entrenamiento invoca al principio de cada época; sin este método la variante
        ablacionada no sería intercambiable con la completa.

        param epoch: Índice de la época actual, ignorado en esta variante.
        return: None siempre; los pesos físicos permanecen a cero.
        """
        return None

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                model=None):
        """Calcula únicamente el error cuadrático medio contra el corpus.

        Devuelve el mismo diccionario de detalle que PINNLoss, con ceros en los
        términos físico y de colocación, para que el historial de entrenamiento y
        los CSV resultantes tengan idéntico esquema de columnas en las dos ramas del
        ablation y puedan compararse sin postprocesado.

        param y_pred: Tensor (batch, 14) con las salidas predichas por el modelo.
        param y_true: Tensor (batch, 14) con las salidas de referencia del corpus.
        param model: Aceptado por compatibilidad de firma con PINNLoss; no se usa.
        return: Tupla (MSE sobre datos, diccionario de detalle con physics y
            collocation a cero).
        """
        loss_data = self.data_loss_fn(y_pred, y_true)
        details = {
            'total':       loss_data.item(),
            'data':        loss_data.item(),
            'physics':     0.0,
            'collocation': 0.0,
            'omega1':      1.0,
            'omega2':      0.0,
            'omega3':      0.0,
        }
        return loss_data, details