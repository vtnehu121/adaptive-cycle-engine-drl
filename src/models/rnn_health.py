"""
Última Fecha de Modificación: 08/Aug/2026
Descripción rnn_health.py: Módulo de monitorización de salud del motor.
Define la arquitectura recurrente HealthMonitoringRNN
(LSTM o GRU) con doble cabezal para predicción de Remaining Useful Life (RUL) y
vector de degradación por componente, el Dataset de ventanas deslizantes sobre
C-MAPSS (Saxena et al., 2008) con sus pseudo-etiquetas y el wrapper HealthMonitor
que entrena, valida y cuantifica la incertidumbre epistémica mediante MC-Dropout
(Gal & Ghahramani, 2016). Aporta al agente de control la estimación de estado de
salud sobre la que adapta su política. La pérdida dual es:

    L = L_RUL + α · L_degradación

Los pseudo-labels de degradación se calculan como delta porcentual de
cuatro sensores (Nf, T30, T50, Nc) respecto a la media de los primeros
`baseline_cycles` ciclos de cada motor:

    fan_eff_proxy = (Nf_actual - Nf_baseline) / |Nf_baseline|
    hpc_eff_proxy = (T30_actual - T30_baseline) / |T30_baseline|
    hpt_eff_proxy = (T50_actual - T50_baseline) / |T50_baseline|
    lpt_eff_proxy = (Nc_actual - Nc_baseline) / |Nc_baseline|
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class HealthMonitoringRNN(nn.Module):
    """Red recurrente con cabezal dual RUL + vector de degradación.

    El cabezal doble es necesario porque el control adaptativo no se conforma con
    saber cuánta vida le queda al motor: necesita saber qué componente se está
    degradando para reasignar los actuadores en consecuencia. Compartir el mismo
    codificador recurrente entre ambas tareas actúa además como regularizador, ya
    que la representación aprendida debe explicar las dos salidas a la vez.
    """

    # 15 sensores informativos, idénticos a CMAPSSLoader.INFORMATIVE_SENSORS.
    # Los 6 sensores restantes de los 21 originales de Saxena 2008 se excluyen
    # por ser constantes en FD001 (T2, P2, P15), fuel-air ratio del combustor
    # (farB) o comandos del controlador (Nf_dmd, PCNfR_dmd). Ver
    # CMAPSSLoader.CONSTANT_SENSORS para la lista completa.
    SENSOR_FEATURES = [
        'T24', 'T30', 'T50', 'P30', 'Nf', 'Nc',
        'epr', 'Ps30', 'phi', 'NRf', 'NRc',
        'BPR', 'htBleed', 'W31', 'W32',
    ]

    # Sensores proxy de degradación por componente y sus etiquetas.
    # El orden es CRÍTICO: debe coincidir con las 4 columnas de salida
    # del cabezal degradation_head y con el orden usado en
    # compute_pseudo_labels() del Dataset.
    DEGRADATION_SENSORS = ['Nf', 'T30', 'T50', 'Nc']
    DEGRADATION_LABELS = ['fan_eff', 'hpc_eff', 'hpt_eff', 'lpt_eff']

    def __init__(self, n_sensors: int = 15, hidden_dim: int = 64,
                 n_layers: int = 2, rnn_type: str = 'lstm',
                 dropout: float = 0.2, bidirectional: bool = False):
        """Construye la proyección de entrada, la RNN y los dos cabezales.

        El cabezal de RUL termina en ReLU para impedir vidas remanentes negativas y
        el de degradación en Tanh porque los deltas relativos están acotados en
        [-1, 1]. El dropout no es solo regularización: es el mecanismo que después
        explota predict_with_uncertainty() para estimar incertidumbre epistémica.

        param n_sensors: Número de sensores de entrada por paso temporal.
        param hidden_dim: Dimensión oculta de la RNN y de los cabezales MLP.
        param n_layers: Número de capas recurrentes apiladas.
        param rnn_type: 'lstm' o 'gru', variante recurrente utilizada.
        param dropout: Probabilidad de dropout, activa también en MC-Dropout.
        param bidirectional: Si es True la RNN es bidireccional y duplica el
            tamaño del estado que reciben los cabezales.
        return: None; deja la red construida y registra su número de parámetros.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.n_directions = 2 if bidirectional else 1

        self.input_proj = nn.Sequential(
            nn.Linear(n_sensors, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        rnn_cls = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * self.n_directions

        self.rul_head = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        )

        self.degradation_head = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
            nn.Tanh(),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"HealthMonitoringRNN ({rnn_type.upper()}) inicializada: "
                    f"{n_params:,} parámetros, salida dual (RUL + degradación)")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Codifica una ventana de sensores y predice RUL y vector de degradación.

        Se queda con el último estado oculto en lugar de con toda la secuencia de
        salidas porque la predicción se refiere al instante final de la ventana: es
        ese estado el que resume el historial reciente del motor. En el caso
        bidireccional se concatenan los estados de ambas direcciones.

        param x: Tensor (batch, sequence_length, n_sensors) con la ventana de
            sensores ya normalizada.
        return: Tupla (RUL predicha de forma (batch, 1), vector de degradación de
            forma (batch, 4) en el orden de DEGRADATION_LABELS).
        """
        h = self.input_proj(x)
        if self.rnn_type == 'lstm':
            _, (h_n, _) = self.rnn(h)
        else:
            _, h_n = self.rnn(h)

        if self.n_directions == 2:
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last = h_n[-1]

        return self.rul_head(last), self.degradation_head(last)

    def predict_with_uncertainty(self, x: torch.Tensor,
                                 n_samples: int = 50) -> dict:
        """Predicción con MC-Dropout para estimar incertidumbre epistémica.

        Ejecuta `n_samples` pases forward con dropout activo y devuelve
        media, desviación estándar e intervalo de confianza al 95% del
        RUL, junto con la media y desviación del vector de degradación.

        Es necesario porque una predicción puntual de vida remanente no basta para
        tomar decisiones de mantenimiento: hace falta saber cuánto se puede confiar
        en ella. Cada pase equivale a muestrear una subred distinta, y la dispersión
        entre pases aproxima la incertidumbre del modelo. El estado original del
        modelo (train o eval) se restaura al terminar para no interferir con el
        flujo del llamante.

        param x: Tensor (batch, sequence_length, n_sensors) con las ventanas a evaluar.
        param n_samples: Número de pases estocásticos de Monte Carlo.
        return: Diccionario con rul_mean, rul_std, los extremos rul_ci_lower y
            rul_ci_upper del intervalo al 95%, y degradation_mean y degradation_std.
        """
        was_training = self.training  # Guarda el estado original
        self.train()   # Activa dropout aunque estemos en inferencia
        rul_samples, deg_samples = [], []
        with torch.no_grad():
            for _ in range(n_samples):
                rul, deg = self.forward(x)
                rul_samples.append(rul)
                deg_samples.append(deg)

        rul_stack = torch.stack(rul_samples)
        deg_stack = torch.stack(deg_samples)

        rul_mean = rul_stack.mean(0).squeeze()
        rul_std = rul_stack.std(0).squeeze()
        deg_mean = deg_stack.mean(0)
        deg_std = deg_stack.std(0)

        # Restaura el estado original (train o eval) del modelo
        self.train(was_training)
        return {
            'rul_mean':     rul_mean,
            'rul_std':      rul_std,
            'rul_ci_lower': rul_mean - 1.96 * rul_std,
            'rul_ci_upper': rul_mean + 1.96 * rul_std,
            'degradation_mean': deg_mean,
            'degradation_std':  deg_std,
        }


class CMAPSSDataset(torch.utils.data.Dataset):
    """Dataset C-MAPSS con pseudo-labels de degradación.

    Genera ventanas deslizantes de longitud `sequence_length` sobre los
    sensores informativos y calcula los pseudo-labels de degradación
    como delta porcentual respecto a la media de los primeros
    `baseline_cycles` ciclos de cada motor.

    Los pseudo-labels son necesarios porque C-MAPSS no publica el estado real de
    cada componente, solo la vida remanente: referenciar cada sensor a su propio
    baseline inicial permite construir una señal de degradación por componente
    normalizada entre motores, que es lo que aprende el segundo cabezal.
    """

    def __init__(self, df: pd.DataFrame, sequence_length: int = 30,
                 max_rul: int = 125, baseline_cycles: int = 10):
        """Materializa en memoria todas las ventanas y sus etiquetas.

        Recorre motor por motor, calcula su baseline robusto y genera las ventanas
        sin cruzar la frontera entre unidades, condición imprescindible para que
        ninguna secuencia mezcle el final de un motor con el principio de otro.
        Precalcularlo todo aquí, en lugar de en __getitem__, evita repetir el mismo
        troceado en cada época de entrenamiento.

        param df: DataFrame C-MAPSS con unit_id, cycle, RUL y los sensores de
            HealthMonitoringRNN.SENSOR_FEATURES.
        param sequence_length: Longitud en ciclos de las ventanas deslizantes.
        param max_rul: Saturación superior de la RUL, práctica estándar en PHM.
        param baseline_cycles: Ciclos iniciales promediados como referencia de
            motor sano para los pseudo-labels de degradación.
        return: None; deja las secuencias y etiquetas cargadas como arrays NumPy.
        """
        self.sequence_length = sequence_length

        sensors = [s for s in HealthMonitoringRNN.SENSOR_FEATURES
                   if s in df.columns]
        deg_sensors = [s for s in HealthMonitoringRNN.DEGRADATION_SENSORS
                       if s in df.columns]

        self.sequences = []
        self.rul_targets = []
        self.deg_targets = []

        for unit_id in df['unit_id'].unique():
            unit = df[df['unit_id'] == unit_id].sort_values('cycle')
            sensor_vals = unit[sensors].values.astype(np.float32)
            rul_vals = np.clip(
                unit['RUL'].values.astype(np.float32), 0, max_rul)

            # Baseline robusto: media de los primeros ciclos
            n_base = min(baseline_cycles, len(unit))
            baselines = {}
            for s in deg_sensors:
                base_mean = unit[s].iloc[:n_base].mean()
                baselines[s] = base_mean if abs(base_mean) > 1e-8 else 1e-8

            # Pseudo-labels de degradación
            deg_vals = np.zeros(
                (len(unit), len(deg_sensors)), dtype=np.float32)
            for j, s in enumerate(deg_sensors):
                deg_vals[:, j] = (
                    unit[s].values - baselines[s]) / abs(baselines[s])

            # Ventanas deslizantes
            for i in range(len(sensor_vals) - sequence_length + 1):
                self.sequences.append(sensor_vals[i:i + sequence_length])
                self.rul_targets.append(rul_vals[i + sequence_length - 1])
                self.deg_targets.append(deg_vals[i + sequence_length - 1])

        self.sequences = np.array(self.sequences)
        self.rul_targets = np.array(self.rul_targets)
        self.deg_targets = np.array(self.deg_targets)

        logger.info(f"CMAPSSDataset preparado: {len(self.sequences)} "
                    f"secuencias, baseline = {baseline_cycles} ciclos, "
                    f"4 pseudo-labels de degradación")

    def __len__(self) -> int:
        """Devuelve el número de ventanas disponibles en el dataset.

        Lo exige la interfaz Dataset de PyTorch: es lo que usa el DataLoader para
        dimensionar las épocas y barajar los índices.

        return: Número total de ventanas deslizantes generadas sobre todos los motores.
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple:
        """Devuelve una ventana y sus dos etiquetas convertidas a tensores.

        La RUL se devuelve como slice de un elemento, no como escalar, para que su
        forma coincida con la salida (batch, 1) del cabezal correspondiente y la MSE
        no dispare un broadcast silencioso.

        param idx: Índice de la ventana solicitada por el DataLoader.
        return: Tupla (ventana de sensores, RUL objetivo, vector de degradación
            objetivo) como tensores de PyTorch.
        """
        return (
            torch.tensor(self.sequences[idx]),
            torch.tensor(self.rul_targets[idx:idx + 1]),
            torch.tensor(self.deg_targets[idx]),
        )


class HealthMonitor:
    """Wrapper de alto nivel para entrenar y evaluar HealthMonitoringRNN.

    Implementa pérdida dual L = L_RUL + α·L_degradación donde α actúa
    como regularizador moderado del cabezal de degradación sin degradar
    el rendimiento sobre el RUL principal.

    Es necesario para que la comparación entre variantes recurrentes del TFG
    (LSTM-64, LSTM-128 y GRU-64) se haga con idéntico preprocesado, partición por
    motor y criterio de parada: solo así la diferencia observada en RMSE puede
    atribuirse a la arquitectura y no al montaje experimental.
    """

    DEFAULT_CHECKPOINT_DIR = Path('checkpoints/health_monitoring')
    # El nombre del checkpoint se construye dinámicamente a partir de
    # rnn_type y hidden_dim en fit() para reflejar la arquitectura entrenada:
    # rnn_lstm_64.pt, rnn_lstm_128.pt, rnn_gru_64.pt, etc.

    def __init__(self, rnn_type: str = 'lstm', hidden_dim: int = 64,
                 n_layers: int = 2, sequence_length: int = 30,
                 lr: float = 1e-3, alpha: float = 0.3,
                 device: str = 'auto'):
        """Construye la red, el optimizador y el planificador sobre el dispositivo.

        El planificador reduce la tasa de aprendizaje al estancarse el RMSE de RUL,
        no según un calendario fijo, porque el número de épocas útiles varía mucho
        entre arquitecturas y un coseno prefijado penalizaría a las más lentas.

        param rnn_type: Variante recurrente, 'lstm' o 'gru'.
        param hidden_dim: Dimensión oculta de la RNN y de los cabezales MLP.
        param n_layers: Capas recurrentes apiladas.
        param sequence_length: Longitud de las ventanas deslizantes del dataset.
        param lr: Tasa de aprendizaje inicial del optimizador Adam.
        param alpha: Peso del término de degradación en la pérdida dual.
        param device: 'auto' para detectar CUDA, o 'cuda'/'cpu' explícito.
        return: None; deja el wrapper listo para prepare_cmapss() y fit().
        """
        if device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.sequence_length = sequence_length
        self.alpha = alpha

        # Semilla fija ANTES de instanciar el modelo, para que la
        # inicialización de pesos sea reproducible entre ejecuciones
        # y las tres arquitecturas del TFG (LSTM-64, LSTM-128, GRU-64)
        # partan de la misma semilla base para una comparación justa.
        torch.manual_seed(42)
        np.random.seed(42)

        self.model = HealthMonitoringRNN(
            n_sensors=len(HealthMonitoringRNN.SENSOR_FEATURES),
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            rnn_type=rnn_type,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5)
        self.train_history = []

        logger.info(
            f"HealthMonitor inicializado ({rnn_type.upper()}) en {self.device}")
        logger.info(
            f"  Pérdida dual: L_RUL + {alpha}·L_degradación")

    def prepare_cmapss(self, loader, subset: str = 'FD001',
                       train_ratio: float = 0.8) -> tuple:
        """Prepara los DataLoaders de entrenamiento y validación desde C-MAPSS.

        Carga el sub-dataset, calcula el health index, normaliza los sensores y
        reparte los motores, no las filas, entre train y validación. La partición
        por motor es imprescindible: si dos ventanas del mismo motor cayesen en
        conjuntos distintos habría fuga de información y el RMSE reportado sería
        optimista. La permutación usa semilla fija para ser reproducible.

        param loader: Instancia de CMAPSSLoader con la ruta al dataset.
        param subset: Sub-dataset a utilizar ('FD001'…'FD004').
        param train_ratio: Fracción de motores destinados a entrenamiento.
        return: Tupla (train_loader, val_loader), también almacenados como atributos
            para que train_epoch() y validate() los usen sin recibirlos.
        """
        df = loader.load_subset(subset)
        df = loader.compute_health_index(df)
        df = loader.normalize(df, method='minmax')

        units = df['unit_id'].unique()
        np.random.RandomState(42).shuffle(units)
        n_train = int(len(units) * train_ratio)

        df_train = df[df['unit_id'].isin(units[:n_train])]
        df_val = df[df['unit_id'].isin(units[n_train:])]

        train_ds = CMAPSSDataset(
            df_train, self.sequence_length, baseline_cycles=10)
        val_ds = CMAPSSDataset(
            df_val, self.sequence_length, baseline_cycles=10)

        self.train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=64, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=128)

        logger.info(f"  Train: {len(train_ds)} secuencias "
                    f"({n_train} motores)")
        logger.info(f"  Val:   {len(val_ds)} secuencias "
                    f"({len(units) - n_train} motores)")
        return self.train_loader, self.val_loader

    def train_epoch(self) -> tuple[float, float]:
        """Entrena el modelo durante una época sobre la pérdida dual.

        Combina el error de RUL con el de degradación ponderado por alpha y aplica
        recorte de norma de gradiente, necesario en redes recurrentes para evitar
        que un lote con una transición brusca de sensores dispare el gradiente y
        deshaga el progreso acumulado.

        return: Tupla (pérdida media de RUL, pérdida media de degradación) sobre
            todos los lotes de la época, sin ponderar por alpha.
        """
        self.model.train()
        rul_losses, deg_losses = [], []

        for xb, yb_rul, yb_deg in self.train_loader:
            xb = xb.to(self.device)
            yb_rul = yb_rul.to(self.device)
            yb_deg = yb_deg.to(self.device)

            self.optimizer.zero_grad()
            rul_pred, deg_pred = self.model(xb)

            loss_rul = nn.functional.mse_loss(rul_pred, yb_rul)
            loss_deg = nn.functional.mse_loss(deg_pred, yb_deg)
            loss = loss_rul + self.alpha * loss_deg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            rul_losses.append(loss_rul.item())
            deg_losses.append(loss_deg.item())

        return float(np.mean(rul_losses)), float(np.mean(deg_losses))

    @torch.no_grad()
    def validate(self) -> dict:
        """Evalúa el modelo en validación con dropout desactivado.

        Acumula todas las predicciones antes de medir, de modo que RMSE y MAE se
        calculen sobre el conjunto completo. Desglosa además el error de degradación
        por componente, necesario para comprobar que el cabezal no está aprendiendo
        solo el componente dominante (HPC) e ignorando el resto.

        return: Diccionario con rul_rmse, rul_mae, deg_rmse_total y el RMSE de
            degradación de fan, HPC, HPT y LPT por separado.
        """
        self.model.eval()
        rul_preds, rul_targs = [], []
        deg_preds, deg_targs = [], []

        for xb, yb_rul, yb_deg in self.val_loader:
            xb = xb.to(self.device)
            rul_p, deg_p = self.model(xb)
            rul_preds.append(rul_p.cpu())
            rul_targs.append(yb_rul)
            deg_preds.append(deg_p.cpu())
            deg_targs.append(yb_deg)

        rul_pred = torch.cat(rul_preds).squeeze()
        rul_true = torch.cat(rul_targs).squeeze()
        deg_pred = torch.cat(deg_preds)
        deg_true = torch.cat(deg_targs)

        rul_rmse = torch.sqrt(((rul_pred - rul_true) ** 2).mean()).item()
        rul_mae = torch.abs(rul_pred - rul_true).mean().item()

        deg_rmse_per_comp = [
            torch.sqrt(((deg_pred[:, i] - deg_true[:, i]) ** 2).mean()).item()
            for i in range(len(HealthMonitoringRNN.DEGRADATION_LABELS))
        ]
        deg_rmse_total = torch.sqrt(
            ((deg_pred - deg_true) ** 2).mean()).item()

        return {
            'rul_rmse': rul_rmse,
            'rul_mae':  rul_mae,
            'deg_rmse_total': deg_rmse_total,
            'deg_rmse_fan': deg_rmse_per_comp[0],
            'deg_rmse_hpc': deg_rmse_per_comp[1],
            'deg_rmse_hpt': deg_rmse_per_comp[2],
            'deg_rmse_lpt': deg_rmse_per_comp[3],
        }

    def validate_with_uq(self, n_mc: int = 50) -> dict:
        """Evalúa con MC-Dropout y mide la calibración del intervalo al 95%.

        Compara la fracción real de etiquetas que caen dentro del intervalo predicho
        con el 95% nominal. Es necesario porque un intervalo de confianza solo es
        útil si está calibrado: una cobertura muy inferior indicaría exceso de
        confianza y una muy superior, intervalos tan anchos que no informan.

        param n_mc: Número de pases estocásticos de Monte Carlo por lote.
        return: Diccionario con rmse_mc (RMSE de la media MC), mean_uncertainty
            (desviación típica media en ciclos) y ci_95_coverage (cobertura real).
        """
        all_means, all_stds, all_targets = [], [], []
        for xb, yb_rul, _ in self.val_loader:
            xb = xb.to(self.device)
            uq = self.model.predict_with_uncertainty(xb, n_samples=n_mc)
            all_means.append(uq['rul_mean'].cpu())
            all_stds.append(uq['rul_std'].cpu())
            all_targets.append(yb_rul.squeeze())

        means = torch.cat(all_means)
        stds = torch.cat(all_stds)
        targs = torch.cat(all_targets)

        rmse = torch.sqrt(((means - targs) ** 2).mean()).item()
        coverage = ((targs >= means - 1.96 * stds) &
                    (targs <= means + 1.96 * stds)).float().mean().item()

        return {
            'rmse_mc':          rmse,
            'mean_uncertainty': stds.mean().item(),
            'ci_95_coverage':   coverage,
        }

    def fit(self, epochs: int = 100, patience: int = 20,
            checkpoint_path: str = None) -> list:
        """Ejecuta el entrenamiento completo, guarda el mejor modelo y lo evalúa.

        Alterna entrenamiento y validación, guarda el checkpoint cada vez que mejora
        el RMSE de RUL y detiene el bucle tras `patience` épocas sin mejora. Al
        terminar recarga el mejor checkpoint y ejecuta una validación final con y sin
        MC-Dropout, de modo que las cifras que quedan en el log corresponden siempre
        al modelo guardado y no a la última época, que puede ser peor.

        param epochs: Número máximo de épocas a ejecutar.
        param patience: Épocas sin mejora del RMSE de RUL antes de parar.
        param checkpoint_path: Ruta del checkpoint; si es None se usa la ruta por
            defecto de la clase.
        return: Lista de diccionarios con el historial por época; el modelo queda
            con los pesos de la mejor época cargados.
        """
        if checkpoint_path is None:
            # Nombre dinámico coherente con plot_rnn_comparison.py:
            # rnn_{lstm|gru}_{hidden_dim}.pt
            rnn_type = self.model.rnn_type
            hidden_dim = self.model.hidden_dim
            filename = f'rnn_{rnn_type}_{hidden_dim}.pt'
            checkpoint_path = str(self.DEFAULT_CHECKPOINT_DIR / filename)

        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        best_rmse = float('inf')
        wait = 0

        logger.info(
            f"Entrenando RNN Health Monitor: {epochs} epochs, "
            f"patience = {patience}")
        logger.info(f"  Pérdida dual: L_RUL + {self.alpha}·L_degradación")

        for epoch in range(epochs):
            loss_rul, loss_deg = self.train_epoch()
            metrics = self.validate()
            self.scheduler.step(metrics['rul_rmse'])

            self.train_history.append({
                'epoch': epoch,
                'train_rul':    loss_rul,
                'train_deg':    loss_deg,
                'val_rul_rmse': metrics['rul_rmse'],
                'val_rul_mae':  metrics['rul_mae'],
                'val_deg_rmse': metrics['deg_rmse_total'],
            })

            if metrics['rul_rmse'] < best_rmse:
                best_rmse = metrics['rul_rmse']
                wait = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                wait += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch + 1:>3}/{epochs} | "
                    f"L_RUL: {loss_rul:.2f} | L_deg: {loss_deg:.4f} | "
                    f"Val RUL RMSE: {metrics['rul_rmse']:.2f} | "
                    f"Val Deg RMSE: {metrics['deg_rmse_total']:.4f}")

            if wait >= patience:
                logger.info(f"  Early stopping activado en epoch {epoch + 1}")
                break

        self.model.load_state_dict(
            torch.load(checkpoint_path, weights_only=False))
        logger.info(f"Mejor modelo cargado: RUL RMSE = {best_rmse:.2f}")

        # Validación final completa
        final = self.validate()
        logger.info("Validación final:")
        logger.info(f"  RUL RMSE:               {final['rul_rmse']:.2f} ciclos")
        logger.info(f"  RUL MAE:                {final['rul_mae']:.2f} ciclos")
        logger.info(f"  Degradación RMSE total: {final['deg_rmse_total']:.4f}")
        logger.info(f"    Fan: {final['deg_rmse_fan']:.4f}  "
                    f"HPC: {final['deg_rmse_hpc']:.4f}  "
                    f"HPT: {final['deg_rmse_hpt']:.4f}  "
                    f"LPT: {final['deg_rmse_lpt']:.4f}")

        # MC-Dropout UQ
        logger.info("Estimación de incertidumbre epistémica (MC-Dropout, 50 pases):")
        uq = self.validate_with_uq()
        logger.info(f"  RMSE (MC):        {uq['rmse_mc']:.2f}")
        logger.info(f"  Incertidumbre σ:  ±{uq['mean_uncertainty']:.2f} ciclos")
        logger.info(f"  Cobertura IC 95%: {uq['ci_95_coverage']:.1%}")

        return self.train_history