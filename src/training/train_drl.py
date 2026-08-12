"""
Última Fecha de Modificación: 08/Aug/2026
Descripción train_drl.py: Interfaz de línea de comandos para entrenar los agentes
de control del motor ACE. Carga el gemelo digital PINN, construye el entorno
ACE-v0, instancia SAC o TD3 con los hiperparámetros calibrados del proyecto y
ejecuta el entrenamiento con evaluación periódica, registro de métricas por
episodio y guardado tanto del mejor checkpoint según reward como del modelo
final. Incluye además la comparativa conjunta SAC/TD3/FADEC que documenta el
apartado de agentes DRL del proyecto.

HALLAZGOS empíricos (semilla=42, mission='mixed', 500k timesteps):    
    - SAC:            reward 209.59 ± 46.23, safety 60.0%
    - TD3:            reward 205.41 ± 40.50, safety 56.0%
    - FADEC baseline: reward -1947.28 ± 1533.57, safety 12.0%
    (El agente TD3+BC winner, reward 249.94 ± 18.95 con safety 84.0%,
    se entrena aparte con train_bc_drl.py.)

Los checkpoints se guardan en `checkpoints/control/{algo}_{mission}/`,
sobreescribiendo los del proyecto activo. Los logs asociados se guardan en
`logs/{algo}_{mission}/` con la misma nomenclatura para poder localizarlos
con `plot_drl_section5.py`.

Reproducibilidad: la semilla por defecto (--seed 42) se propaga al agente de
Stable-Baselines3 (parámetro seed=) y al entorno ACE-v0 (env.reset(seed=)). El
entrenamiento sigue siendo dependiente de la variabilidad de PyTorch en GPU;
para reproducibilidad total ejecutar en CPU o fijar CUBLAS_WORKSPACE_CONFIG.

Referencias:
    Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). "Soft Actor-Critic:
        Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic
        Actor." ICML 2018.
    Fujimoto, S., van Hoof, H., & Meger, D. (2018). "Addressing Function
        Approximation Error in Actor-Critic Methods." ICML 2018.

Uso:
    python src/training/train_drl.py                        # SAC (default)
    python src/training/train_drl.py --algo td3
    python src/training/train_drl.py --timesteps 500000
    python src/training/train_drl.py --mission full
    python src/training/train_drl.py --algo both           # SAC y TD3
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# Configuración global
PINN_CHECKPOINT = Path('checkpoints/digital_twin/pinn.pt')
PINN_HIDDEN_DIM = 224
PINN_N_LAYERS = 8

# Estructura de carpetas del proyecto para los checkpoints DRL activos.
CONTROL_ROOT = Path('checkpoints/control')
LOGS_ROOT = Path('logs')

DEFAULT_MAX_STEPS = 200
DEFAULT_DEGRADATION_RATE = 0.005

EVAL_FREQ = 5000
EVAL_EPISODES_CALLBACK = 10
EVAL_EPISODES_FINAL = 50


# Hiperparámetros del entrenamiento

# Los hiperparámetros por algoritmo se han calibrado empíricamente sobre
# el entorno ACE-v0 y corresponden a los checkpoints activos del
# proyecto (Sección 5). Constituyen un óptimo local para esta
# combinación entorno-recompensa; su alteración requiere reentrenar
# los agentes para preservar coherencia con los resultados reportados.
SAC_HYPERPARAMS = {
    'learning_rate':    1e-4,
    'buffer_size':      100_000,
    'batch_size':       256,
    'tau':              0.005,
    'gamma':            0.99,
    'ent_coef':         0.2,       # con target_entropy='auto'
    'target_entropy':   'auto',
    'train_freq':       1,
    'gradient_steps':   1,
    'learning_starts':  5000,
    'policy_kwargs':    dict(net_arch=[256, 256], log_std_init=-2.0),
}

TD3_HYPERPARAMS = {
    'learning_rate':       1e-4,
    'buffer_size':         100_000,
    'batch_size':          256,
    'tau':                 0.005,
    'gamma':               0.99,
    'train_freq':          1,
    'gradient_steps':      1,
    'learning_starts':     1000,
    'policy_delay':        2,
    'target_policy_noise': 0.2,
    'target_noise_clip':   0.5,
    'policy_kwargs':       dict(net_arch=[256, 256]),
}


# Callbacks e infraestructura
def _import_sb3():
    """Importa Stable-Baselines3 bajo demanda y devuelve los símbolos usados aquí.

    La importación se aplaza a la llamada en lugar de hacerse a nivel de módulo
    porque cargar SB3 y PyTorch cuesta varios segundos: así el script responde de
    inmediato a `--help` o a un argumento mal escrito, sin pagar ese coste.

    return: Tupla con SAC, TD3, BaseCallback, CallbackList, EvalCallback, Monitor y
        DummyVecEnv, en ese orden.
    """
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.callbacks import (
        BaseCallback, CallbackList, EvalCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    return SAC, TD3, BaseCallback, CallbackList, EvalCallback, Monitor, DummyVecEnv


def build_metrics_callback(base_cls, log_dir: Path):
    """Fabrica un callback que persiste a CSV las métricas de cada episodio.

    Se construye como fábrica porque la clase base de los callbacks vive dentro de
    Stable-Baselines3, que se importa de forma diferida: definir la clase a nivel de
    módulo obligaría a cargar la librería solo por tener la definición disponible.

    param base_cls: Clase BaseCallback de Stable-Baselines3 de la que heredar.
    param log_dir: Carpeta donde se escribirá training_metrics.csv.
    return: Instancia del callback ya construida, lista para pasarla a learn().
    """
    class MetricsCallback(base_cls):
        """Registra reward, longitud, violaciones y métricas de motor.

        Es necesario porque el logging propio de SB3 solo guarda recompensa y
        longitud: las curvas de la memoria necesitan además empuje, consumo,
        temperatura de turbina y violaciones, que solo están disponibles en el
        diccionario info de cada paso.
        """

        def __init__(self, verbose: int = 0):
            """Prepara la carpeta de logs y la lista acumuladora de métricas.

            param verbose: Si es mayor que cero se emite una traza cada diez
                episodios completados.
            return: None; deja el callback listo para el bucle de entrenamiento.
            """
            super().__init__(verbose)
            self.log_dir = log_dir
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.episode_metrics = []

        def _on_step(self) -> bool:
            """Captura las métricas de los episodios que acaban de terminar.

            Se invoca en cada paso del entrenamiento, pero solo actúa cuando el
            wrapper Monitor ha insertado la clave 'episode' en info, que es la señal
            de que un episodio se ha cerrado y sus agregados están disponibles.

            return: True siempre, para que el entrenamiento continúe; devolver False
                lo abortaría.
            """
            for info in self.locals.get('infos', []):
                if 'episode' in info:
                    self.episode_metrics.append({
                        'timestep':   self.num_timesteps,
                        'reward':     info['episode']['r'],
                        'length':     info['episode']['l'],
                        'violations': info.get('episode_violations', 0),
                        'thrust':     info.get('thrust', 0),
                        'sfc':        info.get('sfc', 0),
                        'T4':         info.get('T4', 0),
                    })
                    if self.verbose > 0 and \
                            len(self.episode_metrics) % 10 == 0:
                        logger.info(
                            f"  Episodio {len(self.episode_metrics)}: "
                            f"R = {info['episode']['r']:.1f}, "
                            f"Violaciones = "
                            f"{info.get('episode_violations', 0)}")
            return True

        def _on_training_end(self) -> None:
            """Vuelca a disco todas las métricas acumuladas durante el entrenamiento.

            Se escribe una sola vez al final en lugar de ir añadiendo filas: el
            entrenamiento hace cientos de miles de pasos y abrir el CSV en cada
            episodio penalizaría el rendimiento sin ninguna ventaja.

            return: None; genera training_metrics.csv dentro de la carpeta de logs.
            """
            import pandas as pd
            df = pd.DataFrame(self.episode_metrics)
            output = self.log_dir / 'training_metrics.csv'
            df.to_csv(output, index=False)
            logger.info(f"Métricas de entrenamiento guardadas en {output}")

    return MetricsCallback()


def load_pinn():
    """Carga el gemelo digital activo, o devuelve None si no está disponible.

    Degradar a None en lugar de abortar es deliberado: el entorno sabe funcionar con
    su modelo simplificado interno, de modo que una instalación sin checkpoint
    entrenado todavía puede ejecutar el flujo completo para verificarlo. La carga usa
    strict=False para tolerar checkpoints con claves de normalización adicionales.

    return: Instancia de BraytonPINN en modo evaluación, o None si el checkpoint no
        existe o no puede cargarse.
    """
    from src.models.pinn import BraytonPINN
    import torch

    if not PINN_CHECKPOINT.exists():
        logger.warning(f"PINN no encontrado en {PINN_CHECKPOINT}; "
                       "el entorno operará sin gemelo digital.")
        return None

    try:
        pinn = BraytonPINN(hidden_dim=PINN_HIDDEN_DIM,
                           n_layers=PINN_N_LAYERS)
        pinn.load_state_dict(
            torch.load(PINN_CHECKPOINT, weights_only=True), strict=False)
        pinn.eval()
        n_params = sum(p.numel() for p in pinn.parameters())
        logger.info(f"PINN cargado como gemelo digital: {n_params:,} "
                    f"parámetros, 3 constraint layers")
        return pinn
    except Exception as exc:
        logger.warning(f"No se pudo cargar el PINN: {exc}")
        return None


def build_env(pinn, mission: str, Monitor):
    """Construye un ACEEnv envuelto en Monitor con la configuración del proyecto.

    El wrapper Monitor es el que agrega recompensa y longitud por episodio y las
    inyecta en info; sin él, el callback de métricas no tendría de dónde leerlas.
    Centralizar aquí la construcción garantiza que el entorno de entrenamiento y el
    de evaluación se creen con parámetros idénticos.

    param pinn: Gemelo digital a usar como modelo del motor, o None.
    param mission: Perfil de misión con el que se generarán los episodios.
    param Monitor: Clase Monitor de Stable-Baselines3, recibida por la importación
        diferida.
    return: Entorno ACEEnv envuelto en Monitor, listo para vectorizar.
    """
    from src.environments.ace_env import ACEEnv
    env = ACEEnv(
        pinn_model=pinn,
        mission_profile=mission,
        max_steps=DEFAULT_MAX_STEPS,
        degradation_rate=DEFAULT_DEGRADATION_RATE,
    )
    return Monitor(env)


def instantiate_agent(algo: str, algo_cls, train_env, log_dir: Path,
                      seed: int):
    """Instancia el agente SAC o TD3 con los hiperparámetros calibrados del proyecto.

    Selecciona el bloque de hiperparámetros según el algoritmo y fija la semilla. Es
    necesario para que los dos algoritmos se construyan siempre desde la misma
    receta documentada: cualquier ajuste ad hoc en el punto de llamada rompería la
    correspondencia entre los checkpoints publicados y los resultados de la memoria.

    param algo: Identificador del algoritmo, 'sac' o 'td3'.
    param algo_cls: Clase del algoritmo de Stable-Baselines3 a instanciar.
    param train_env: Entorno vectorizado sobre el que aprenderá el agente.
    param log_dir: Carpeta base donde se escribirán los eventos de TensorBoard.
    param seed: Semilla de inicialización, para reproducibilidad.
    return: Agente de Stable-Baselines3 listo para llamar a learn().
    """
    hparams = SAC_HYPERPARAMS if algo == 'sac' else TD3_HYPERPARAMS
    return algo_cls(
        'MlpPolicy',
        train_env,
        verbose=0,
        seed=seed,
        tensorboard_log=str(log_dir / 'tensorboard'),
        **hparams,
    )


def build_run_paths(algo: str, mission: str) -> tuple:
    """Devuelve (log_dir, checkpoint_dir) para una configuración concreta.

    Sobreescribe los checkpoints y logs del run anterior con la misma
    combinación algo/mission. Esta convención garantiza que
    plot_drl_section5.py encuentra siempre el modelo activo.

    La nomenclatura fija, sin marca temporal, es una decisión consciente: los scripts
    de figuras localizan los resultados por nombre, de modo que rutas con timestamp
    obligarían a editarlos tras cada reentrenamiento.

    param algo: Identificador del algoritmo, 'sac' o 'td3'.
    param mission: Perfil de misión del entrenamiento.
    return: Tupla (carpeta de logs, carpeta de checkpoints), ambas ya creadas.
    """
    # Nomenclatura {algo}_{mission}: coherente con la estructura de
    # checkpoints/control/ y logs/ del proyecto. El TD3+BC entrenado
    # con train_bc_drl.py usa su propia carpeta 'td3_bc' generada por
    # ese script, no por esta función.
    run_name = f'{algo}_{mission}'
    log_dir = LOGS_ROOT / run_name
    checkpoint_dir = CONTROL_ROOT / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, checkpoint_dir


# Entrenamiento y evaluación
def train_agent(algo: str = 'sac', total_timesteps: int = 500_000,
                mission: str = 'mixed',
                pinn_model=None, seed: int = 42) -> Dict:
    """Entrena un agente DRL de principio a fin y devuelve su evaluación final.

    Orquesta todo el flujo: carga del gemelo digital, construcción de los entornos de
    entrenamiento y evaluación, instanciación del agente, aprendizaje con callbacks,
    guardado del modelo final y evaluación sobre episodios independientes. Se usan dos
    entornos separados a propósito, para que las métricas de evaluación no se midan
    sobre la misma instancia que el agente está explorando.

    param algo: Algoritmo a entrenar, 'sac' o 'td3'.
    param total_timesteps: Presupuesto total de pasos de entrenamiento.
    param mission: Perfil de misión con el que se generan los episodios.
    param pinn_model: Gemelo digital ya cargado; si es None se carga aquí.
    param seed: Semilla de reproducibilidad del agente.
    return: Diccionario de resultados de la evaluación final, también volcado a
        results.json dentro de la carpeta de logs.
    """
    (SAC, TD3, BaseCallback, CallbackList,
     EvalCallback, Monitor, DummyVecEnv) = _import_sb3()

    if pinn_model is None:
        pinn_model = load_pinn()

    log_dir, checkpoint_dir = build_run_paths(algo, mission)

    logger.info(f"Entrenamiento DRL: {algo.upper()}")
    logger.info(f"  Timesteps: {total_timesteps:,}")
    logger.info(f"  Misión:    {mission}")
    logger.info(f"  Seed:      {seed}")
    logger.info(f"  Log dir:   {log_dir}")
    logger.info(f"  Checkpoint dir: {checkpoint_dir}")

    train_env = DummyVecEnv([lambda: build_env(pinn_model, mission, Monitor)])
    eval_env = DummyVecEnv([lambda: build_env(pinn_model, mission, Monitor)])

    algo_cls = SAC if algo == 'sac' else TD3
    model = instantiate_agent(algo, algo_cls, train_env, log_dir, seed)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir),
        log_path=str(log_dir),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES_CALLBACK,
        deterministic=True,
    )
    metrics_callback = build_metrics_callback(BaseCallback, log_dir)
    callbacks = CallbackList([eval_callback, metrics_callback])

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = checkpoint_dir / f'{algo}_final'
    model.save(str(final_path))
    logger.info(f"Modelo final guardado en {final_path}")

    logger.info(f"Evaluación final sobre {EVAL_EPISODES_FINAL} episodios")
    results = evaluate_agent(model, eval_env, algo, total_timesteps,
                             mission, seed, final_path)

    with open(log_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    log_results_summary(algo, results)
    return results


def evaluate_agent(model, eval_env, algo: str, timesteps: int,
                    mission: str, seed: int, checkpoint_path: Path) -> Dict:
    """Evalúa el modelo entrenado sobre un número fijo de episodios completos.

    Ejecuta la política de forma determinista, sin exploración, para medir lo que el
    agente ha aprendido y no lo que su ruido de exploración produce. Accede al
    entorno interno del vectorizado porque la interfaz clásica (obs, info) devuelve el
    diccionario info completo, del que se extrae el recuento de violaciones.

    param model: Agente entrenado con el método predict() de Stable-Baselines3.
    param eval_env: Entorno vectorizado de evaluación, con un único sub-entorno.
    param algo: Identificador del algoritmo, incluido en el resultado.
    param timesteps: Pasos de entrenamiento consumidos, incluidos en el resultado.
    param mission: Perfil de misión evaluado, incluido en el resultado.
    param seed: Semilla usada, incluida en el resultado para trazabilidad.
    param checkpoint_path: Ruta del modelo evaluado, incluida en el resultado.
    return: Diccionario con la configuración del run, la recompensa media, su
        desviación y rango, las violaciones por episodio y la tasa de seguridad.
    """
    rewards = []
    violations_total = 0

    # eval_env es un DummyVecEnv con un único sub-entorno; accedemos
    # directamente al ACEEnv subyacente para invocar reset/step con la
    # firma clásica (obs, info) en lugar de la vectorizada.
    inner_env = eval_env.envs[0]

    for _ in range(EVAL_EPISODES_FINAL):
        obs, info = inner_env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = inner_env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        violations_total += info.get('episode_violations', 0)

    return {
        'algo':            algo,
        'timesteps':       timesteps,
        'mission':         mission,
        'seed':            seed,
        'mean_reward':     float(np.mean(rewards)),
        'std_reward':      float(np.std(rewards)),
        'min_reward':      float(np.min(rewards)),
        'max_reward':      float(np.max(rewards)),
        'mean_violations': violations_total / EVAL_EPISODES_FINAL,
        'safety_rate':     1 - violations_total / (EVAL_EPISODES_FINAL
                                                    * DEFAULT_MAX_STEPS),
        'checkpoint':      str(checkpoint_path),
    }


def log_results_summary(algo: str, results: Dict) -> None:
    """Emite por el log un resumen legible de los resultados de evaluación.

    Deja las cifras clave a la vista al terminar un entrenamiento que puede haber
    durado horas, sin obligar a abrir el results.json para saber si el run ha salido
    bien o conviene descartarlo.

    param algo: Identificador del algoritmo, usado como encabezado.
    param results: Diccionario de resultados devuelto por evaluate_agent().
    return: None; el resumen se emite por el logger del módulo.
    """
    logger.info(f"Resultados {algo.upper()}:")
    logger.info(f"  Reward:         {results['mean_reward']:.2f} "
                f"± {results['std_reward']:.2f}")
    logger.info(f"  Rango reward:   [{results['min_reward']:.2f}, "
                f"{results['max_reward']:.2f}]")
    logger.info(f"  Violaciones/ep: {results['mean_violations']:.2f}")
    logger.info(f"  Tasa seguridad: {results['safety_rate']:.1%}")


def compare_sac_vs_td3(total_timesteps: int = 500_000,
                        mission: str = 'mixed', seed: int = 42) -> tuple:
    """Entrena SAC y TD3 y los compara con el baseline FADEC en la misma misión.

    Encadena los dos entrenamientos, evalúa después el controlador convencional en
    condiciones idénticas y tabula las tres columnas. Es necesario porque la
    conclusión del TFG no es el rendimiento absoluto de un algoritmo, sino la
    comparación entre ellos y frente a la referencia industrial: ejecutarlos con la
    misma misión y semilla en una sola pasada elimina cualquier diferencia de montaje.

    param total_timesteps: Presupuesto de pasos para cada uno de los dos agentes.
    param mission: Perfil de misión común a los tres controladores.
    param seed: Semilla compartida por ambos entrenamientos.
    return: Tupla (resultados de SAC, resultados de TD3, resultados del FADEC).
    """
    logger.info("Comparativa SAC vs TD3 vs FADEC")

    results_sac = train_agent('sac', total_timesteps, mission, seed=seed)
    results_td3 = train_agent('td3', total_timesteps, mission, seed=seed)

    logger.info("Evaluando FADEC baseline en las mismas condiciones")
    from src.agents.fadec_baseline import FADECBaseline
    from src.environments.ace_env import ACEEnv
    fadec_env = ACEEnv(mission_profile=mission,
                       max_steps=DEFAULT_MAX_STEPS)
    fadec = FADECBaseline()
    results_fadec = fadec.evaluate(fadec_env,
                                   n_episodes=EVAL_EPISODES_FINAL)

    logger.info("Comparativa final:")
    logger.info(f"  {'Métrica':<20} {'SAC':>10} {'TD3':>10} {'FADEC':>10}")
    logger.info(f"  {'-' * 52}")
    logger.info(
        f"  {'Reward medio':<20} "
        f"{results_sac.get('mean_reward', 0):>10.2f} "
        f"{results_td3.get('mean_reward', 0):>10.2f} "
        f"{results_fadec.get('mean_reward', 0):>10.2f}")
    logger.info(
        f"  {'Tasa seguridad':<20} "
        f"{results_sac.get('safety_rate', 0):>10.1%} "
        f"{results_td3.get('safety_rate', 0):>10.1%} "
        f"{results_fadec.get('safety_rate', 0):>10.1%}")
    logger.info(
        f"  {'Violaciones/ep':<20} "
        f"{results_sac.get('mean_violations', 0):>10.2f} "
        f"{results_td3.get('mean_violations', 0):>10.2f} "
        f"{results_fadec.get('mean_violations', 0):>10.2f}")

    return results_sac, results_td3, results_fadec


def main() -> None:
    """Punto de entrada por línea de comandos del entrenamiento DRL.

    Analiza los argumentos y despacha hacia el entrenamiento de un solo algoritmo o
    hacia la comparativa completa. Concentrar aquí el parseo mantiene las funciones
    de entrenamiento libres de dependencias de la CLI y por tanto reutilizables desde
    un notebook o desde otro script.

    return: None; el resultado del entrenamiento se persiste en disco y se registra
        en el log.
    """
    parser = argparse.ArgumentParser(
        description="Entrenamiento DRL para el control del motor ACE")
    parser.add_argument('--algo', choices=['sac', 'td3', 'both'],
                        default='sac',
                        help="Algoritmo a entrenar. 'both' encadena SAC + TD3 "
                             "+ evaluación de FADEC en la misma misión.")
    parser.add_argument('--timesteps', type=int, default=500_000,
                        help="Presupuesto de pasos de entrenamiento (500k "
                             "reproduce los checkpoints activos)")
    parser.add_argument('--mission',
                        choices=['mixed', 'cruise', 'combat', 'full'],
                        default='mixed',
                        help="Perfil de misión ('mixed' reproduce los "
                             "checkpoints activos)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Semilla para el agente SB3 y el entorno ACE-v0")
    args = parser.parse_args()

    if args.algo == 'both':
        compare_sac_vs_td3(args.timesteps, args.mission, args.seed)
    else:
        train_agent(args.algo, args.timesteps, args.mission, seed=args.seed)


if __name__ == '__main__':
    main()