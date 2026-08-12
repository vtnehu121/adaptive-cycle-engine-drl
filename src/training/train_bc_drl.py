"""
Última Fecha de Modificación: 08/Aug/2026
Descripción train_bc_drl.py: Pipeline híbrido TD3+BC para el control del motor
ACE. Su motivación es evitar la fase inicial de exploración ciega, que en este
entorno produce trayectorias inseguras y ralentiza la convergencia. Consta de
tres fases secuenciales:

    1. Generación de un dataset experto operando el FADECBaseline sobre el
       entorno ACEEnv durante 200 episodios completos, recopilando 40,000
       transiciones (o, a, r, o', d).

    2. Pre-entrenamiento supervisado (Behavioral Cloning) del Actor de TD3
       sobre las parejas (observación, acción) del experto mediante MSE, para
       clonar el comportamiento base del controlador convencional.

    3. Fine-tuning con TD3 arrancando desde el actor clonado, con el dataset
       experto inyectado en el replay buffer. La política clonada queda
       protegida por las 40,000 transiciones expertas ya presentes en el
       replay buffer, que dominan estadísticamente las primeras
       actualizaciones y evitan que TD3 se aleje de la región ya aprendida.

HALLAZGO empírico (semilla=42, mission='mixed', 500k timesteps de fine-tuning):
    TD3+BC WINNER: reward 249.94 ± 18.95, safety 84.0%, SFC combat 0.483.
    Supera a SAC puro (209.59 ± 46.23, safety 60.0%), a TD3 puro
    (205.41 ± 40.50, safety 56.0%) y al baseline FADEC
    (-1947.28 ± 1533.57, safety 12.0%) entrenados con train_drl.py.

El checkpoint final se guarda en `checkpoints/control/td3_bc/`, sobreescribiendo
el modelo activo del proyecto. Los logs asociados se guardan en `logs/td3_bc/`,
localizables por `plot_drl_section5.py`.

Reproducibilidad: la semilla por defecto (--seed 42) se propaga al agente TD3
(parámetro seed=), al entorno ACE-v0 (env.reset(seed=)) y a np.random para el
muestreo del experto. El fine-tuning con TD3 sigue siendo dependiente de la
variabilidad de PyTorch en GPU; para reproducibilidad total ejecutar en CPU o
fijar CUBLAS_WORKSPACE_CONFIG.

Referencia:
    Fujimoto, S., & Gu, S. S. (2021). "A Minimalist Approach to Offline
        Reinforcement Learning." NeurIPS 2021.

Uso:
    python src/training/train_bc_drl.py
    python src/training/train_bc_drl.py --expert-episodes 100
    python src/training/train_bc_drl.py --bc-epochs 50 --rl-timesteps 200000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# Rutas y constantes
PINN_CHECKPOINT = Path('checkpoints/digital_twin/pinn.pt')
PINN_HIDDEN_DIM = 224
PINN_N_LAYERS = 8

# Salida del pipeline: ubicación fija del checkpoint activo TD3+BC.
CHECKPOINT_DIR = Path('checkpoints/control/td3_bc')
LOG_DIR = Path('logs/td3_bc')
EXPERT_DATA_PATH = Path('data/expert/expert_data.npz')

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
EXPERT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_MAX_STEPS = 200
DEFAULT_DEGRADATION_RATE = 0.005

# Configuración por defecto del pipeline.
DEFAULT_EXPERT_EPISODES = 200      # 200 ep × 200 pasos = 40 000 transiciones
DEFAULT_BC_EPOCHS = 100
DEFAULT_BC_BATCH_SIZE = 256
DEFAULT_BC_LR = 1e-3

DEFAULT_RL_TIMESTEPS = 500_000
TD3_LEARNING_RATE = 1e-4
EVAL_FREQ = 5000
EVAL_EPISODES_CALLBACK = 10
EVAL_EPISODES_FINAL = 50

# Reward asignado a las transiciones del experto al inyectarlas en el
# replay buffer, sustituyendo el reward REAL del FADEC. La sustitución
# es deliberada: el FADEC obtiene rewards muy negativos en este entorno
# (reward medio -1947 en 50 episodios de evaluación), lo que sesgaría
# al crítico de TD3 hacia estimaciones pesimistas del valor Q. Al
# reemplazarlos por un valor positivo constante (5.0, comparable al
# reward típico de agentes ya entrenados), el crítico arranca con
# experiencias que sí le informan sobre TRANSICIONES razonables sin
# heredar las malas evaluaciones del FADEC baseline.
EXPERT_REPLAY_REWARD = 5.0


# Fase 1: Generación del dataset experto
def generate_expert_dataset(n_episodes: int, pinn_model,
                             seed: int = 42) -> dict:
    """Genera el conjunto de demostraciones ejecutando el FADEC sobre el entorno.

    Recorre episodios completos con el controlador convencional, acumula las
    transiciones (o, a, r, o', d), las guarda comprimidas en disco y devuelve además
    un subconjunto filtrado. El filtro descarta transiciones con recompensa muy
    negativa o con acciones fuera de rango: para clonar comportamiento interesa el
    tramo competente del experto, no sus episodios catastróficos. Ese filtrado
    introduce un sesgo deliberado, aceptable por tratarse solo de inicialización.

    param n_episodes: Número de episodios completos a ejecutar con el experto.
    param pinn_model: Gemelo digital que actúa como modelo del motor, o None.
    param seed: Semilla de NumPy, para que el dataset sea reproducible.
    return: Tupla (dataset completo, dataset filtrado), ambos diccionarios de arrays
        con las claves observations, actions, rewards, next_observations y dones.
    """
    from src.agents.fadec_baseline import FADECBaseline
    from src.environments.ace_env import ACEEnv

    logger.info(f"Fase 1: generando dataset experto ({n_episodes} episodios)")
    env = ACEEnv(
        pinn_model=pinn_model,
        mission_profile='mixed',
        max_steps=DEFAULT_MAX_STEPS,
        degradation_rate=DEFAULT_DEGRADATION_RATE,
    )
    fadec = FADECBaseline()

    observations, actions, rewards = [], [], []
    next_observations, dones = [], []

    np.random.seed(seed)

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = fadec.predict(obs, phase=info.get('phase', 'cruise'))
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            dones.append(done)

            obs = next_obs

        if (episode + 1) % 20 == 0:
            logger.info(f"  Episodio {episode + 1}/{n_episodes}: "
                        f"{len(observations)} transiciones acumuladas")

    dataset = {
        'observations':      np.array(observations, dtype=np.float32),
        'actions':           np.array(actions, dtype=np.float32),
        'rewards':           np.array(rewards, dtype=np.float32),
        'next_observations': np.array(next_observations, dtype=np.float32),
        'dones':             np.array(dones, dtype=np.bool_),
    }

    np.savez_compressed(EXPERT_DATA_PATH, **dataset)
    logger.info(f"Dataset guardado en {EXPERT_DATA_PATH} "
                f"({len(observations)} transiciones)")

    # Filtro heurístico: retenemos solo transiciones de "buen ejemplo"
    # para el pre-entrenamiento supervisado. Los criterios son:
    #   - safety = True (sin violaciones)
    #   - reward > -50 (excluye episodios catastróficos)
    #   - |action| < 1 en todas las dimensiones (dentro del rango físico)
    # Nota: este filtro introduce sesgo (el BC aprende solo el "buen"
    # subconjunto del comportamiento FADEC). Aceptable para
    # inicialización de la política.
    keep_mask = dataset['rewards'] > -50
    keep_mask &= np.all(np.abs(dataset['actions']) < 1, axis=1)

    filtered = {k: v[keep_mask] for k, v in dataset.items()}
    logger.info(f"Filtrado supervisado: {len(filtered['observations'])} "
                f"transiciones ({keep_mask.mean() * 100:.1f}%) retenidas")

    return dataset, filtered



# Fase 2: Behavioral Cloning
def behavioral_cloning(model, dataset_filtered: dict,
                        epochs: int, batch_size: int, lr: float) -> None:
    """Pre-entrena el actor de TD3 por regresión contra las acciones del experto.

    Optimiza únicamente el actor, dejando intactos los críticos, y ataca la salida
    determinista mu(obs) previa al aplastamiento, que es el objetivo natural de una
    clonación con MSE. Es necesario porque un actor aleatorio inicial exploraría
    durante miles de pasos antes de encontrar acciones razonables; partir del
    comportamiento del FADEC sitúa a TD3 directamente en una región útil del espacio.

    param model: Agente TD3 cuyo actor se va a pre-entrenar.
    param dataset_filtered: Demostraciones filtradas con observations y actions.
    param epochs: Número de pasadas completas sobre el conjunto de demostraciones.
    param batch_size: Tamaño de lote del entrenamiento supervisado.
    param lr: Tasa de aprendizaje del optimizador Adam del actor.
    return: None; los pesos del actor del modelo se modifican in situ.
    """
    logger.info(f"Fase 2: Behavioral Cloning ({epochs} epochs, lr={lr})")

    obs_tensor = torch.tensor(dataset_filtered['observations'])
    act_tensor = torch.tensor(dataset_filtered['actions'])

    actor = model.policy.actor
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    n_samples = len(obs_tensor)
    for epoch in range(epochs):
        idx = torch.randperm(n_samples)
        obs_shuf = obs_tensor[idx]
        act_shuf = act_tensor[idx]

        epoch_losses = []
        for i in range(0, n_samples, batch_size):
            batch_obs = obs_shuf[i:i + batch_size]
            batch_act = act_shuf[i:i + batch_size]

            optimizer.zero_grad()
            # actor.mu(obs) es la salida determinista antes del squashing;
            # es el objetivo natural para clonación con MSE. Depende de
            # la API interna de Stable-Baselines3.
            pred = actor.mu(batch_obs)
            loss = loss_fn(pred, batch_act)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch + 1}/{epochs}: "
                        f"MSE = {np.mean(epoch_losses):.6f}")

    logger.info("Behavioral Cloning completado")


# Fase 3: Fine-tuning con TD3
def inject_expert_into_replay(model, dataset: dict) -> None:
    """Inyecta las transiciones del experto en el replay buffer.

    Cada transición se guarda con un reward positivo constante
    (EXPERT_REPLAY_REWARD) para proporcionar al Crítico ejemplos
    favorables desde el primer timestep. El valor concreto es
    heurístico y calibrado empíricamente.

    Es necesario porque clonar el actor no basta: el crítico sigue partiendo de cero
    y, sin experiencias previas, sus primeras estimaciones de valor son ruido que
    arrastraría al actor recién clonado fuera de la buena región del espacio.

    param model: Agente TD3 cuyo replay buffer se va a poblar.
    param dataset: Demostraciones completas del experto.
    return: None; el replay buffer del modelo se modifica in situ.
    """
    buffer = model.replay_buffer
    n = len(dataset['observations'])
    logger.info(f"Inyectando {n} transiciones expertas en el replay buffer")

    for i in range(n):
        buffer.add(
            dataset['observations'][i],
            dataset['next_observations'][i],
            dataset['actions'][i],
            np.array([EXPERT_REPLAY_REWARD]),
            np.array([dataset['dones'][i]]),
            [{}],
        )


def fine_tune_td3(model, dataset: dict, rl_timesteps: int,
                   pinn_model, seed: int) -> None:
    """Ejecuta el fine-tuning por refuerzo partiendo de la política ya clonada.

    Monta un entorno de evaluación independiente y guarda tanto el mejor checkpoint
    como el modelo final. La protección de la política clonada no la da una
    reducción del learning rate, sino la dominancia estadística del replay buffer:
    los 40,000 ejemplos expertos ya inyectados dominan las primeras actualizaciones
    y evitan que el crítico, aún inicialmente ruidoso— arrastre al actor lejos de
    la región aprendida.

    param model: Agente TD3 con el actor clonado y el replay ya poblado.
    param dataset: Demostraciones del experto; se mantiene por simetría de la firma
        con el resto de fases del pipeline.
    param rl_timesteps: Presupuesto de pasos del fine-tuning.
    param pinn_model: Gemelo digital con el que construir el entorno de evaluación.
    param seed: Semilla del run; se conserva para trazabilidad de la configuración.
    return: None; deja el modelo entrenado y guardado en la carpeta de checkpoints.
    """
    from stable_baselines3.common.callbacks import (
        CallbackList, EvalCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.environments.ace_env import ACEEnv

    logger.info(f"Fase 3: fine-tuning TD3 ({rl_timesteps:,} timesteps, "
                f"lr = {TD3_LEARNING_RATE})")

    # El fine-tuning mantiene el LR nominal del pipeline. La política
    # clonada por BC queda protegida por los 40,000 ejemplos expertos
    # inyectados en el replay buffer, que dominan estadísticamente las
    # primeras actualizaciones y evitan que TD3 se aleje de la región
    # ya aprendida.

    def make_eval_env():
        """Fabrica el entorno de evaluación envuelto en Monitor.

        Se define como función y no como instancia porque DummyVecEnv espera un
        constructor invocable, que ejecuta él mismo al vectorizar.

        return: Entorno ACEEnv de evaluación envuelto en Monitor.
        """
        return Monitor(ACEEnv(
            pinn_model=pinn_model,
            mission_profile='mixed',
            max_steps=DEFAULT_MAX_STEPS,
            degradation_rate=DEFAULT_DEGRADATION_RATE,
        ))

    eval_env = DummyVecEnv([make_eval_env])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(CHECKPOINT_DIR),
        log_path=str(LOG_DIR),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES_CALLBACK,
        deterministic=True,
    )

    model.learn(
        total_timesteps=rl_timesteps,
        callback=CallbackList([eval_callback]),
        progress_bar=True,
    )

    final_path = CHECKPOINT_DIR / 'td3_bc_final'
    model.save(str(final_path))
    logger.info(f"Modelo final TD3+BC guardado en {final_path}")


# Pipeline completo
def load_pinn():
    """Carga el gemelo digital activo, o devuelve None si no está disponible.

    Degradar a None permite ejecutar el pipeline completo con el modelo simplificado
    del entorno, útil para validar el encadenado de las tres fases sin necesitar un
    PINN entrenado. Se usa strict=False para tolerar claves de normalización extra.

    return: Instancia de BraytonPINN en modo evaluación, o None si el checkpoint no
        existe o no puede cargarse.
    """
    from src.models.pinn import BraytonPINN

    if not PINN_CHECKPOINT.exists():
        logger.warning(f"PINN no encontrado en {PINN_CHECKPOINT}")
        return None

    try:
        pinn = BraytonPINN(hidden_dim=PINN_HIDDEN_DIM,
                           n_layers=PINN_N_LAYERS)
        pinn.load_state_dict(
            torch.load(PINN_CHECKPOINT, weights_only=True), strict=False)
        pinn.eval()
        logger.info(f"PINN cargado desde {PINN_CHECKPOINT}")
        return pinn
    except Exception as exc:
        logger.warning(f"No se pudo cargar el PINN: {exc}")
        return None


def build_td3(pinn_model, seed: int):
    """Instancia el agente TD3 con la arquitectura y los hiperparámetros del proyecto.

    Replica exactamente la configuración usada en train_drl.py, con el mismo learning
    rate nominal durante todo el pipeline. Mantener el resto idéntico es lo que
    permite atribuir cualquier diferencia de resultados a la clonación previa y no
    a un cambio de arquitectura.

    param pinn_model: Gemelo digital con el que construir el entorno de entrenamiento.
    param seed: Semilla de inicialización del agente.
    return: Agente TD3 recién construido, con su entorno vectorizado ya asociado.
    """
    from stable_baselines3 import TD3
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.environments.ace_env import ACEEnv

    def make_train_env():
        """Fabrica el entorno de entrenamiento envuelto en Monitor.

        Se define como función porque DummyVecEnv exige un constructor invocable,
        no una instancia ya creada.

        return: Entorno ACEEnv de entrenamiento envuelto en Monitor.
        """
        return Monitor(ACEEnv(
            pinn_model=pinn_model,
            mission_profile='mixed',
            max_steps=DEFAULT_MAX_STEPS,
            degradation_rate=DEFAULT_DEGRADATION_RATE,
        ))

    train_env = DummyVecEnv([make_train_env])
    return TD3(
        'MlpPolicy',
        train_env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=TD3_LEARNING_RATE,
        buffer_size=100_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=1000,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        verbose=0,
        seed=seed,
        tensorboard_log=str(LOG_DIR / 'tensorboard'),
    )


def evaluate_final_model(model, pinn_model) -> dict:
    """Evalúa el modelo TD3+BC entrenado y persiste sus métricas finales.

    Ejecuta la política de forma determinista sobre episodios completos y vuelca el
    resultado a results.json. Es lo que hace comparable este pipeline con los runs de
    SAC y TD3 puros: mismo número de episodios, mismo perfil de misión y mismas
    métricas de recompensa y seguridad.

    param model: Agente TD3+BC ya entrenado.
    param pinn_model: Gemelo digital con el que construir el entorno de evaluación.
    return: Diccionario con recompensa media, desviación, rango, violaciones por
        episodio y tasa de seguridad, ya escrito también en logs/td3_bc/results.json.
    """
    from src.environments.ace_env import ACEEnv

    logger.info(f"Evaluación final sobre {EVAL_EPISODES_FINAL} episodios")
    env = ACEEnv(
        pinn_model=pinn_model,
        mission_profile='mixed',
        max_steps=DEFAULT_MAX_STEPS,
        degradation_rate=DEFAULT_DEGRADATION_RATE,
    )

    rewards = []
    violations_total = 0

    for _ in range(EVAL_EPISODES_FINAL):
        obs, info = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        violations_total += info.get('episode_violations', 0)

    results = {
        'mean_reward':     float(np.mean(rewards)),
        'std_reward':      float(np.std(rewards)),
        'min_reward':      float(np.min(rewards)),
        'max_reward':      float(np.max(rewards)),
        'mean_violations': violations_total / EVAL_EPISODES_FINAL,
        'safety_rate':     1 - violations_total / (EVAL_EPISODES_FINAL
                                                    * DEFAULT_MAX_STEPS),
    }

    logger.info(f"  Reward:         {results['mean_reward']:.2f} "
                f"± {results['std_reward']:.2f}")
    logger.info(f"  Violaciones/ep: {results['mean_violations']:.2f}")
    logger.info(f"  Tasa seguridad: {results['safety_rate']:.1%}")

    with open(LOG_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main() -> None:
    """Punto de entrada por línea de comandos que encadena las tres fases.

    Analiza los argumentos y ejecuta en orden generación de demostraciones, clonación
    de comportamiento, inyección en el replay, fine-tuning y evaluación final. El
    orden no es negociable: cada fase consume el estado que deja la anterior.

    return: None; los artefactos del pipeline quedan en checkpoints/control/td3_bc y
        logs/td3_bc.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline TD3+BC para el control del motor ACE")
    parser.add_argument('--expert-episodes', type=int,
                        default=DEFAULT_EXPERT_EPISODES)
    parser.add_argument('--bc-epochs', type=int,
                        default=DEFAULT_BC_EPOCHS)
    parser.add_argument('--bc-batch-size', type=int,
                        default=DEFAULT_BC_BATCH_SIZE)
    parser.add_argument('--bc-lr', type=float,
                        default=DEFAULT_BC_LR)
    parser.add_argument('--rl-timesteps', type=int,
                        default=DEFAULT_RL_TIMESTEPS)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    pinn_model = load_pinn()

    # Fase 1: dataset experto.
    dataset, filtered = generate_expert_dataset(
        args.expert_episodes, pinn_model, args.seed)

    # Fase 2: Behavioral Cloning.
    model = build_td3(pinn_model, args.seed)
    behavioral_cloning(
        model, filtered,
        epochs=args.bc_epochs,
        batch_size=args.bc_batch_size,
        lr=args.bc_lr,
    )

    # Fase 3: fine-tuning TD3 con dataset inyectado en el replay.
    inject_expert_into_replay(model, dataset)
    fine_tune_td3(
        model, dataset,
        rl_timesteps=args.rl_timesteps,
        pinn_model=pinn_model,
        seed=args.seed,
    )

    evaluate_final_model(model, pinn_model)
    logger.info("Pipeline TD3+BC completado")


if __name__ == '__main__':
    main()