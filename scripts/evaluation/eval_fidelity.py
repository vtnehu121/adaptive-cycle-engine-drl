"""
Última Fecha de Modificación: 16/Aug/2026
Descripción eval_fidelity.py: Script que mide hasta qué punto las conclusiones
del estudio comparativo dependen de la fidelidad del modelo del motor. Somete al
mismo controlador FADEC, con la misma tabla de consulta y la misma semilla de
ruido de actuación, a dos motores distintos: el modelo analítico simplificado
que ACEEnv emplea como respaldo cuando se instancia con pinn_model=None, y el
gemelo digital BraytonPINN. Todo el protocolo está sembrado, de modo que las
cifras son reproducibles paso a paso entre ejecuciones.

Sostiene la Sección 6.5 y la Tabla 6.6 de la memoria; el desglose por fase de
misión que escribe alimenta la Tabla B.10 del Anexo B.

Salida:
    results/section_results/RESULTADOS_FIDELIDAD.json

Uso:
    python scripts/evaluation/eval_fidelity.py
"""

import json
import logging
import os
import sys
from pathlib import Path

# UTF-8 portable
if sys.platform == 'win32':
    os.system('')
    try:
        os.system('chcp 65001 >nul 2>&1')
    except Exception:
        pass

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.fadec_baseline import FADECBaseline
from src.environments.ace_env import ACEEnv
from src.models.pinn import BraytonPINN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger(__name__)

CHECKPOINT = PROJECT_ROOT / 'checkpoints' / 'digital_twin' / 'pinn.pt'
OUTPUT_PATH = (PROJECT_ROOT / 'results' / 'section_results'
               / 'RESULTADOS_FIDELIDAD.json')

N_EPISODES = 20
MAX_STEPS = 200
MISSION_PROFILE = 'mixed'
# Semilla base del experimento. El episodio i se reinicia con BASE_SEED + i, de
# modo que los dos modelos del motor afrontan exactamente la misma secuencia de
# misiones: sin esa igualdad la comparación mediría la suerte del muestreo de
# fases, no la fidelidad del modelo. Es la diferencia con plot_drl_section5.py,
# que promedia sobre misiones aleatorias porque allí lo que interesa es el
# rendimiento esperado y no la comparabilidad punto a punto.
BASE_SEED = 42
PHASES = ['takeoff', 'climb', 'cruise', 'combat', 'descent']


def load_pinn(checkpoint_path: Path) -> BraytonPINN:
    """Carga el gemelo digital deduciendo su arquitectura del propio checkpoint.

    Lee la anchura de la proyección de entrada y cuenta los bloques residuales
    presentes en las claves, en lugar de exigir que quien ejecuta el script
    recuerde con qué configuración se entrenó cada fichero.

    param checkpoint_path: Ruta del checkpoint .pt del PINN.
    return: Modelo BraytonPINN cargado y en modo evaluación.
    """
    state = torch.load(checkpoint_path, map_location='cpu')
    state = state.get('model_state_dict', state)
    hidden = state['input_proj.0.weight'].shape[0]
    n_layers = len({int(k.split('.')[1]) for k in state
                    if k.startswith('residual_blocks.')})
    model = BraytonPINN(hidden_dim=hidden, n_layers=n_layers)
    model.load_state_dict(state, strict=False)
    model.eval()
    logger.info(f"BraytonPINN {hidden} x {n_layers} cargado: "
                f"{sum(p.numel() for p in model.parameters()):,} parámetros")
    return model


def evaluate_engine(pinn_model, label: str) -> dict:
    """Ejecuta el FADEC contra un modelo del motor y agrega las métricas.

    Contabiliza la seguridad por paso y no por episodio, porque el propósito del
    experimento es localizar dónde y cuánto se sale el baseline de la envolvente,
    y una tasa por episodio saturaría a cero en cuanto un solo paso la violase.
    El desglose por fase se acumula en el mismo recorrido para no repetir los
    rollouts.

    param pinn_model: Gemelo digital con el que resolver el estado del motor, o
        None para emplear el modelo analítico simplificado interno de ACEEnv.
    param label: Nombre del modelo del motor, solo para el registro.
    return: Diccionario con recompensa media y su desviación, temperatura media y
        máxima, pasos inseguros sobre el total y las mismas métricas por fase.
    """
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)

    env = ACEEnv(pinn_model=pinn_model,
                 mission_profile=MISSION_PROFILE,
                 max_steps=MAX_STEPS)
    fadec = FADECBaseline(seed=BASE_SEED)

    episode_rewards, all_T4 = [], []
    unsafe_steps, total_steps = 0, 0
    phase_data = {p: {'thrust': [], 'sfc': [], 'T4': [], 'reward': []}
                  for p in PHASES}

    for episode in range(N_EPISODES):
        obs, info = env.reset(seed=BASE_SEED + episode)
        fadec.reset()
        ep_reward, done = 0.0, False

        while not done:
            action, _ = fadec.predict(obs, phase=info.get('phase', 'cruise'))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            total_steps += 1
            all_T4.append(info.get('T4', 0))
            if not info.get('safe', True):
                unsafe_steps += 1

            phase = info.get('phase', 'cruise')
            if phase in phase_data:
                phase_data[phase]['thrust'].append(info.get('thrust', 0))
                phase_data[phase]['sfc'].append(info.get('sfc', 0))
                phase_data[phase]['T4'].append(info.get('T4', 0))
                phase_data[phase]['reward'].append(reward)

        episode_rewards.append(ep_reward)

    results = {
        'engine':        label,
        'n_episodes':    N_EPISODES,
        'max_steps':     MAX_STEPS,
        'base_seed':     BASE_SEED,
        'mean_reward':   float(np.mean(episode_rewards)),
        'std_reward':    float(np.std(episode_rewards)),
        'T4_mean':       float(np.mean(all_T4)),
        'T4_max':        float(np.max(all_T4)),
        'unsafe_steps':  int(unsafe_steps),
        'total_steps':   int(total_steps),
        'safety_rate':   1 - unsafe_steps / total_steps,
        'phase_metrics': {
            p: {
                'n_steps':      len(d['thrust']),
                'mean_thrust':  float(np.mean(d['thrust'])),
                'mean_sfc':     float(np.mean(d['sfc'])),
                'mean_T4':      float(np.mean(d['T4'])),
                'max_T4':       float(np.max(d['T4'])),
                'mean_reward':  float(np.mean(d['reward'])),
                'unsafe_steps': int(sum(1 for t in d['T4'] if t >= 3200)),
            }
            for p, d in phase_data.items() if d['thrust']
        },
    }

    logger.info(f"  {label}: reward = {results['mean_reward']:+.2f} "
                f"± {results['std_reward']:.2f}, "
                f"T4 media = {results['T4_mean']:.0f} °R, "
                f"pasos inseguros = {unsafe_steps}/{total_steps} "
                f"(seguridad {results['safety_rate']:.1%})")
    return results


def log_comparison(results: dict) -> None:
    """Emite por el log la tabla comparativa y el desglose por fase.

    Deja en el log exactamente las mismas cifras que sustentan la Tabla 6.6 de la
    memoria y la Tabla B.10 del anexo, para poder contrastarlas sin abrir el JSON.

    param results: Diccionario {clave del motor: métricas} devuelto por
        evaluate_engine.
    return: None; la tabla se emite por el logger del módulo.
    """
    logger.info("")
    logger.info("Fidelidad del modelo del motor (Tabla 6.6):")
    logger.info(f"  {'Modelo del motor':<26} {'Reward':>11} {'Seguridad':>10} "
                f"{'T4 media':>10} {'Inseguros':>12}")
    for r in results.values():
        logger.info(f"  {r['engine']:<26} {r['mean_reward']:>+11.2f} "
                    f"{r['safety_rate']:>9.1%} {r['T4_mean']:>9.0f}° "
                    f"{r['unsafe_steps']:>6}/{r['total_steps']:<5}")

    logger.info("")
    logger.info("Desglose por fase de misión (Tabla B.10):")
    logger.info(f"  {'Fase':<10} {'Pasos':>6} {'Modelo':<24} "
                f"{'Empuje':>10} {'SFC':>7} {'T4':>8} {'T4 max':>8} {'R/paso':>9}")
    for phase in PHASES:
        for key, r in results.items():
            m = r['phase_metrics'].get(phase)
            if m is None:
                continue
            etiqueta = phase if key == list(results)[0] else ''
            pasos = f"{m['n_steps']}" if key == list(results)[0] else ''
            logger.info(f"  {etiqueta:<10} {pasos:>6} {r['engine']:<24} "
                        f"{m['mean_thrust']:>10.0f} {m['mean_sfc']:>7.3f} "
                        f"{m['mean_T4']:>8.0f} {m['max_T4']:>8.0f} "
                        f"{m['mean_reward']:>+9.3f}")


def main() -> None:
    """Punto de entrada que compara los dos modelos del motor y escribe el JSON.

    Aborta si falta el checkpoint del gemelo digital, porque sin él el
    experimento pierde su término de comparación y el modelo analítico por sí
    solo no responde a la pregunta que el script plantea.

    return: None; deja el resultado en results/section_results/RESULTADOS_FIDELIDAD.json.
    """
    logger.info("Fidelidad del modelo del motor: FADEC contra dos motores")
    logger.info(f"Protocolo: {N_EPISODES} episodios de {MAX_STEPS} pasos, "
                f"perfil {MISSION_PROFILE}, semillas de entorno "
                f"{BASE_SEED} – {BASE_SEED + N_EPISODES - 1}")

    if not CHECKPOINT.exists():
        logger.error(f"Checkpoint no encontrado: {CHECKPOINT}. "
                     "Ejecutar src/training/train_pinn.py")
        return

    pinn = load_pinn(CHECKPOINT)

    results = {
        'analytical': evaluate_engine(None, 'Analítico simplificado'),
        'pinn':       evaluate_engine(pinn, 'BraytonPINN 224 x 8'),
    }

    log_comparison(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    logger.info("")
    logger.info(f"Guardado: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
