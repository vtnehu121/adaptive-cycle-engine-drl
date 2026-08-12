"""
Última Fecha de Modificación: 08/Aug/2026
Descripción plot_drl_section5.py: Script que produce la evaluación cuantitativa
y las figuras de la Sección 5 de la memoria, donde se compara el controlador
convencional FADEC con los tres agentes de refuerzo (SAC, TD3 y TD3+BC). Carga
los checkpoints activos, ejecuta 50 episodios deterministas por controlador
sobre el entorno ACE-v0 y produce el JSON `results/section_results/
RESULTADOS_SECCION_5.json` junto con cinco figuras:

    fig19_training_curves.png    Convergencia del reward SAC/TD3/TD3+BC
    fig20_drl_vs_fadec.png       Mejora porcentual DRL frente a FADEC
    fig21_mission_profile.png    Perfil de misión completo comparado
    fig22_reward_by_phase.png    Rendimiento por fase de misión
    fig23_actuator_actions.png   Trayectorias de los 4 actuadores

El script no reentrena en ningún momento; opera únicamente en modo inferencia
sobre los agentes ya entrenados. Debido a la aleatoriedad inherente de los
rollouts DRL, ejecutar este script tras el checkpoint puede producir pequeñas
variaciones numéricas frente a las figuras originalmente reportadas en el TFG.

Uso:
    python scripts/plotting/plot_drl_section5.py
"""

import json
import logging
import sys
from pathlib import Path
import os

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

PINN_CHECKPOINT = Path('checkpoints/digital_twin/pinn.pt')

CONTROL_ROOT = Path('checkpoints/control')
TD3_BC_CHECKPOINT = CONTROL_ROOT / 'td3_bc' / 'best_model.zip'
SAC_CHECKPOINT = CONTROL_ROOT / 'sac_mixed' / 'best_model.zip'
TD3_CHECKPOINT = CONTROL_ROOT / 'td3_mixed' / 'best_model.zip'

LOGS_ROOT = Path('logs')
LOG_DIRS = {
    'SAC':    LOGS_ROOT / 'sac_mixed',
    'TD3':    LOGS_ROOT / 'td3_mixed',
    'TD3+BC': LOGS_ROOT / 'td3_bc',
}

FIG_DIR = Path('results/figures')
RESULTS_JSON = Path('results/section_results/RESULTADOS_SECCION_5.json')
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)

MISSION_PROFILE = 'mixed'
MISSION_PROFILE_FULL = 'full'
DEFAULT_MAX_STEPS = 200
N_EVAL_EPISODES = 50

PINN_HIDDEN_DIM = 224
PINN_N_LAYERS = 8

COLORS = {
    'SAC':    '#E63946',
    'TD3':    '#F4A261',
    'TD3+BC': '#2A9D8F',
    'FADEC':  '#457B9D',
}

PHASE_COLORS = {
    'takeoff': '#FFE0B2',
    'climb':   '#C8E6C9',
    'cruise':  '#BBDEFB',
    'combat':  '#FFCDD2',
    'descent': '#E1BEE7',
}
PHASES = ['takeoff', 'climb', 'cruise', 'combat', 'descent']

plt.rcParams.update({
    'figure.dpi':        150,
    'font.size':         12,
    'font.family':       'serif',
    'axes.titlesize':    13,
    'axes.labelsize':    12,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


# Carga de modelos
def load_pinn():
    """Carga el gemelo digital que hará de modelo del motor en la evaluación.

    Si no está disponible devuelve None y el entorno recurre a su modelo
    simplificado, lo que permite que el script se ejecute de principio a fin para
    verificar el flujo, aunque las cifras resultantes no sean las de la memoria.

    return: Instancia de BraytonPINN en modo evaluación, o None si el checkpoint no
        existe o no puede cargarse.
    """
    if not PINN_CHECKPOINT.exists():
        logger.warning(f"PINN no encontrado en {PINN_CHECKPOINT}")
        return None
    try:
        pinn = BraytonPINN(hidden_dim=PINN_HIDDEN_DIM,
                           n_layers=PINN_N_LAYERS)
        pinn.load_state_dict(
            torch.load(PINN_CHECKPOINT, weights_only=False), strict=False)
        pinn.eval()
        logger.info(f"PINN cargado desde {PINN_CHECKPOINT}")
        return pinn
    except Exception as exc:
        logger.warning(f"No se pudo cargar el PINN: {exc}")
        return None


def load_td3_bc():
    """Carga el agente híbrido TD3+BC desde su checkpoint activo.

    Tolera su ausencia devolviendo None, de modo que la comparativa pueda generarse
    aunque falte una de las variantes: la figura simplemente mostrará una columna
    menos en lugar de abortar la ejecución completa.

    return: Agente TD3 cargado, o None si el checkpoint no existe o falla la carga.
    """
    from stable_baselines3 import TD3
    if not TD3_BC_CHECKPOINT.exists():
        logger.warning(f"TD3+BC no encontrado en {TD3_BC_CHECKPOINT}")
        return None
    try:
        model = TD3.load(str(TD3_BC_CHECKPOINT))
        logger.info(f"TD3+BC cargado desde {TD3_BC_CHECKPOINT}")
        return model
    except Exception as exc:
        logger.warning(f"No se pudo cargar TD3+BC: {exc}")
        return None


def load_td3():
    """Carga el agente TD3 entrenado desde cero, sin clonación previa.

    Es la referencia que permite aislar la aportación del pre-entrenamiento por
    Behavior Cloning: sin él no podría distinguirse cuánto del rendimiento de TD3+BC procede
    del algoritmo y cuánto de haber partido del comportamiento del FADEC.

    return: Agente TD3 cargado, o None si el checkpoint no existe o falla la carga.
    """
    from stable_baselines3 import TD3
    if not TD3_CHECKPOINT.exists():
        logger.warning(f"TD3 puro no encontrado en {TD3_CHECKPOINT}")
        return None
    try:
        model = TD3.load(str(TD3_CHECKPOINT))
        logger.info(f"TD3 puro cargado desde {TD3_CHECKPOINT}")
        return model
    except Exception as exc:
        logger.warning(f"No se pudo cargar TD3 puro: {exc}")
        return None


def load_sac():
    """Carga el agente SAC entrenado sobre el perfil de misión mixto.

    Aporta a la comparativa el punto de vista de una política estocástica con
    regularización por entropía, frente al carácter determinista de TD3: comparar
    ambas familias es parte de la discusión metodológica de la Sección 5.

    return: Agente SAC cargado, o None si el checkpoint no existe o falla la carga.
    """
    from stable_baselines3 import SAC
    if not SAC_CHECKPOINT.exists():
        logger.warning(f"SAC no encontrado en {SAC_CHECKPOINT}")
        return None
    try:
        model = SAC.load(str(SAC_CHECKPOINT))
        logger.info(f"SAC cargado desde {SAC_CHECKPOINT}")
        return model
    except Exception as exc:
        logger.warning(f"No se pudo cargar SAC: {exc}")
        return None


def load_all_controllers() -> tuple:
    """Carga PINN + los cuatro controladores. Cualquiera ausente se ignora.

    El orden de inserción (SAC → TD3 → TD3+BC → FADEC) importa porque determina el
    orden de columnas y barras en todas las figuras, y con él la lectura de la
    comparación. Se sigue la progresión narrativa: primero los agentes
    tabula rasa (SAC, TD3), después el híbrido con pre-entrenamiento (TD3+BC), y
    finalmente el baseline clásico (FADEC) como referencia. El FADEC se instancia
    siempre, ya que no depende de ningún checkpoint y es la referencia sin la cual
    la comparativa carecería de sentido.

    return: Tupla (diccionario {nombre: controlador} en el orden narrativo, gemelo
        digital cargado o None).
    """
    pinn = load_pinn()
    controllers = {}
    sac = load_sac()
    if sac is not None:
        controllers['SAC'] = sac
    td3 = load_td3()
    if td3 is not None:
        controllers['TD3'] = td3
    td3bc = load_td3_bc()
    if td3bc is not None:
        controllers['TD3+BC'] = td3bc
    controllers['FADEC'] = FADECBaseline()
    logger.info("FADEC baseline instanciado")
    return controllers, pinn


# Evaluación
def run_episode(env, controller, name: str,
                deterministic: bool = True,
                seed: int = None) -> dict:
    """Ejecuta un episodio completo registrando la traza paso a paso.

    Resuelve internamente la diferencia de interfaz entre el FADEC, que necesita
    conocer la fase de misión explícitamente, y los agentes de Stable-Baselines3, que
    solo reciben la observación. Esa uniformización es lo que permite que todas las
    figuras y métricas se construyan con una única función de rollout, evitando que
    controladores distintos se evalúen con bucles distintos.

    param env: Entorno ACEEnv sobre el que ejecutar el episodio.
    param controller: Controlador a evaluar, agente o baseline.
    param name: Nombre del controlador, que decide cómo se le consulta la acción.
    param deterministic: Si es True la política se evalúa sin exploración.
    param seed: Semilla del reinicio; si es None el episodio es aleatorio.
    return: Diccionario con las listas de pasos, fases, recompensas, empuje, consumo,
        temperatura de turbina, acciones y estado de seguridad de cada paso.
    """
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()
    history = {
        'steps':   [],
        'phases':  [],
        'rewards': [],
        'thrust':  [],
        'sfc':     [],
        'T4':      [],
        'actions': [],
        'safe':    [],
    }
    step = 0
    done = False
    while not done:
        if name == 'FADEC':
            action, _ = controller.predict(
                obs, phase=info.get('phase', 'cruise'))
        else:
            action, _ = controller.predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        history['steps'].append(step)
        history['phases'].append(info.get('phase', 'cruise'))
        history['rewards'].append(reward)
        history['thrust'].append(info.get('thrust', 0))
        history['sfc'].append(info.get('sfc', 0))
        history['T4'].append(info.get('T4', 0))
        history['actions'].append(action.copy())
        history['safe'].append(info.get('safe', True))
        step += 1

    return history


def evaluate_controllers(controllers: dict, pinn,
                          n_episodes: int = N_EVAL_EPISODES) -> dict:
    """Evalúa cada controlador en n_episodes y agrega métricas por fase.

    Deliberadamente no se fija la semilla: con una semilla concreta el baseline
    FADEC caía siempre en las mismas misiones desfavorables y su resultado dejaba
    de ser representativo, exagerando la ventaja del DRL. Promediar sobre misiones
    aleatorias mide lo que interesa, que es el rendimiento esperado a lo largo de
    la distribución de misiones. El desglose por fase se acumula en el mismo
    recorrido para no tener que repetir los rollouts.

    param controllers: Diccionario {nombre: controlador} a evaluar.
    param pinn: Gemelo digital con el que construir los entornos de evaluación.
    param n_episodes: Número de episodios por controlador.
    return: Diccionario {nombre: métricas} con recompensa media y desviación,
        episodios con violación, tasa de seguridad y métricas medias por fase.
    """
    results = {}
    for name, controller in controllers.items():
        logger.info(f"Evaluando {name} sobre {n_episodes} episodios...")
        env = ACEEnv(pinn_model=pinn,
                      mission_profile=MISSION_PROFILE,
                      max_steps=DEFAULT_MAX_STEPS)
        episode_rewards = []
        violations = 0
        phase_data = {p: {'thrust': [], 'sfc': [], 'reward': []}
                      for p in PHASES}

        for _ in range(n_episodes):
            history = run_episode(env, controller, name)
            episode_rewards.append(sum(history['rewards']))
            if not all(history['safe']):
                violations += 1
            for i, phase in enumerate(history['phases']):
                if phase in phase_data:
                    phase_data[phase]['thrust'].append(history['thrust'][i])
                    phase_data[phase]['sfc'].append(history['sfc'][i])
                    phase_data[phase]['reward'].append(history['rewards'][i])

        phase_summary = {}
        for phase, data in phase_data.items():
            if data['thrust']:
                phase_summary[phase] = {
                    'mean_thrust': float(np.mean(data['thrust'])),
                    'mean_sfc':    float(np.mean(data['sfc'])),
                    'mean_reward': float(np.mean(data['reward'])),
                }

        results[name] = {
            'mean_reward':   float(np.mean(episode_rewards)),
            'std_reward':    float(np.std(episode_rewards)),
            'violations':    violations,
            'safety_rate':   1 - violations / n_episodes,
            'phase_metrics': phase_summary,
        }

        logger.info(f"  {name}: reward = {results[name]['mean_reward']:.2f} "
                    f"+/- {results[name]['std_reward']:.2f}, "
                    f"seguridad = {results[name]['safety_rate']:.1%}")

    return results


# Figuras
def fig19_training_curves(output_path: Path) -> None:
    """Dibuja las curvas de convergencia de la recompensa de los tres agentes.

    Lee los archivos de evaluación periódica que dejó el entrenamiento y superpone la
    media con una banda de una desviación típica. Es necesario porque el rendimiento
    final no cuenta toda la historia: la figura muestra además con qué rapidez llega
    cada agente y con cuánta estabilidad, que es donde se aprecia la ventaja de haber
    partido de una política clonada. Si no hay registros disponibles se omite con un
    aviso, en lugar de generar una figura vacía.

    param output_path: Ruta del archivo PNG a generar.
    return: None; la figura queda escrita en output_path, salvo que no haya datos.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    plotted_any = False
    for algo in ('SAC', 'TD3', 'TD3+BC'):
        log_dir = LOG_DIRS.get(algo)
        if log_dir is None or not log_dir.exists():
            continue
        eval_file = log_dir / 'evaluations.npz'
        if not eval_file.exists():
            continue
        data = np.load(str(eval_file))
        timesteps = data['timesteps']
        results = data['results']
        mean_reward = results.mean(axis=1)
        std_reward = results.std(axis=1)
        color = COLORS[algo]
        ax.plot(timesteps, mean_reward, color=color, linewidth=2, label=algo)
        ax.fill_between(timesteps, mean_reward - std_reward,
                        mean_reward + std_reward, alpha=0.2, color=color)
        plotted_any = True
    if not plotted_any:
        logger.warning("No hay evaluations.npz disponibles; se omite fig19")
        plt.close()
        return
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Reward medio')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {output_path}")


def fig20_drl_vs_fadec(results: dict, output_path: Path) -> None:
    """Dibuja la mejora porcentual de cada agente respecto al baseline FADEC.

    Tres paneles con las tres dimensiones de la comparación: recompensa, seguridad y
    consumo en crucero, todos referidos al FADEC como cero. Expresarlo como mejora
    relativa y no en valor absoluto es lo que hace la figura legible, ya que las tres
    magnitudes tienen unidades y órdenes completamente distintos. Se omite si el
    baseline no ha sido evaluado, pues sin él no hay referencia.

    param results: Diccionario de métricas por controlador.
    param output_path: Ruta del archivo PNG a generar.
    return: None; la figura queda escrita en output_path, salvo que falte el FADEC.
    """
    if 'FADEC' not in results:
        logger.warning("FADEC no evaluado; se omite fig20")
        return
    fadec_reward = results['FADEC']['mean_reward']
    fadec_safety = results['FADEC']['safety_rate']
    fadec_sfc_cruise = results['FADEC'].get(
        'phase_metrics', {}).get('cruise', {}).get('mean_sfc', 0.65)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    drl_controllers = [c for c in results if c != 'FADEC']
    ctrl_colors = [COLORS.get(c, 'gray') for c in drl_controllers]

    # Panel A: mejora en reward.
    reward_gains = [
        ((results[c]['mean_reward'] - fadec_reward)
         / abs(fadec_reward)) * 100 if fadec_reward != 0 else 0
        for c in drl_controllers
    ]
    bars_r = axes[0].bar(drl_controllers, reward_gains,
                          color=ctrl_colors, alpha=0.85)
    axes[0].axhline(y=0, color=COLORS['FADEC'], linestyle='--',
                    linewidth=2, label='FADEC baseline')
    axes[0].set_ylabel('Mejora [%]')
    axes[0].set_title(f'Mejora en reward (baseline: {fadec_reward:.0f})')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars_r, reward_gains):
        offset = 1 if val >= 0 else -3
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + offset,
                     f'{val:+.1f}%', ha='center',
                     fontweight='bold', fontsize=12)

    # Panel B: mejora relativa en tasa de seguridad.
    safety_gains = []
    for c in drl_controllers:
        s = results[c]['safety_rate']
        gain = ((s - fadec_safety) / fadec_safety) * 100 \
            if fadec_safety > 0 else s * 100
        safety_gains.append(gain)
    bars_s = axes[1].bar(drl_controllers, safety_gains,
                          color=ctrl_colors, alpha=0.85)
    axes[1].axhline(y=0, color=COLORS['FADEC'], linestyle='--',
                    linewidth=2, label='FADEC baseline')
    axes[1].set_ylabel('Mejora relativa [%]')
    axes[1].set_title(f'Mejora en seguridad '
                      f'(FADEC: {fadec_safety * 100:.0f}%)')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars_s, safety_gains):
        offset = 2 if val >= 0 else -5
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + offset,
                     f'{val:+.0f}%', ha='center',
                     fontweight='bold', fontsize=12)

    # Panel C: reducción de SFC en crucero.
    sfc_reductions = []
    for c in drl_controllers:
        sfc = results[c].get(
            'phase_metrics', {}).get('cruise', {}).get('mean_sfc', 0)
        reduction = ((fadec_sfc_cruise - sfc) / fadec_sfc_cruise) * 100 \
            if sfc > 0 else 0
        sfc_reductions.append(reduction)
    bars_sfc = axes[2].bar(drl_controllers, sfc_reductions,
                            color=ctrl_colors, alpha=0.85)
    axes[2].axhline(y=0, color=COLORS['FADEC'], linestyle='--',
                    linewidth=2, label='FADEC baseline')
    axes[2].set_ylabel('Reduccion de SFC [%]')
    axes[2].set_title(f'Eficiencia en crucero '
                      f'(FADEC SFC: {fadec_sfc_cruise:.3f})')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars_sfc, sfc_reductions):
        offset = 0.2 if val >= 0 else -0.5
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + offset,
                     f'{val:+.1f}%', ha='center',
                     fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {output_path}")


def _shade_mission_phases(env, axes) -> None:
    """Colorea el fondo de los ejes con bandas que delimitan las fases de misión.

    Sin estas bandas las series temporales serían difíciles de interpretar: los saltos
    de empuje o de temperatura solo cobran sentido cuando se ve que coinciden con una
    transición de fase. Se reinicia el entorno antes de leer la secuencia para
    asegurar que exista una, aunque no se haya ejecutado ningún episodio todavía.

    param env: Entorno del que tomar la secuencia de fases de misión.
    param axes: Eje o colección de ejes de matplotlib a sombrear.
    return: None; los ejes se modifican in situ.
    """
    env.reset()
    total = 0
    axes_iter = axes.flat if hasattr(axes, 'flat') else axes
    for phase, duration in env.mission_sequence:
        color = PHASE_COLORS.get(phase, 'gray')
        for ax in axes_iter:
            ax.axvspan(total, total + duration, alpha=0.15, color=color)
        total += duration


def fig21_mission_profile(controllers: dict, pinn,
                          output_path: Path) -> None:
    """Dibuja la evolución temporal comparada de los cuatro controladores.

    Cuatro paneles con eje temporal compartido: empuje, consumo, temperatura de
    turbina con sus límites marcados y posición del VFGV. Usa el perfil de misión
    completo, no el mixto de la evaluación, porque aquí interesa una secuencia de
    fases realista y legible en lugar de una aleatoria. Es la figura que permite ver
    el comportamiento, y no solo el resultado agregado: cómo cada controlador
    reacciona a las transiciones y cuánto margen deja frente al límite térmico.

    param controllers: Diccionario {nombre: controlador} a representar.
    param pinn: Gemelo digital con el que construir el entorno.
    param output_path: Ruta del archivo PNG a generar.
    return: None; la figura queda escrita en output_path.
    """
    fig, axes = plt.subplots(4, 1, figsize=(18, 16), sharex=True)

    env = ACEEnv(pinn_model=pinn,
                  mission_profile=MISSION_PROFILE_FULL,
                  max_steps=DEFAULT_MAX_STEPS)

    linestyles = {
        'SAC':    '-',
        'TD3':    '-',
        'TD3+BC': '-',
        'FADEC':  '--',
    }

    for name, controller in controllers.items():
        history = run_episode(env, controller, name)
        steps = history['steps']
        color = COLORS.get(name, 'gray')
        linestyle = linestyles.get(name, '-')
        axes[0].plot(steps, history['thrust'], color=color,
                     linestyle=linestyle, linewidth=2, label=name)
        axes[1].plot(steps, history['sfc'], color=color,
                     linestyle=linestyle, linewidth=2, label=name)
        axes[2].plot(steps, history['T4'], color=color,
                     linestyle=linestyle, linewidth=2, label=name)
        actions = np.array(history['actions'])
        if len(actions) > 0:
            axes[3].plot(steps, actions[:, 0] * 20, color=color,
                         linestyle=linestyle, linewidth=2,
                         label=f'{name} VFGV')

    axes[0].set_ylabel('Thrust [lbf]')
    axes[0].legend(loc='upper right')
    axes[1].set_ylabel('SFC [lbm/lbf.h]')
    axes[1].legend(loc='upper right')
    axes[2].set_ylabel('T4 [R]')
    axes[2].axhline(y=3200, color='red', linestyle=':',
                    linewidth=2, alpha=0.7, label='T4 max')
    axes[2].axhline(y=3000, color='orange', linestyle=':',
                    linewidth=1, alpha=0.5, label='T4 warn')
    axes[2].legend(loc='upper right')
    axes[3].set_ylabel('VFGV [deg]')
    axes[3].set_xlabel('Step')
    axes[3].legend(loc='upper right')

    _shade_mission_phases(
        ACEEnv(pinn_model=pinn,
               mission_profile=MISSION_PROFILE_FULL,
               max_steps=DEFAULT_MAX_STEPS),
        axes,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {output_path}")


def fig22_reward_by_phase(results: dict, output_path: Path) -> None:
    """Dibuja el empuje y el consumo medios por fase, agrupados por controlador.

    Es necesario porque el promedio sobre la misión completa oculta el compromiso que
    define el problema: un agente puede ganar en crucero gastando menos y perder en
    combate dando menos empuje, y solo el desglose por fase revela si la política ha
    aprendido a priorizar de forma distinta en cada régimen, que es exactamente lo que
    la recompensa dependiente de fase pretende inducir.

    param results: Diccionario de métricas por controlador.
    param output_path: Ruta del archivo PNG a generar.
    return: None; la figura queda escrita en output_path.
    """
    controllers = list(results.keys())
    x = np.arange(len(PHASES))
    width = 0.8 / len(controllers)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, controller in enumerate(controllers):
        phase_metrics = results[controller].get('phase_metrics', {})
        thrust_vals = [phase_metrics.get(p, {}).get('mean_thrust', 0)
                       for p in PHASES]
        sfc_vals = [phase_metrics.get(p, {}).get('mean_sfc', 0)
                    for p in PHASES]
        color = COLORS.get(controller, 'gray')
        axes[0].bar(x + i * width, thrust_vals, width,
                    label=controller, color=color, alpha=0.85)
        axes[1].bar(x + i * width, sfc_vals, width,
                    label=controller, color=color, alpha=0.85)

    center_offset = (len(controllers) - 1) * width / 2
    axes[0].set_xlabel('Fase de misión')
    axes[0].set_ylabel('Thrust medio [lbf]')
    axes[0].set_xticks(x + center_offset)
    axes[0].set_xticklabels(PHASES, rotation=15)
    axes[0].legend()
    axes[1].set_xlabel('Fase de misión')
    axes[1].set_ylabel('SFC medio')
    axes[1].set_xticks(x + center_offset)
    axes[1].set_xticklabels(PHASES, rotation=15)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {output_path}")


def fig23_actuator_actions(controllers: dict, pinn,
                            output_path: Path) -> None:
    """Dibuja la trayectoria de los cuatro actuadores para cada controlador.

    Un panel por actuador, con las acciones desnormalizadas a sus unidades físicas.
    Es la figura que explica el porqué de las demás: muestra qué hace realmente cada
    política con las geometrías variables, el afterburner y el sangrado, y permite    contrastar los escalones rígidos del scheduling del FADEC con la modulación
    continua que aprenden los agentes.

    param controllers: Diccionario {nombre: controlador} a representar.
    param pinn: Gemelo digital con el que construir el entorno.
    param output_path: Ruta del archivo PNG a generar.
    return: None; la figura queda escrita en output_path.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    env = ACEEnv(pinn_model=pinn,
                  mission_profile=MISSION_PROFILE_FULL,
                  max_steps=DEFAULT_MAX_STEPS)

    actuator_names = ['VFGV [deg]', 'VGV-T [deg]',
                      'Afterburner (AB)', 'Sangrado (Bleed)']
    actuator_scales = [20.0, 15.0, 1.0, 1.0]

    linestyles = {
        'SAC':    '-',
        'TD3':    '-',
        'TD3+BC': '-',
        'FADEC':  '--',
    }

    for name, controller in controllers.items():
        history = run_episode(env, controller, name)
        actions = np.array(history['actions'])
        steps = history['steps']
        color = COLORS.get(name, 'gray')
        linestyle = linestyles.get(name, '-')

        for i, (ax, act_name, scale) in enumerate(
                zip(axes.flat, actuator_names, actuator_scales)):
            values = (actions[:, i] * scale if i < 2
                      else (actions[:, i] + 1) / 2)
            ax.plot(steps, values, color=color,
                    linestyle=linestyle, linewidth=2, label=name)
            ax.set_ylabel(act_name)
            ax.set_xlabel('Step')
            ax.legend(fontsize=9)

    _shade_mission_phases(
        ACEEnv(pinn_model=pinn,
               mission_profile=MISSION_PROFILE_FULL,
               max_steps=DEFAULT_MAX_STEPS),
        axes,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Guardado: {output_path}")


# Reporte por consola
def log_comparison_table(results: dict) -> None:
    """Emite por el log la tabla comparativa de las métricas principales.

    Presenta un controlador por columna, con recompensa, dispersión, tasa de seguridad
    y consumo en las fases más representativas. Deja las cifras exactas disponibles
    para la memoria, que las figuras solo muestran de forma aproximada.

    param results: Diccionario de métricas por controlador.
    return: None; la tabla se emite por el logger del módulo.
    """
    logger.info("Comparativa final:")
    header = f"  {'Metrica':<20}" + \
        ''.join(f"{c:>12}" for c in results.keys())
    logger.info(header)
    logger.info(f"  {'-' * (20 + 12 * len(results))}")
    row_reward = f"  {'Reward medio':<20}" + \
        ''.join(f"{r['mean_reward']:>12.2f}" for r in results.values())
    logger.info(row_reward)
    row_std = f"  {'+/- Std':<20}" + \
        ''.join(f"{r['std_reward']:>12.2f}" for r in results.values())
    logger.info(row_std)
    row_safety = f"  {'Tasa seguridad':<20}" + \
        ''.join(f"{r['safety_rate']:>12.1%}" for r in results.values())
    logger.info(row_safety)

    logger.info("SFC por fase:")
    for phase in ['takeoff', 'cruise', 'combat']:
        line = f"  {phase:<20}"
        for r in results.values():
            sfc = r.get('phase_metrics', {}).get(
                phase, {}).get('mean_sfc', 0)
            line += f"{sfc:>12.3f}"
        logger.info(line)


# Punto de entrada
def main() -> None:
    """Punto de entrada que evalúa los controladores y genera las cinco figuras.

    Carga los modelos, ejecuta la evaluación, persiste el JSON de resultados y encadena
    las figuras. Guardar el JSON antes de dibujar es intencionado: la evaluación es la
    parte cara y así su resultado queda a salvo aunque falle la generación de alguna
    figura. Se aborta si no hay al menos dos controladores.

    return: None; sale sin generar nada si faltan controladores, y en caso contrario
        deja el JSON en results/ y los PNG en results/figures.
    """
    logger.info("Evaluacion Seccion 5: DRL vs FADEC")

    controllers, pinn = load_all_controllers()

    if len(controllers) < 2:
        logger.error("No hay controladores suficientes para la comparativa")
        return

    results = evaluate_controllers(controllers, pinn)

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Resultados guardados en {RESULTS_JSON}")

    log_comparison_table(results)

    logger.info("Generando figuras...")
    fig19_training_curves(FIG_DIR / 'fig19_training_curves.png')
    fig20_drl_vs_fadec(results, FIG_DIR / 'fig20_drl_vs_fadec.png')
    fig21_mission_profile(controllers, pinn,
                           FIG_DIR / 'fig21_mission_profile.png')
    fig22_reward_by_phase(results, FIG_DIR / 'fig22_reward_by_phase.png')
    fig23_actuator_actions(controllers, pinn,
                            FIG_DIR / 'fig23_actuator_actions.png')
    logger.info("Seccion 5 completada: 5 figuras generadas")


if __name__ == '__main__':
    main()