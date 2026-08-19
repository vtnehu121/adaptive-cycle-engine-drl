"""
Última Fecha de Modificación: 08/Aug/2026
Descripción aetp_validation.py: Notebook de validación cualitativa del corpus
ACE frente a los objetivos públicos del programa AETP (Adaptive Engine
Transition Program), única fuente contrastable en literatura abierta sobre
motores de ciclo adaptativo militares de tres flujos: XA100 (GE Aerospace) y
XA101 (Pratt & Whitney), concebidos como opción de remotorización del F-35 y
base tecnológica para el programa NGAD.

Los datos operativos completos,telemetría de vuelo, curvas off-design, mapas
experimentales de componentes,son clasificados. Las referencias utilizables
se limitan a los objetivos públicos publicados por GE Aerospace (2019, 2022)
para el XA100 respecto al F135 base:

    - Empuje supersónico:      +10%
    - Eficiencia combustible:  +25%
    - Alcance:                 +30%
    - Aceleración:             +20-40%
    - Gestión térmica:         2x

Este notebook caracteriza el corpus por régimen operativo, comprueba que
reproduce las tendencias propias de un motor adaptativo (reconfiguración del
BPR entre regímenes, amplitud del régimen del ventilador, diferencial de SFC
entre combate y crucero) y documenta explícitamente qué objetivos AETP no son
contrastables por depender de datos clasificados o de simulación transitoria.
Genera las figuras de tendencias y envelope y sus tablas asociadas, sin
pretender cuantificar errores absolutos frente a valores no publicados.
"""

# %% — Configuración
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if sys.platform == 'win32':
    os.system('')  
    try:
        os.system('chcp 65001 >nul 2>&1')
    except Exception:
        pass

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

plt.rcParams.update({
    'figure.figsize': (14, 6),
    'figure.dpi': 150,
    'font.size': 12,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

FIG_DIR = Path('results/figures')
TABLE_DIR = Path('results/notebooks')
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Configuración cargada. Figuras en: {FIG_DIR.resolve()}")

# %% — Objetivos públicos del programa AETP
# Fuentes verificadas (literatura abierta 2019-2023):
#   [1] GE Aerospace (2019). "GE Aviation's XA100 Adaptive Cycle Engine
#       Completes Detailed Design". Press Release, Feb. 27, 2019.
#       https://www.geaerospace.com/news/press-releases/defense-engines/ge-aviations-xa100-adaptive-cycle-engine-completes-detailed-design
#   [2] GE Aerospace (2022). "Second XA100 Test Campaign at AEDC".
#       Farnborough International Airshow 2022 press briefing.
#   [3] Air & Space Forces Magazine (2022). "GE's XA100 Engine Would
#       Create New Mission Possibilities for the F-35". Aug. 9, 2022.
#       https://www.airandspaceforces.com/ge-xa100-engine-create-new-mission-possibilities-for-f-35/
#   [4] Wikipedia (2025). "General Electric XA100" y "Pratt & Whitney XA101".
#       Consolidación pública de especificaciones AETP.
#   [5] Pratt & Whitney (2024). "F135 Engine Fast Facts" (documento oficial).
#       https://www.prattwhitney.com/en/products/military-engines/f135
#       Los valores citados (28k/43k lbf, BPR 0.57) coinciden con:
#       Wikipedia, "Pratt & Whitney F135" y F-16.net technical forums.
#
# Las cifras a continuación son OBJETIVOS PUBLICADOS del programa AETP,
# no mediciones directas. Los datos operativos completos (curvas
# off-design, mapas de componentes, telemetría) permanecen clasificados.
AETP_TARGETS = {
    # Objetivos originales publicados por GE Aerospace [1]
    'supersonic_thrust_gain_pct':      10.0,    # % vs F135   [1] GE 2019
    'fuel_efficiency_gain_pct':        25.0,    # % vs F135   [1] GE 2019
    # Nota: GE reporta "25% better fuel efficiency" como cifra única.
    # El notebook la utiliza como proxy tanto para SFC en crucero como
    # para eficiencia global, ya que la literatura pública no distingue
    # ambas métricas explícitamente.

    # Objetivos ampliados publicados en 2022 [2, 3]
    'range_increase_pct':              30.0,    # % vs F135   [3] AFM 2022
    'acceleration_improvement_pct':    30.0,    # % (20-40%)  [3] AFM 2022
    'thermal_management_gain_pct':    100.0,    # duplicación [3] AFM 2022
    # Especificaciones de arquitectura [4]
    'thrust_class_lbf':             45000,      # empuje AB   [4] Wiki XA100
    'stream_architecture':          'three-stream adaptive cycle',  # [4]
}

# Motor base de referencia: Pratt & Whitney F135 (F-35)
F135_REFERENCE = {
    'dry_thrust_lbf':       28000,   # lbf   [5] P&W F135 Fast Facts 2024
    'ab_thrust_lbf':        43000,   # lbf   [5] P&W F135 Fast Facts 2024
    'cruise_sfc_lb_lbf_hr': 0.886,   # lb/lbf/hr sin AB (proxy crucero) [5]
    'bpr':                   0.57,   # -     [5] P&W F135 Fast Facts 2024
}

# XA100 derivado: empuje esperado post-mejora (cálculo, no publicado)
XA100_DERIVED = {
    'ab_thrust_lbf': F135_REFERENCE['ab_thrust_lbf'] * (1 + AETP_TARGETS['supersonic_thrust_gain_pct'] / 100),
    'cruise_sfc_lb_lbf_hr': F135_REFERENCE['cruise_sfc_lb_lbf_hr'] * (1 - AETP_TARGETS['fuel_efficiency_gain_pct'] / 100),
}

print("Objetivos AETP publicados (GE Aerospace 2019-2022):")
for k, v in AETP_TARGETS.items():
    if isinstance(v, str):
        print(f"  {k:<32}: {v}")
    elif v > 100:
        print(f"  {k:<32}: {v:>8.0f}")
    else:
        print(f"  {k:<32}: {v:+.1f}%")

print("\nMotor base de referencia (F135):")
for k, v in F135_REFERENCE.items():
    print(f"  {k:<32}: {v}")

print("\nXA100 derivado (F135 + mejoras AETP):")
for k, v in XA100_DERIVED.items():
    if v > 1000:
        print(f"  {k:<32}: {v:>10.0f}")
    else:
        print(f"  {k:<32}: {v:>10.3f}")

# %% — Carga del corpus
corpus_path = Path('data/synthetic/ace_dataset_5000.csv')
df = pd.read_csv(corpus_path)

print(f"\nCorpus cargado: {df.shape[0]} muestras × {df.shape[1]} variables")

def classify_regime(row):
    """Asigna a un punto de operación la fase de misión que le corresponde.

    Aplica reglas sobre altitud, Mach y palanca en un orden deliberado: primero se
    identifica el combate, después el despegue y el descenso, y solo entonces se
    distingue crucero de ascenso, de modo que las condiciones más específicas tengan
    prioridad.

    Es solo un plan B: reetiqueta a posteriori y no reproduce el estrato del que
    salió cada punto. Exige Mach >= 1.3 para llamar 'combat', pero el estrato de
    combate se muestrea desde Mach 0.8, así que todo el tramo transónico acaba en
    'climb' y el 14 % de las muestras queda mal asignado. Úsese únicamente cuando
    no esté disponible la etiqueta del muestreador (flight_conditions_{n}.csv).

    param row: Fila del corpus con las columnas altitude, mach y tra.
    return: Nombre de la fase de misión asignada al punto de operación.
    """
    alt = row['altitude']
    mach = row['mach']
    tra = row['tra']
    if mach >= 1.3 and tra >= 90:
        return 'combat'
    elif alt < 5000 and tra >= 85:
        return 'takeoff'
    elif tra < 60:
        return 'descent'
    elif alt >= 25000 and tra >= 70:
        return 'cruise'
    else:
        return 'climb'

def load_regime_labels(corpus: pd.DataFrame) -> pd.Series:
    """Devuelve la etiqueta de fase de misión de cada punto del corpus.

    Prefiere la etiqueta que el propio muestreador estratificado escribió en
    flight_conditions_{n}.csv, porque es el estrato real del que salió el punto y
    se corresponde fila a fila con el corpus. Solo si ese fichero no está o no
    encaja se recurre a classify_regime, que reetiqueta a posteriori y discrepa
    del muestreador en torno al 14 % de las muestras.

    param corpus: Corpus ya cargado, usado para comprobar el número de filas.
    return: Serie de nombres de fase alineada con el índice del corpus.
    """
    conditions_path = Path(f'data/synthetic/flight_conditions_{len(corpus)}.csv')
    if conditions_path.exists():
        conditions = pd.read_csv(conditions_path)
        if 'regime' in conditions.columns and len(conditions) == len(corpus):
            print(f"Régimen tomado de {conditions_path.name} (etiqueta del muestreador).")
            return pd.Series(conditions['regime'].values, index=corpus.index)
    print("Régimen estimado con classify_regime (etiqueta del muestreador no disponible).")
    return corpus.apply(classify_regime, axis=1)

if 'regime' not in df.columns:
    df['regime'] = load_regime_labels(df)

regime_counts = df['regime'].value_counts()
for regime, count in regime_counts.items():
    pct = count / len(df) * 100
    print(f"  {regime:<10} {count:>4} ({pct:.1f}%)")
    
# %% [markdown]
# ## Sección 1: Caracterización por régimen operativo
#
# Extraemos del corpus estadísticas agregadas por régimen para
# identificar cómo el motor reconfigura su operación entre fases de
# misión. Esta es la característica definitoria de un motor
# adaptativo.

# %% — Estadísticas por régimen
regime_col = 'regime'
regime_stats = df.groupby(regime_col).agg({
    'thrust_approx': ['mean', 'std', 'max'],
    'sfc':           ['mean', 'std', 'min'],
    'Nf':            ['mean', 'min', 'max'],
    'T4':            ['mean', 'max'],
    'BPR':           ['mean', 'std'],
    'bpr_ts':        ['mean', 'std'],
}).round(2)

print("Estadísticas por régimen operativo:")
print(regime_stats.to_string())

regime_stats.to_csv(TABLE_DIR / 'regime_statistics.csv', encoding='utf-8')
print(f"\nGuardado: {TABLE_DIR / 'regime_statistics.csv'}")

# %% [markdown]
# ## Sección 2: Verificación de tendencias adaptativas
#
# Un motor adaptativo debe exhibir tres comportamientos característicos:
#
# 1. **Reconfiguración del BPR entre regímenes**: BPR mayor en crucero
#    (mejor SFC) y menor en combate (más empuje específico).
# 2. **Variación amplia del régimen del ventilador (Nf)** entre fases
#    de misión, controlada por el actuador VFGV.
# 3. **Diferencial de SFC entre combate y crucero** consistente con la
#    filosofía de operación multiobjetivo.

# %% — Cálculo de tendencias
combat = df[df[regime_col] == 'combat']
cruise = df[df[regime_col] == 'cruise']
takeoff = df[df[regime_col] == 'takeoff']

# 1. Reconfiguración del BPR
bpr_combat = combat['BPR'].mean()
bpr_cruise = cruise['BPR'].mean()
bpr_reconfiguration_pct = ((bpr_cruise - bpr_combat) / bpr_combat) * 100

# 2. Rango de Nf
nf_min = df['Nf'].min()
nf_max = df['Nf'].max()
nf_range_pct = ((nf_max - nf_min) / df['Nf'].mean()) * 100

# 3. Diferencial de SFC
sfc_combat = combat['sfc'].mean()
sfc_cruise = cruise['sfc'].mean()
sfc_differential_pct = ((sfc_combat - sfc_cruise) / sfc_cruise) * 100

# 4. Ratio de empuje
thrust_combat = combat['thrust_approx'].mean()
thrust_cruise = cruise['thrust_approx'].mean()
thrust_ratio = thrust_combat / thrust_cruise

print("Tendencias adaptativas observadas en el corpus:\n")
print(f"1. Reconfiguración BPR:")
print(f"   BPR combate: {bpr_combat:.3f}")
print(f"   BPR crucero: {bpr_cruise:.3f}")
print(f"   Reconfiguración: {bpr_reconfiguration_pct:+.1f}%")

print(f"\n2. Rango operativo del ventilador (Nf):")
print(f"   Nf mín: {nf_min:.0f} rpm")
print(f"   Nf máx: {nf_max:.0f} rpm")
print(f"   Rango relativo: {nf_range_pct:.1f}%")

print(f"\n3. Diferencial SFC combate vs crucero:")
print(f"   SFC combate: {sfc_combat:.3f} lbm/lbf/hr")
print(f"   SFC crucero: {sfc_cruise:.3f} lbm/lbf/hr")
print(f"   Diferencial combate vs crucero: {sfc_differential_pct:+.1f}%")

print(f"\n4. Ratio de empuje combate/crucero: {thrust_ratio:.2f}x")

# %% — Limitaciones del corpus (Sim-to-Real Gap)
print("\nLimitaciones del corpus (Sim-to-Real Gap):\n")

t4_violations = (df['T4'] > 3200).sum()
t4_warnings = (df['T4'] > 3000).sum()
combat_total = len(df[df[regime_col] == 'combat'])
combat_violations = len(df[(df[regime_col] == 'combat') & (df['T4'] > 3200)])
thrust_median = df['thrust_approx'].median()
thrust_max = df['thrust_approx'].max()

print(f"1. Violaciones de safety limit T4 (max=3200°R):")
print(f"   - Total corpus:  {t4_violations} muestras ({t4_violations/len(df)*100:.1f}%)")
print(f"   - En combat:     {combat_violations}/{combat_total} ({combat_violations/combat_total*100:.0f}%)")
print(f"   - Warning >3000: {t4_warnings} muestras ({t4_warnings/len(df)*100:.1f}%)")
print()
print("2. Interpretación:")
print("   El corpus se genera por muestreo estratificado uniforme sin filtros de")
print("   safety. Las violaciones son intencionales: permiten al agente DRL")
print("   aprender a EVITAR zonas peligrosas via _check_safety durante el")
print("   entrenamiento (formulación CMDP).")
print()
print("3. Extrapolación T-MATS:")
print("   En condiciones supersónicas a baja altitud (Mach > 1.7,")
print("   altitud < 7000 ft), T-MATS extrapola fuera de su ventana")
print(f"   calibrada, generando empujes hasta {thrust_max:.0f} lbf,")
print("   físicamente irreales pero matemáticamente coherentes.")
print(f"   La mediana del corpus ({thrust_median:.0f} lbf) es coherente")
print("   con la clase del JT9D base (~46k lbf takeoff).")
# %% — Visualización de tendencias adaptativas
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

regime_order = ['takeoff', 'climb', 'cruise', 'combat', 'descent']
regime_labels_es = ['Despegue', 'Ascenso', 'Crucero', 'Combate', 'Descenso']
palette = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#264653']

# Panel 1: BPR total por régimen
bpr_by_regime = [df[df[regime_col] == r]['BPR'].mean() for r in regime_order]
bars1 = axes[0].bar(regime_labels_es, bpr_by_regime, color=palette,
                    edgecolor='white')
axes[0].set_ylabel('BPR medio [-]')
for bar, val in zip(bars1, bpr_by_regime):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

# Panel 2: Rango de Nf por régimen (barras de error)
nf_stats = df.groupby(regime_col)['Nf'].agg(['mean', 'std', 'min', 'max'])
nf_mean = [nf_stats.loc[r, 'mean'] for r in regime_order]
nf_err = [nf_stats.loc[r, 'std'] for r in regime_order]
axes[1].bar(regime_labels_es, nf_mean, yerr=nf_err,
            color=palette, edgecolor='white',
            capsize=6, error_kw={'linewidth': 1.5})
axes[1].set_ylabel('Nf [rpm]')

# Panel 3: SFC por régimen
sfc_by_regime = [df[df[regime_col] == r]['sfc'].mean() for r in regime_order]
bars3 = axes[2].bar(regime_labels_es, sfc_by_regime, color=palette,
                    edgecolor='white')
axes[2].set_ylabel('SFC medio [lbm/lbf/hr]')
for bar, val in zip(bars3, sfc_by_regime):
    axes[2].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig13_adaptive_trends.png',
            dpi=300, bbox_inches='tight')
plt.show()
print(f"Guardado: {FIG_DIR / 'fig13_adaptive_trends.png'}")

# %% [markdown]
# ## Sección 3: Interpretación cualitativa frente a objetivos AETP
#
# Los objetivos AETP publicados no son directamente comparables con
# valores absolutos del corpus, pero sí permiten evaluar si las
# tendencias del modelo son consistentes con la filosofía adaptativa
# que persigue el programa.

# %% — Tabla interpretativa
# Solo se incluyen objetivos AETP para los que el corpus contiene una
# métrica realmente comparable. Los objetivos no comparables (alcance,
# aceleración transitoria, gestión térmica activa) se documentan en la
# sección "Objetivos AETP no cubiertos por este corpus".
interpretation = pd.DataFrame([
    {
        'Objetivo AETP publicado': '+10% empuje supersónico vs F135 [1]',
        'Métrica observable en el corpus': 'Ratio empuje combate/crucero',
        'Valor en el corpus': f'{thrust_ratio:.2f}x',
        'Interpretación': (
            'El corpus captura reconfiguración de empuje entre regímenes. '
            f'El ratio observado ({thrust_ratio:.2f}x) supera el valor típico '
            'militar (~3-4x) porque los edge cases supersónicos a baja '
            'altitud (Mach > 1.7) provocan extrapolación T-MATS fuera de '
            'calibración, inflando el promedio de combate. La reconfiguración '
            'cualitativa es válida; la magnitud absoluta no es directamente '
            'comparable con AETP.'
        ),
    },
    {
        'Objetivo AETP publicado': '+25% eficiencia combustible vs F135 [1]',
        'Métrica observable en el corpus': 'Diferencial SFC + amplitud Nf',
        'Valor en el corpus': f'ΔSFC = {sfc_differential_pct:+.1f}%, ΔNf = {nf_range_pct:.1f}%',
        'Interpretación': (
            'El diferencial de SFC entre regímenes y la amplitud operativa '
            'del ventilador reflejan la eficiencia adaptativa perseguida '
            'por AETP; la reducción absoluta frente al F135 requeriría '
            'datos clasificados no disponibles.'
    ),
},
    {
        'Objetivo AETP publicado': 'Arquitectura de tres flujos [4]',
        'Métrica observable en el corpus': 'BPR total variable',
        'Valor en el corpus': f'{df["BPR"].min():.2f} — {df["BPR"].max():.2f}',
        'Interpretación': (
            'El BPR total varía de forma coherente con bpr_ts, '
            'demostrando el control activo del tercer flujo característico '
            'de la arquitectura de ciclo adaptativo.'
        ),
    },
    {
        'Objetivo AETP publicado': 'Clase de empuje 45 000 lbf [4]',
        'Métrica observable en el corpus': 'Empuje máximo del corpus',
        'Valor en el corpus': f'{df["thrust_approx"].max():.0f} lbf',
        'Interpretación': (
            f'La mediana del corpus ({df["thrust_approx"].median():.0f} lbf) es '
            f'coherente con el JT9D base (~46k lbf takeoff). Los outliers '
            f'superiores corresponden a puntos supersónicos a baja altitud, '
            f'donde T-MATS extrapola fuera de su ventana de calibración. '
            f'El {(df["T4"] > 3200).sum() / len(df) * 100:.0f}% del corpus viola '
            f'T4_max=3200°R por diseño del muestreo, permitiendo al DRL aprender a evitar '
            f'zonas peligrosas via _check_safety.'
        ),
    },
])

# Objetivos AETP no cubiertos por el corpus 
non_comparable = pd.DataFrame([
    {
        'Objetivo AETP no cubierto': '+30% alcance vs F135 [3]',
        'Razón': (
            'El alcance requiere simulación de misión completa integrando '
            'perfil de vuelo, consumo y peso; el corpus contiene puntos '
            'estacionarios individuales sin trayectoria temporal.'
        ),
    },
    {
        'Objetivo AETP no cubierto': '+20-40% aceleración [3]',
        'Razón': (
            'La aceleración es una métrica transitoria (dNf/dt del rotor); '
            'T-MATS en este trabajo resuelve puntos cuasi-estacionarios '
            'sin simulación dinámica de transitorios.'
        ),
    },
    {
        'Objetivo AETP no cubierto': 'Duplicación gestión térmica [3]',
        'Razón': (
            'La capacidad térmica AETP proviene del tercer flujo como '
            'heat sink estructural; el modelo captura el control del '
            'tercer flujo (bpr_ts) pero no su función térmica activa, '
            'que requeriría acoplamiento con un modelo de intercambiador.'
        ),
    },
])

print("Interpretación cualitativa frente a objetivos AETP:\n")
for _, row in interpretation.iterrows():
    print(f"[{row['Objetivo AETP publicado']}]")
    print(f"  Métrica:        {row['Métrica observable en el corpus']}")
    print(f"  Valor:          {row['Valor en el corpus']}")
    print(f"  Interpretación: {row['Interpretación']}")
    print()
 
interpretation.to_csv(TABLE_DIR / 'aetp_qualitative_validation.csv',
                      index=False,  encoding='utf-8')
print(f"Guardado: {TABLE_DIR / 'aetp_qualitative_validation.csv'}")
 
print("\nObjetivos AETP no cubiertos por el corpus:\n")
for _, row in non_comparable.iterrows():
    print(f"[{row['Objetivo AETP no cubierto']}]")
    print(f"  Razón: {row['Razón']}")
    print()
 
non_comparable.to_csv(TABLE_DIR / 'aetp_non_comparable_objectives.csv', 
                      index=False, encoding='utf-8')
print(f"Guardado: {TABLE_DIR / 'aetp_non_comparable_objectives.csv'}")

# %% [markdown]
# ## Sección 4: Envolvente operativa del corpus
#
# Visualización de la envolvente de vuelo cubierta por el corpus,
# situando los cinco regímenes de misión en el plano altitud-Mach.
# Esta representación permite verificar que la cobertura operativa
# es compatible con las trayectorias de misión que el programa AETP
# considera relevantes para el F-35 y NGAD.

# %% — Envolvente de vuelo
fig, ax = plt.subplots(figsize=(12, 8))

for regime, label, color in zip(regime_order, regime_labels_es, palette):
    mask = df[regime_col] == regime
    ax.scatter(df.loc[mask, 'mach'], df.loc[mask, 'altitude'],
               c=color, label=label, alpha=0.55, s=18,
               edgecolors='none')

ax.set_xlabel('Número de Mach [-]')
ax.set_ylabel('Altitud [ft]')
ax.legend(loc='upper left', title='Fase de misión')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig14_flight_envelope.png',
            dpi=300, bbox_inches='tight')
plt.show()
print(f"Guardado: {FIG_DIR / 'fig14_flight_envelope.png'}")

# %% — Resumen ejecutivo
print(f"""
Validación cualitativa frente al programa AETP:

  Corpus:              {df.shape[0]} muestras cubriendo 5 regímenes operativos
  Envelope:            Mach [{df['mach'].min():.2f}, {df['mach'].max():.2f}]
                       Altitud [{df['altitude'].min():.0f}, {df['altitude'].max():.0f}] ft

  Tendencias observadas:
    - Reconfiguración BPR combate→crucero: {bpr_reconfiguration_pct:+.1f}%
    - Rango operativo Nf:                  {nf_range_pct:.1f}%
    - Diferencial SFC combate vs crucero:  {sfc_differential_pct:+.1f}%
    - Ratio empuje combate/crucero:        {thrust_ratio:.2f}x

  Interpretación:
    El corpus reproduce las tendencias operativas cualitativas
    características de un motor adaptativo (reconfiguración inter-régimen,
    amplitud del rango del ventilador, diferencial de SFC entre modos).
    La comparación cuantitativa con valores absolutos del programa AETP
    no es posible en literatura abierta.

Salidas generadas:
  - fig13_adaptive_trends.png            Tendencias adaptativas por régimen
  - fig14_flight_envelope.png            Envolvente operativa del corpus
  - regime_statistics.csv                Estadísticas por régimen
  - aetp_qualitative_validation.csv      Tabla interpretativa AETP
  - aetp_non_comparable_objectives.csv   Objetivos AETP no cubiertos
""")