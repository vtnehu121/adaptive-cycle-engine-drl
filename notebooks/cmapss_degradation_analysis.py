"""
Última Fecha de Modificación: 08/Aug/2026
Descripción cmapss_degradation_analysis.py: Notebook que analiza el dataset NASA
C-MAPSS (Saxena et al., PHM 2008) y visualiza cómo sus patrones de deterioro se
trasladan al corpus sintético del ACE. Recorre la distribución de vida remanente
de los cuatro sub-datasets, las trayectorias de los sensores más informativos,
la tabla de desgaste natural del paper de referencia y el mapeo entre componentes
de C-MAPSS y actuadores del ACE, para terminar contrastando las distribuciones
de ambos corpus.

No inyecta degradación: la lógica de inyección estocástica reside íntegramente
en `src/data_gen/degradation.py`; este notebook consume el corpus degradado ya
generado por el pipeline y se limita a caracterizarlo.

Referencia:
    Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). "Damage
    propagation modeling for aircraft engine run-to-failure simulation."
    International Conference on Prognostics and Health Management (PHM),
    IEEE. DOI: 10.1109/PHM.2008.4711414
"""

# %% — Configuración
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Forzar UTF-8 en toda la cadena de I/O 
if sys.platform == 'win32':
    os.system('')  
    try:
        os.system('chcp 65001 >nul 2>&1')
    except Exception:
        pass

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path('.').resolve()))

from src.preprocessing.cmapss_loader import CMAPSSLoader

plt.rcParams.update({
    'figure.figsize': (12, 7),
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

# %% [markdown]
# # Análisis C-MAPSS y visualización del corpus ACE degradado
#
# El dataset NASA C-MAPSS (Saxena, Goebel, Simon; PHM 2008) es el
# benchmark de referencia en Prognostics and Health Management (PHM)
# de motores de turbina. Sus cuatro sub-datasets simulan turbofanes
# convencionales de doble eje con degradación estocástica del HPC.
#
# En este notebook (1) exploramos las trayectorias de degradación de
# C-MAPSS FD001–FD004, (2) documentamos la Table 3 del paper con los
# valores de desgaste natural por componente y (3) visualizamos el
# corpus ACE degradado generado por el DegradationInjector.

# %% — Carga del dataset C-MAPSS
loader = CMAPSSLoader(data_path='data/CMAPSS_DATA')
loader.load_all()
loader.summary()

fd001 = loader.load_subset('FD001')
fd001 = loader.compute_health_index(fd001)

print(f"\nFD001: {fd001.shape[0]} filas × {fd001.shape[1]} variables")
print(f"Motores: {fd001['unit_id'].nunique()}")
print(f"Ciclos por motor:")
print(fd001.groupby('unit_id')['cycle'].max().describe().round(0).to_string())

# %% [markdown]
# ## Sección 1: Distribución del RUL por sub-dataset
#
# Cada sub-dataset representa una combinación distinta de condiciones
# operativas y modos de fallo. FD001 y FD003 comparten una sola
# condición operativa (más simples); FD002 y FD004 cubren seis
# condiciones (más complejos y realistas).

# %% — Distribución del RUL
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = ['#2A9D8F', '#E63946', '#457B9D', '#E9C46A']

for ax, subset_name, color in zip(axes.flat,
                                   ['FD001', 'FD002', 'FD003', 'FD004'],
                                   colors):
    df_sub = loader.datasets[subset_name]
    rul_per_unit = df_sub.groupby('unit_id')['RUL'].max()

    ax.hist(rul_per_unit, bins=30, color=color, alpha=0.7, edgecolor='white')
    ax.axvline(rul_per_unit.mean(), color='red', linestyle='--',
               label=f'μ = {rul_per_unit.mean():.0f} ciclos')
    ax.set_xlabel('RUL máximo [ciclos]')
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'{subset_name} - {loader.SUBSETS[subset_name]["description"]}')
    ax.legend()

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig07_rul_distribution.png',
            dpi=300, bbox_inches='tight')
plt.show()
print(f"Guardado: {FIG_DIR / 'fig07_rul_distribution.png'}")

# %% [markdown]
# ## Sección 2: Trayectorias de degradación por sensor
#
# Seleccionamos los seis sensores con mayor variabilidad temporal
# entre motores (excluyendo los sensores constantes identificados en
# `CMAPSSLoader.CONSTANT_SENSORS`) y visualizamos las trayectorias
# de diez motores representativos de FD001.

# %% — Selección automática de sensores más variables
candidate_sensors = [s for s in CMAPSSLoader.INFORMATIVE_SENSORS
                     if s in fd001.columns]

sensor_variability = {}
for s in candidate_sensors:
    grouped = fd001.groupby('unit_id')[s].agg(['min', 'max', 'mean'])
    denom = grouped['mean'].abs().replace(0, 1)
    variation = ((grouped['max'] - grouped['min']) / denom).mean()
    sensor_variability[s] = variation

top_sensors = sorted(sensor_variability,
                     key=sensor_variability.get,
                     reverse=True)[:6]

label_map = {
    'T30':  'T₃₀ HPC outlet [°R]',
    'P30':  'P₃₀ HPC outlet [psia]',
    'T50':  'T₅₀ LPT outlet [°R]',
    'Nc':   'Nc core speed [rpm]',
    'Ps30': 'P_s30 static HPC [psia]',
    'phi':  'φ fuel/air ratio',
    'BPR':  'BPR bypass ratio',
    'W31':  'W31 HPT coolant [lb/s]',
    'W32':  'W32 LPT coolant [lb/s]',
    'T24':  'T₂₄ LPC outlet [°R]',
    'farB': 'far_B combustor',
    'htBleed': 'HT bleed enthalpy',
    'Nf':   'Nf fan speed [rpm]',
    'epr':  'EPR',
}

print(f"Sensores seleccionados por variabilidad: {top_sensors}")

# %% — Trayectorias de degradación
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

units_to_plot = fd001['unit_id'].unique()[:10]
color_palette = plt.cm.tab10(np.linspace(0, 1, len(units_to_plot)))

for ax, sensor in zip(axes.flat, top_sensors):
    for unit, c in zip(units_to_plot, color_palette):
        unit_data = fd001[fd001['unit_id'] == unit]
        ax.plot(unit_data['cycle'], unit_data[sensor],
                alpha=0.7, linewidth=0.9, color=c)
    ax.set_xlabel('Ciclos de operación')
    ax.set_ylabel(label_map.get(sensor, sensor))
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig08_degradation_trajectories.png',
            dpi=300, bbox_inches='tight')
plt.show()
print(f"Guardado: {FIG_DIR / 'fig08_degradation_trajectories.png'}")

# %% [markdown]
# ## Sección 3: Health index y tendencia media de degradación
#
# El Health Index (HI) se calcula con clipping en 125 ciclos siguiendo
# la práctica estándar de la literatura (Heimes 2008, Zheng et al.
# 2017): permanece saturado a 1 durante la operación normal y
# transiciona linealmente hacia 0 en los últimos ciclos hasta el
# fallo.

# %% — Health index y tendencias medias
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for unit in units_to_plot:
    unit_data = fd001[fd001['unit_id'] == unit]
    axes[0].plot(unit_data['cycle'], unit_data['HI'],
                 alpha=0.5, linewidth=0.8)
axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Fallo')
axes[0].set_xlabel('Ciclos de operación')
axes[0].set_ylabel('Health index [0, 1]')
axes[0].legend()

fd001_norm = loader.normalize(fd001.copy(), method='minmax')
cycle_means = fd001_norm.groupby('cycle')[['T30', 'P30', 'Nf', 'T50']].mean()
for col in ['T30', 'P30', 'Nf', 'T50']:
    axes[1].plot(cycle_means.index, cycle_means[col], label=col, linewidth=1.5)
axes[1].set_xlabel('Ciclos de operación')
axes[1].set_ylabel('Valor normalizado [0, 1]')
axes[1].legend()

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig09_health_index.png',
            dpi=300, bbox_inches='tight')
plt.show()
print(f"Guardado: {FIG_DIR / 'fig09_health_index.png'}")

# %% — Análisis de varianza por sensor
print("Análisis de varianza por sensor:")
print(f"{'Sensor':<12} {'Varianza':>14}  Categoría")

sensor_names = [c for c in loader.COLUMN_NAMES[5:] if c in fd001.columns]
sensor_vars = fd001[sensor_names].var().sort_values(ascending=False)
for sensor, var in sensor_vars.items():
    category = 'informativo' if var > 0.1 else 'constante (candidato a filtrar)'
    print(f"{sensor:<12} {var:>14.4f}  {category}")

# %% [markdown]
# ## Sección 4: Table 3 de Saxena (2008)
#
# La Table 3 del paper documenta el desgaste natural de fondo por
# componente (Fan, LPC, HPT, LPT). Estos valores son deterministas 
# y comunes a todos los motores del dataset.
#
# Es importante notar que la Table 3 **NO incluye el HPC**: en el
# paper el HPC es el fault mode principal del dataset y se modela con
# la ecuación exponencial estocástica h(t) = 1 - exp(-a·t^b), con
# parámetros a ∈ [0.001, 0.003] y b ∈ [1.4, 1.6] muestreados por
# motor. Esta distinción está implementada en el DegradationInjector.

# %% — Table 3 de Saxena
wear_table = {
    'Component': ['Fan_Efficiency', 'Fan_Flow',
                  'LPC_Efficiency', 'LPC_Flow',
                  'HPT_Efficiency', 'HPT_Flow',
                  'LPT_Efficiency', 'LPT_Flow'],
    'Initial (%)':      [-0.18, -0.26, -0.62, -1.01, -0.48, +0.08, -0.10, +0.08],
    '3000 cycles (%)':  [-1.50, -2.04, -1.46, -2.08, -2.63, +1.76, -0.54, +0.26],
    '6000 cycles (%)':  [-2.85, -3.65, -2.61, -4.00, -3.81, +2.57, -1.08, +0.42],
}

wear_df = pd.DataFrame(wear_table)
print("Table 3 de Saxena (2008): desgaste natural por componente")
print(wear_df.to_string(index=False))
wear_df.to_csv(TABLE_DIR / 'saxena_wear_table.csv', index=False)
print(f"\nGuardado: {TABLE_DIR / 'saxena_wear_table.csv'}")

# %% [markdown]
# ## Sección 5: Mapeo C-MAPSS → ACE
#
# Correspondencia paramétrica entre los componentes degradables de
# C-MAPSS y los actuadores/estaciones del motor ACE de tres flujos.
# Este mapeo es la base de la interpretabilidad del corpus degradado.

# %% — Tabla de mapeo
print("Mapeo C-MAPSS → actuadores ACE")
print(f"{'Componente':<12} {'Sensores C-MAPSS':<30} "
      f"{'Actuador ACE':<15} Descripción")

for comp, info in CMAPSSLoader.CMAPSS_TO_ACE_MAP.items():
    sensors = ', '.join(info['cmapss_sensors'])
    actuator = info['ace_actuator'] or '—'
    print(f"{comp:<12} {sensors:<30} {actuator:<15} {info['description']}")

# %% [markdown]
# ## Sección 6: Corpus ACE degradado
#
# El pipeline (`python src/pipeline.py`) genera dos corpus:
#
#   - `data/synthetic/ace_dataset_5000.csv`: corpus nominal
#   - `data/processed/ace_dataset_5000_degraded.csv`: corpus con
#     degradación inyectada por `DegradationInjector`
#
# Este bloque carga el corpus degradado y visualiza los deltas
# aplicados por componente y su relación con el health index.

# %% — Carga del corpus degradado
degraded_path = Path('data/processed/ace_dataset_5000_degraded.csv')
ace_degraded = pd.read_csv(degraded_path)

print(f"Corpus degradado: {ace_degraded.shape[0]} filas × "
      f"{ace_degraded.shape[1]} variables")
print(f"Ciclos de vida: [{ace_degraded['life_cycle'].min()}, "
      f"{ace_degraded['life_cycle'].max()}]")
print(f"Health index:   [{ace_degraded['HI'].min():.3f}, "
      f"{ace_degraded['HI'].max():.3f}]")

# %% — Visualización de la degradación inyectada
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Panel 1: Distribución del ciclo de vida
axes[0, 0].hist(ace_degraded['life_cycle'], bins=30,
                color='#2A9D8F', alpha=0.7, edgecolor='white')
axes[0, 0].set_xlabel('Ciclo de vida asignado')
axes[0, 0].set_ylabel('Frecuencia')

# Panel 2: Health index vs ciclo
sc1 = axes[0, 1].scatter(ace_degraded['life_cycle'], ace_degraded['HI'],
                          c=ace_degraded['life_cycle'], cmap='RdYlGn',
                          s=10, alpha=0.6)
axes[0, 1].set_xlabel('Ciclo de vida')
axes[0, 1].set_ylabel('Health index')
plt.colorbar(sc1, ax=axes[0, 1], label='Ciclo')

# Panel 3: Delta HPC (modelo exponencial estocástico)
if 'hpc_eff_delta' in ace_degraded.columns:
    axes[0, 2].scatter(ace_degraded['life_cycle'],
                       ace_degraded['hpc_eff_delta'] * 100,
                       c='#E63946', s=10, alpha=0.5)
    axes[0, 2].set_xlabel('Ciclo de vida')
    axes[0, 2].set_ylabel('Δη HPC [%]')

# Panel 4: Nf original vs degradado
if 'Nf_degraded' in ace_degraded.columns:
    axes[1, 0].scatter(ace_degraded['Nf'], ace_degraded['Nf_degraded'],
                       c=ace_degraded['life_cycle'], cmap='plasma',
                       s=10, alpha=0.5)
    lims = [ace_degraded['Nf'].min(), ace_degraded['Nf'].max()]
    axes[1, 0].plot(lims, lims, 'r--', linewidth=1,
                    label='Sin degradación')
    axes[1, 0].set_xlabel('Nf original [rpm]')
    axes[1, 0].set_ylabel('Nf degradado [rpm]')
    axes[1, 0].legend()

# Panel 5: T30 original vs degradado
if 'T30_degraded' in ace_degraded.columns:
    axes[1, 1].scatter(ace_degraded['T30'], ace_degraded['T30_degraded'],
                       c=ace_degraded['life_cycle'], cmap='plasma',
                       s=10, alpha=0.5)
    lims = [ace_degraded['T30'].min(), ace_degraded['T30'].max()]
    axes[1, 1].plot(lims, lims, 'r--', linewidth=1,
                    label='Sin degradación')
    axes[1, 1].set_xlabel('T₃₀ original [°R]')
    axes[1, 1].set_ylabel('T₃₀ degradado [°R]')
    axes[1, 1].legend()

# Panel 6: Severidad de degradación por componente
delta_cols = ['fan_eff_delta', 'hpc_eff_delta',
              'hpt_eff_delta', 'lpt_eff_delta']
if all(c in ace_degraded.columns for c in delta_cols):
    comp_deltas = ace_degraded[delta_cols].abs().mean() * 100
    axes[1, 2].barh(['Fan', 'HPC', 'HPT', 'LPT'], comp_deltas.values,
                    color=['#2A9D8F', '#E63946', '#E9C46A', '#457B9D'])
    axes[1, 2].set_xlabel('Degradación media absoluta [%]')
    for i, v in enumerate(comp_deltas.values):
        axes[1, 2].text(v + 0.01, i, f'{v:.3f}%',
                        va='center', fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig10_degradation_injection.png',
            dpi=300, bbox_inches='tight')
plt.show()
print(f"Guardado: {FIG_DIR / 'fig10_degradation_injection.png'}")

# %% [markdown]
# ## Sección 7: Comparación C-MAPSS vs ACE
#
# Comparamos las distribuciones de los sensores comunes entre C-MAPSS
# (FD001) y el corpus ACE. Los rangos operativos difieren
# significativamente porque C-MAPSS FD001 simula un turbofán civil en 
# UNA sola condición operativa (nivel del mar, crucero), mientras que 
# el corpus ACE cubre 5 fases de misión (Mach 0.1-1.8, altitud 0-42k ft).
#
# Esta divergencia NO invalida el uso de Saxena como fuente: transferimos
# los PATRONES TEMPORALES RELATIVOS de Table 3 (porcentajes de
# degradación acumulada a 3000 y 6000 ciclos), no los valores absolutos
# de los sensores. Los mecanismos físicos de degradación (erosión de
# álabes, fouling, deformación térmica) son universales para turbofanes,
# aunque los rangos operativos difieran entre motor civil (C-MAPSS) y
# motor militar adaptativo (ACE). 
#
# Las limitaciones de esta transferencia (degradación acelerada en 
# régimen combat, subestimación potencial en fases de alta demanda) 
# están documentadas en la Sección 6 del notebook corpus_analysis.py.

# %% — Distribuciones comparadas
common_sensors = ['T30', 'P30', 'Nf', 'Nc', 'T50', 'epr']
common_available = [s for s in common_sensors
                    if s in fd001.columns and s in ace_degraded.columns]

MIN_FRAC = 0.012          # anchura mínima de un soporte, en fracción del panel
SERIES = [('C-MAPSS FD001', '#457B9D'), ('ACE sintético', '#E63946')]


def _fmt_range(x):
    """Formatea un extremo de rango evitando la notación científica."""
    return f'{x:,.2f}'.rstrip('0').rstrip('.') if abs(x) < 100 else f'{x:,.0f}'

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for ax, sensor in zip(axes.flat, common_available):
    data = [fd001[sensor].dropna().values,
            ace_degraded[sensor].dropna().values]

    lo = min(v.min() for v in data)
    hi = max(v.max() for v in data)
    pad = 0.05 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.05
    xlo, xhi = lo - pad, hi + pad
    span = xhi - xlo

    for vals, (label, color) in zip(data, SERIES):
        width = vals.max() - vals.min()
        if width < MIN_FRAC * span:
            center = 0.5 * (vals.max() + vals.min())
            ax.bar(center, 1.0, width=MIN_FRAC * span, align='center',
                   color=color, alpha=0.60, edgecolor='white',
                   linewidth=0.5, label=label, zorder=3)
        else:
            nbins = int(np.clip(width / (0.004 * span), 6, 30))
            counts, edges = np.histogram(vals, bins=nbins)
            counts = counts / counts.max()
            ax.bar(edges[:-1], counts, width=np.diff(edges), align='edge',
                   color=color, alpha=0.60, edgecolor='white',
                   linewidth=0.4, label=label, zorder=2)

    ax.text(0.98, 0.97,
            '\n'.join(f'{label.split()[0]}: '
                      f'{_fmt_range(v.min())} – {_fmt_range(v.max())}'
                      for v, (label, _) in zip(data, SERIES)),
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.8',
                      alpha=0.9))

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0, 1.30)
    ax.set_xlabel(sensor)
    ax.set_ylabel('Frecuencia relativa (por dominio)')
    ax.grid(alpha=0.3)

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=True,
           bbox_to_anchor=(0.5, 1.02))

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig11_cmapss_vs_ace.png',
            dpi=300, bbox_inches='tight')
plt.show()
print(f"Guardado: {FIG_DIR / 'fig11_cmapss_vs_ace.png'}")

# %% — Tabla de solapamiento de rangos
print(f"{'Sensor':<8} {'C-MAPSS min':>12} {'C-MAPSS max':>12} "
      f"{'ACE min':>12} {'ACE max':>12} {'Solape':>8}")

overlap_rows = []
for sensor in common_available:
    c_min, c_max = fd001[sensor].min(), fd001[sensor].max()
    a_min, a_max = ace_degraded[sensor].min(), ace_degraded[sensor].max()

    ov_min = max(c_min, a_min)
    ov_max = min(c_max, a_max)
    total = max(c_max, a_max) - min(c_min, a_min)
    ov_pct = max(0, (ov_max - ov_min) / total * 100) if total else 0

    print(f"{sensor:<8} {c_min:>12.2f} {c_max:>12.2f} "
          f"{a_min:>12.2f} {a_max:>12.2f} {ov_pct:>7.1f}%")
    overlap_rows.append({
        'sensor': sensor,
        'cmapss_min': round(c_min, 2),
        'cmapss_max': round(c_max, 2),
        'ace_min': round(a_min, 2),
        'ace_max': round(a_max, 2),
        'overlap_pct': round(ov_pct, 2),
    })

pd.DataFrame(overlap_rows).to_csv(TABLE_DIR / 'range_overlap.csv',
                                   index=False)

# %% — Correlación degradación-Health Index
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fd001_sample = fd001.sample(min(2000, len(fd001)), random_state=42)
for sensor, color in zip(['T30', 'Nf'], ['#457B9D', '#2A9D8F']):
    axes[0].scatter(fd001_sample[sensor], fd001_sample['HI'],
                    alpha=0.2, s=5, c=color, label=sensor)
axes[0].set_xlabel('Valor del sensor')
axes[0].set_ylabel('Health index')
axes[0].legend()

if 'hpc_eff_delta' in ace_degraded.columns:
    axes[1].scatter(ace_degraded['hpc_eff_delta'] * 100, ace_degraded['HI'],
                    alpha=0.4, s=10, c='#E63946', label='Δη HPC')
if 'fan_eff_delta' in ace_degraded.columns:
    axes[1].scatter(ace_degraded['fan_eff_delta'] * 100, ace_degraded['HI'],
                    alpha=0.4, s=10, c='#2A9D8F', label='Δη Fan')
axes[1].set_xlabel('Degradación eficiencia [%]')
axes[1].set_ylabel('Health index')
axes[1].legend()

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig12_degradation_hi_correlation.png',
            dpi=300, bbox_inches='tight')
plt.show()
print(f"Guardado: {FIG_DIR / 'fig12_degradation_hi_correlation.png'}")

# %% — Resumen ejecutivo
print(f"""
Dataset C-MAPSS:
  Sub-datasets:         4 (FD001–FD004)
  Motores totales:      709
  Sensores informativos: {len(CMAPSSLoader.INFORMATIVE_SENSORS)}
  Sensores constantes:   {len(CMAPSSLoader.CONSTANT_SENSORS)}

Protocolo de inyección (implementado en DegradationInjector):
  Fan, LPC, HPT, LPT: interpolación exponencial de Table 3 de Saxena
  HPC:                modelo exponencial estocástico h(t) = 1 - exp(-a·t^b)
                      con a ∈ [0.001, 0.003] y b ∈ [1.4, 1.6]

Corpus ACE degradado:
  Archivo:      data/processed/ace_dataset_5000_degraded.csv
  Dimensiones:  {ace_degraded.shape[0]} × {ace_degraded.shape[1]}

Salidas generadas:
  - fig07_rul_distribution.png            Distribución RUL por sub-dataset
  - fig08_degradation_trajectories.png    Trayectorias por sensor (10 motores)
  - fig09_health_index.png                Health index y tendencias medias
  - fig10_degradation_injection.png       Deltas inyectados y health index
  - fig11_cmapss_vs_ace.png               Distribuciones comparadas
  - fig12_degradation_hi_correlation.png  Correlación degradación–HI
  - saxena_wear_table.csv                 Table 3 de Saxena
  - range_overlap.csv                     Solapamiento de rangos C-MAPSS vs ACE
""")