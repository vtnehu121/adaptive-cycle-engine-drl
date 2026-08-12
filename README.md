# Control Multiobjetivo de Motores de Ciclo Adaptativo de Tercer Flujo mediante Arquitectura Híbrida PINN–DRL


[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.12](https://img.shields.io/badge/PyTorch-2.12.1-ee4c2c.svg)](https://pytorch.org/)
[![Stable-Baselines3 2.9](https://img.shields.io/badge/SB3-2.9.0-green.svg)](https://stable-baselines3.readthedocs.io/)
[![MATLAB R2026a](https://img.shields.io/badge/MATLAB-R2026a-orange.svg)](https://www.mathworks.com/)
[![T-MATS](https://img.shields.io/badge/NASA-T--MATS-red.svg)](https://github.com/nasa/T-MATS)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Trabajo de Fin de Grado - Grado en Computación e Inteligencia Artificial**
Universidad Alfonso X el Sabio (UAX), Madrid · Curso 2025–2026

Sistema de control por Deep Reinforcement Learning para los cuatro actuadores de geometría variable de un **motor de ciclo adaptativo (ACE) de tres flujos**. El agente aprende sobre un **gemelo digital PINN** que sustituye al simulador termodinámico T-MATS, reduciendo el coste de cada consulta al motor de segundos a **1,84 ms (P99)** y haciendo así viable el entrenamiento por refuerzo. Una capa Safe-RL con barreras logarítmicas mantiene la política dentro de los límites termomecánicos, y un módulo de Health Monitoring con RNN bayesianas aporta la estimación de deterioro sobre la que el control se adapta.

---

## Tabla de contenidos

1. [Motivación y problema](#1-motivación-y-problema)
2. [Enfoque técnico](#2-enfoque-técnico)
3. [Datasets](#3-datasets)
4. [Requisitos](#4-requisitos)
5. [Estructura del proyecto](#5-estructura-del-proyecto)
6. [Instalación](#6-instalación)
7. [Uso: cómo ejecutar el proyecto](#7-uso-cómo-ejecutar-el-proyecto)
8. [Resultados principales](#8-resultados-principales)
9. [Referencias académicas](#9-referencias-académicas)
10. [Autora y contacto](#10-autora-y-contacto)

---

## 1. Motivación y problema

### ¿Qué problema resuelve?

Un turbofán convencional tiene una geometría fija: su punto de máxima eficiencia queda determinado en el diseño y todo lo demás es compromiso. Un **motor de ciclo adaptativo** rompe esa limitación añadiendo un **tercer flujo** de aire cuyo reparto puede modificarse en vuelo, junto con álabes de geometría variable en ventilador y turbina. El motor puede así comportarse como un turbofán de alto índice de derivación en crucero (bajo consumo) y como un turbojet en combate (alto empuje), y usar el tercer flujo como sumidero térmico de la aeronave.

Esa flexibilidad tiene un precio: **cuatro actuadores acoplados que hay que coordinar simultáneamente** (VFGV, VGV-T, postcombustión y sangrado), optimizando a la vez cuatro objetivos que compiten entre sí — empuje, consumo específico, potencia eléctrica extraída y capacidad de refrigeración — sin violar en ningún momento los límites de temperatura de turbina ni los márgenes de bombeo de los compresores.

El controlador industrial estándar, el **FADEC**, resuelve esto con *gain scheduling*: tablas de consulta con posiciones fijas por fase de vuelo. Funciona, pero es rígido — no reacciona ante la degradación progresiva del motor ni ante puntos de operación fuera de la tabla.

### ¿Por qué es importante?

Este trabajo demuestra empíricamente dos cosas que motivan la propuesta:

- **El FADEC colapsa donde el modelo lo permite.** Evaluado contra un modelo simplificado del motor obtiene un reward medio de **288,3** con seguridad del 100%. El mismo controlador, evaluado contra el gemelo digital PINN, cae a **−1 947** con una tasa de episodios limpios del **12%**: en combate maximiza empuje (125 364 lbf) sacrificando la integridad térmica del motor (T₄ media de 3 620 °R frente al límite de 3 200 °R, con violación en el 99,5% de los pasos). El gemelo digital hace visible una limitación que un modelo simplificado oculta.
- **El agente aprendido descubre el compromiso.** TD3+BC obtiene en combate 72 496 lbf con T₄ media de 3 113 °R y **cero violaciones** sobre 4 000 pasos. Renuncia a un 42% del empuje a cambio de operar dentro de la envolvente segura y preservar la vida del motor.

Además, entrenar directamente sobre T-MATS es inviable: un algoritmo off-policy necesita centenares de miles de interacciones y cada simulación en Simulink cuesta segundos. De ahí la necesidad del gemelo digital: no es un adorno metodológico, es la condición que hace posible el resto del trabajo.

---

## 2. Enfoque técnico

El sistema se compone de tres módulos entrenados por separado y acoplados en un único entorno de control.

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Motor ACE de tres flujos (T-MATS)                 │
│   Inlet → Fan → LPC → HPC → Burner → HPT → LPT → Nozzle             │
│              └─ Splitter ─┬─ Bypass  → Nozzle bypass                │
│                           └─ Tercer flujo (VFGV) → Nozzle TS        │
└────────────────────────────┬────────────────────────────────────────┘
                             │  corpus sintético 5 000 × 27
                             ▼
        ┌────────────────────────────────┐   ┌──────────────────────────┐
        │  Gemelo Digital  BraytonPINN   │   │  Health Monitoring RNN   │
        │  224×8, 838 443 parámetros     │   │  GRU-64, cabezal dual    │
        │  3 constraint layers exactas   │   │  RUL + degradación       │
        │  6 residuos PDE Brayton        │   │  MC-Dropout (N=50)       │
        │  MAPE 4,64% · P99 1,84 ms      │   │  RMSE 13,94 ciclos       │
        └───────────────┬────────────────┘   └────────────┬─────────────┘
                        │                                 │ health index
                        ▼                                 ▼
        ┌────────────────────────────────────────────────────────────┐
        │            ACEEnv  (Gymnasium, CMDP)                       │
        │  Acción: 4 actuadores continuos ∈ [-1, 1]                  │
        │  Observación: 19 variables normalizadas                    │
        │  Recompensa multiobjetivo con pesos por fase de misión     │
        │  Safe-RL: barrera log en T₄ + márgenes de bombeo           │
        └───────────────────────────┬────────────────────────────────┘
                                    │
          ┌─────────────┬───────────┴───────────┬──────────────┐
          ▼             ▼                       ▼              ▼
      FADEC          SAC                     TD3           TD3 + BC
   (baseline)                                            (mejor política)
```

### 2.1 Gemelo digital: `BraytonPINN`

Red **Physics-Informed Neural Network** (PyTorch puro, sin frameworks intermedios) que aprende el mapa `(altitud, Mach, TRA, bpr_ts) → 14 variables termodinámicas`.

- **Backbone residual** de 8 bloques `LayerNorm → Linear → GELU → Dropout → Linear`, dimensión oculta 224 (838 443 parámetros, 3,35 MB).
- **Tres *constraint layers*.** `farB`, `P2` y `SFC` no se aprenden: se calculan como identidades exactas (scheduling FADEC, presión total ISA con corrección de Mach, y definición operativa del consumo). La red dedica toda su capacidad a las once variables no triviales. La ablación demuestra que esta decisión vale un **57% de mejora en MAPE**.
- **Pérdida física de seis residuos PDE** del ciclo Brayton (numerados PDE 1 – PDE 6 en `pinn.py`): relación isentrópica del compresor, balance energético del eje, momento de la tobera, eficiencia de combustión, conservación de masa en flujo bloqueado y balance del splitter de tres vías. Más regularizadores de monotonía (T₂ < T₃₀ < T₄ > T₅₀), presión, positividad, jerarquía de ejes y rango de OPR.
- **Puntos de colocación.** La pérdida física se evalúa además sobre 64 puntos aleatorios del dominio operativo por batch (`CollocationSampler(n_points=64)`), donde no existen datos. Es lo que distingue una PINN de una red con regularización termodinámica *a posteriori*. El dominio de colocación es **deliberadamente más amplio que la envolvente de entrenamiento** (0 – 50 000 ft, Mach 0 – 2,0, TRA 20 – 100%, bpr_ts 0 – 0,6), de modo que la física también restrinja al modelo en la zona que después se somete a las pruebas OOD (§8.3).
- **Currículo progresivo** de los pesos ω₂ y ω₃ durante el *warmup*, para evitar las patologías de gradiente descritas por Wang et al. (2021) cuando los términos físicos entran de golpe.

### 2.2 Health Monitoring: `HealthMonitoringRNN`

Red recurrente con **cabezal dual**: predice a la vez la vida útil remanente (RUL) y un vector de degradación de cuatro componentes (fan, HPC, HPT, LPT). Saber *cuánta* vida queda no basta para el control adaptativo; hace falta saber *qué* se está degradando.

- Ventanas deslizantes de 30–50 ciclos sobre 15 sensores informativos de C-MAPSS.
- **Pseudo-etiquetas de degradación** derivadas del delta relativo de Nf, T30, T50 y Nc frente al promedio de los 10 primeros ciclos de cada motor (C-MAPSS no publica el estado real por componente, solo la RUL).
- Pérdida dual `L = L_RUL + 0,3·L_degradación`.
- **Incertidumbre epistémica por MC-Dropout** (Gal & Ghahramani, 2016) con 50 pases estocásticos, y verificación de la calibración del intervalo al 95%.
- Partición **por motor**, no por fila, para eliminar la fuga de información entre ventanas del mismo motor.

### 2.3 Control: `ACEEnv` + agentes

El problema se formula como **CMDP** (proceso de decisión de Markov con restricciones): la seguridad no es un objetivo más que pueda compensarse con empuje, sino una restricción contabilizada aparte.

- **Espacio de acción**: 4 continuos normalizados en [−1, 1] → VFGV [−20°, +20°], VGV-T [−15°, +15°], postcombustión [0, 1], sangrado [0,01, 0,10].
- **Espacio de observación**: 19 variables normalizadas (10 de estado termodinámico, 4 de posición de actuadores, 3 de condición de vuelo, fase codificada y health index).
- **Recompensa multiobjetivo con pesos dependientes de la fase.** El compromiso cambia por completo a lo largo de la misión: en combate manda el empuje (w = 0,45), en crucero el consumo (w = 0,40).
- **Safe-RL**: barrera logarítmica sobre T₄ que empieza a penalizar en la zona de advertencia (3 000 °R), mucho antes del límite duro (3 200 °R). `SafetyLayer` aplica esa misma barrera superior a **T₄, Nf y Nc**, y castiga el empuje negativo; el ∞ teórico de la barrera se sustituye por una penalización finita (`PENALTY_HARD = 10.0`, más −50 al detectar violación) para que un único paso inseguro no invalide el episodio entero durante el aprendizaje. Los **márgenes de bombeo** entran aparte, como penalización lineal sobre el *proxy* `sm_fan`/`sm_hpc` dentro de la recompensa de `ACEEnv`, no como barrera logarítmica: el proxy no es lo bastante preciso para justificar una barrera diferenciable.
- El episodio **nunca termina** por violación, solo se trunca al agotar los pasos: cortarlo daría al agente el incentivo perverso de provocar una violación para escapar de un estado de baja recompensa. Cada paso seguro suma además un *survival bonus* de +1.
- **Agentes comparados**: SAC, TD3, y un pipeline híbrido **TD3+BC** en tres fases — el FADEC genera 40 000 demostraciones, se clona su comportamiento en el actor por regresión MSE, y se hace *fine-tuning* por refuerzo con el replay buffer pre-poblado.

---

## 3. Datasets

| Dataset | Origen | Contenido | Ubicación |
|---|---|---|---|
| **Corpus sintético ACE** | Generado con T-MATS (NASA Glenn) sobre el modelo `ACE_3stream_brayton.slx` | 5 000 puntos de operación × 27 variables termodinámicas, muestreo estratificado en 5 fases de misión | `data/synthetic/ace_dataset_5000.csv` |
| **Condiciones de vuelo** | `FlightEnvelope`, muestreo estratificado por fase con semilla 42 | 5 000 tuplas (altitud, Mach, TRA, bpr_ts, bleed) | `data/synthetic/flight_conditions_5000.csv` |
| **Corpus degradado** | `DegradationInjector` sobre el corpus nominal | Corpus + deltas de degradación por componente + health index | `data/processed/ace_dataset_5000_degraded.csv` |
| **NASA C-MAPSS** | Saxena, Goebel & Simon (PHM 2008) | FD001–FD004: 709 motores simulados *run-to-failure* | `data/CMAPSS_DATA/` |
| **Demostraciones experto** | 200 episodios del FADEC sobre `ACEEnv` | 40 000 transiciones (o, a, r, o', d) para behavioral cloning | `data/expert/expert_data.npz` |

**Los cinco vienen incluidos en el repositorio.** No hace falta descargar nada ni ejecutar MATLAB para reproducir los resultados: el corpus ya está generado y los checkpoints entrenados también.

> **Nomenclatura de los ficheros del corpus.** `src/pipeline.py` construye los nombres a partir del número de muestras solicitado: `ace_dataset_{n}.csv`, `flight_conditions_{n}.csv` y `ace_dataset_{n}_degraded.csv`. Con el valor por defecto (`--n-samples 5000`) resultan exactamente los ficheros versionados; una prueba rápida con `--n-samples 100` genera `ace_dataset_100.csv` **sin sobrescribir** el corpus oficial, pero los scripts de evaluación y entrenamiento seguirán apuntando por defecto al de 5 000.

### Envolvente de vuelo muestreada

| Fase | Peso | Altitud (ft) | Mach | TRA (%) | bpr_ts |
|---|---|---|---|---|---|
| Takeoff | 15% | 0 – 2 000 | 0,10 – 0,35 | 90 – 100 | 0,10 – 0,30 |
| Climb | 20% | 2 000 – 25 000 | 0,40 – 0,70 | 80 – 95 | 0,20 – 0,40 |
| Cruise | 25% | 25 000 – 42 000 | 0,75 – 0,90 | 70 – 85 | 0,30 – 0,50 |
| Combat | 25% | 5 000 – 25 000 | 0,80 – 1,80 | 95 – 100 | 0,05 – 0,20 |
| Descent | 15% | 5 000 – 20 000 | 0,30 – 0,60 | 30 – 60 | 0,30 – 0,50 |

Cada fase define un **hiperrectángulo** en el espacio de parámetros y dentro de él el muestreo es **uniforme e independiente en cada dimensión**; no se usa hipercubo latino. Los rangos corresponden a la envolvente operativa típica de un caza multirrol de 4ª generación (F/A-18E/F, F-16), y los pesos están calibrados para garantizar cobertura suficiente del envelope durante el entrenamiento del gemelo digital — mayor densidad en las fases de alta variabilidad (cruise, combat) y menor en las transitorias cortas (takeoff, descent) — no para reproducir la fracción de tiempo real de cada fase.

### Modelo de degradación

Adapta el protocolo de Saxena et al. (2008) a la topología de tres flujos combinando dos mecanismos:

1. **Desgaste natural determinista** (Table 3 del paper) para fan, LPC, HPT y LPT. La curva es un perfil monótono `t^b` **anclado en los dos extremos publicados** (inicial en t = 0 y 6 000 ciclos en t = 1). El punto intermedio de 3 000 ciclos se conserva en `WEAR_TABLE` por trazabilidad con el paper, pero no se impone como restricción: la trayectoria se desvía de él en un 20 – 30% (p. ej. `fan_eff` a t = 0,5 da ≈ −1,12% frente al −1,50% tabulado).
2. **Degradación exponencial estocástica** `h(t) = 1 − exp(−a·t^b)` con `a ∈ [0,001, 0,003]` y `b ∈ [1,4, 1,6]` muestreados por motor, reservada al **HPC** como modo de fallo dominante — igual que en el dataset original. Se aplica sobre **t normalizado en [0, 1]**, no sobre ciclos absolutos como en el paper: los parámetros `a` están recalibrados para ese esquema, de modo que los deltas del HPC queden en el mismo orden de magnitud (~0,2%) que el desgaste natural tabulado.

Solo cuatro de los deltas se propagan a las variables termodinámicas del corpus (`fan_eff → Nf`, `hpc_eff → T30, P30`, `hpt_eff → T50`, `lpt_eff → Nc`), que son los que la RNN predice de forma agregada. Los restantes (`fan_flow`, `lpc_eff/flow`, `hpt_flow`, `lpt_flow`, `hpc_flow`) se calculan y se guardan como columnas por trazabilidad con la Table 3 completa, pero no se propagan.

---

## 4. Requisitos

### 4.1 Software

| Herramienta | Versión en `requirements.txt` | Necesaria para | ¿Obligatoria? |
|---|---|---|---|
| Python | 3.10 (probado en 3.10.0) | Todo el código | Sí |
| PyTorch | `==2.12.1` (fijada) | PINN, RNN, agentes DRL | Sí |
| Stable-Baselines3 | `==2.9.0` (fijada) | SAC, TD3, TD3+BC | Sí |
| Gymnasium | `==1.3.0` (fijada) | Entorno `ACE-v0` | Sí |
| scikit-learn | `==1.7.2` (fijada) | Normalización de C-MAPSS | Sí |
| NumPy | `==2.2.6` (fijada) | Manipulación de datos | Sí |
| pandas | `>=2.0.0` (entorno de referencia: 2.3.3) | Manipulación de datos | Sí |
| matplotlib / seaborn | `>=3.7.0` / `>=0.12.0` (referencia: 3.10.9 / 0.13.2) | Figuras | Sí |
| TensorBoard | `>=2.20.0` | Logging de entrenamiento DRL | Sí |
| tqdm / rich | `>=4.66.0` / `>=13.0.0` | Barra de progreso de SB3 (`progress_bar=True`) | Sí |
| pytest / pytest-cov | `>=7.4.0` / `>=4.1.0` | Tests unitarios y cobertura | Sí |
| MATLAB + Simulink | R2026a | Ejecutar T-MATS | **Solo** para regenerar el corpus |
| T-MATS | v1.2 (NASA Glenn) | Simulador termodinámico | **Solo** para regenerar el corpus |
| MATLAB Engine for Python | `>=26.1` (debe casar con la release de MATLAB) | Puente Python ↔ MATLAB | **Solo** para regenerar el corpus |

Las versiones marcadas como *fijadas* (`==`) son las que determinan los resultados numéricos y garantizan que los checkpoints guardados se recarguen y reproduzcan las mismas cifras; el resto usa `>=` por no influir en ellos.

> El corpus y los checkpoints están versionados en el repositorio. **MATLAB solo es necesario si quieres regenerar los datos desde cero.** Todo lo demás — entrenamiento de PINN, RNN y agentes, evaluación y generación de las 32 figuras — funciona únicamente con Python.

### 4.2 Hardware

Equipo de referencia con el que se han generado **todos** los resultados de este trabajo:

| Componente | Especificación | Uso |
|---|---|---|
| CPU | Intel Core i7-12650H (10 núcleos / 16 hilos) | Pipeline de datos, interfaz Python–T-MATS, entrenamiento del PINN y de la GRU-64, entrenamiento y evaluación de los agentes DRL |
| GPU | Intel UHD Graphics (integrada) | **No utilizada para cómputo tensorial** |
| RAM | 16 GB DDR5 | Corpus en memoria, replay buffer del DRL |
| Disco | ~3 GB libres | Repositorio, entorno virtual y checkpoints |

> **Todo el entrenamiento y la evaluación se ejecutan íntegramente en CPU.** No se ha empleado aceleración por GPU en ningún punto del proyecto: el equipo solo dispone de gráficos integrados Intel, sin soporte CUDA. Los módulos seleccionan el dispositivo con `device='auto'`, que al no detectar CUDA recae en CPU de forma automática.
>
> Esto no es una limitación del trabajo sino un resultado en sí mismo: el gemelo digital alcanza **1,84 ms de latencia en el percentil 99 sobre CPU** (§8.2), de modo que el presupuesto temporal del lazo de control se cumple **sin necesidad de hardware acelerador**, que es precisamente lo relevante para un despliegue embarcado. El coste está en el tiempo de entrenamiento, no en el de inferencia.

No hace falta ninguna GPU para reproducir el proyecto. Quien disponga de una tarjeta NVIDIA puede acelerar el entrenamiento sin tocar el código (la detección de dispositivo es automática); las instrucciones opcionales están en el paso 5 de cada guía de instalación.

---

## 5. Estructura del proyecto

```
adaptive-cycle-engine-drl/
│
├── data/                              # Datos (todos incluidos en el repositorio)
│   ├── CMAPSS_DATA/                   # NASA C-MAPSS FD001–FD004 + RUL
│   ├── synthetic/                     # Corpus nominal generado por T-MATS
│   │   ├── ace_dataset_5000.csv
│   │   └── flight_conditions_5000.csv
│   ├── processed/                     # Corpus con degradación inyectada
│   │   └── ace_dataset_5000_degraded.csv
│   └── expert/                        # Demostraciones del FADEC para BC
│       └── expert_data.npz
│
├── models/                            # Modelo Simulink y scripts MATLAB
│   ├── ACE_3stream_brayton.slx        # Modelo ACE de tres flujos (ya construido)
│   ├── init_ACE.m                     # Orquestador: carga T-MATS + mapas JT9D
│   ├── setup_ACE_params.m             # Parámetros físicos del ACE (estructura ACE.*)
│   ├── modify_JT9D_to_ACE.m           # Construcción única: inserta los bloques del tercer flujo
│   └── Simsetup/                      # Mapas de componentes (fan, LPC, HPC, HPT, LPT…)
│
├── src/                               # Código fuente Python
│   ├── pipeline.py                    # Orquestador de las 5 etapas de datos
│   ├── data_gen/
│   │   ├── flight_envelope.py         # Muestreo estratificado del envelope
│   │   └── degradation.py             # Inyección de degradación C-MAPSS
│   ├── simulators/
│   │   └── tmats_interface.py         # Puente Python ↔ MATLAB ↔ T-MATS
│   ├── preprocessing/
│   │   └── cmapss_loader.py           # Carga, RUL, health index, normalización
│   ├── models/
│   │   ├── pinn.py                    # BraytonPINN + pérdida física + wrapper
│   │   ├── pinn_ablated.py            # Variantes sin CL y sin física
│   │   └── rnn_health.py           
│   ├── environments/
│   │   └── ace_env.py                 # Entorno Gymnasium ACE-v0
│   ├── agents/
│   │   ├── fadec_baseline.py          # Baseline FADEC con gain scheduling
│   │   └── safe_rl.py                 # SafetyLayer + ActionFilter
│   └── training/
│       ├── train_pinn.py              # CLI del gemelo digital
│       ├── train_pinn_ablation.py     # CLI de las variantes ablacionadas
│       ├── train_rnn.py               # CLI del monitor de salud
│       ├── train_drl.py               # CLI de SAC / TD3
│       └── train_bc_drl.py            # Pipeline TD3 + Behavioral Cloning
│
├── scripts/
│   ├── evaluation/
│   │   ├── eval_pinn.py               # MAPE por variable y latencia
│   │   ├── eval_pde_residuals.py      # Consistencia física fuera de train
│   │   └── eval_rnn_compare.py        # Comparativa LSTM-128 / LSTM-64 / GRU-64
│   └── plotting/
│       ├── plot_pinn_evaluation.py    # fig28 – fig30
│       ├── plot_rnn_comparison.py     # fig15 – fig18
│       ├── plot_rnn_single.py         # fig31 – fig32
│       ├── plot_drl_section5.py       # fig19 – fig23 + RESULTADOS_SECCION_5.json
│       └── plot_section6.py           # fig24 – fig27 + RESULTADOS_SECCION_6.json
│
├── notebooks/                         # Scripts con celdas '# %%' (no requieren Jupyter)
│   ├── corpus_analysis.py             # Análisis del corpus      → fig01 – fig06
│   ├── cmapss_degradation_analysis.py # Análisis C-MAPSS         → fig07 – fig12
│   └── aetp_validation.py             # Validación frente a AETP → fig13 – fig14
│
├── checkpoints/                       # Pesos entrenados (incluidos)
│   ├── digital_twin/                  # pinn.pt, pinn_no_cl.pt, pinn_no_physics.pt
│   │                                  # + training_history_pinn.csv (curvas de fig28)
│   ├── health_monitoring/             # rnn_gru_64.pt, rnn_lstm_64.pt, rnn_lstm_128.pt
│   └── control/                       # sac_mixed/, td3_mixed/, td3_bc/
│                                      #   (best_model.zip + {algo}_final.zip)
│
├── logs/                              # Métricas y eventos de TensorBoard por run
│   ├── sac_mixed/  td3_mixed/  td3_bc/    (cada uno con su results.json)
│
├── results/
│   ├── figures/                       # Las 32 figuras del TFG
│   ├── section_results/               # RESULTADOS_SECCION_5.json y _6.json
│   └── notebooks/                     # Tablas CSV de los análisis
│
├── unit_test/                         # 14 tests unitarios (pytest)
├── requirements.txt
├── pytest.ini
├── LICENSE
└── README.md
```

---

## 6. Instalación

Los pasos 1 a 5 son **obligatorios**. Los pasos 6 y 7 (MATLAB y T-MATS) solo hacen falta si quieres regenerar el corpus sintético desde el simulador.

### 6.1 Instalación en Windows

#### Paso 1 — Instalar Python 3.10

1. Descarga el instalador desde <https://www.python.org/downloads/release/python-31011/> (*Windows installer 64-bit*).
2. Durante la instalación marca **`Add python.exe to PATH`**.
3. Verifica en PowerShell:

```powershell
python --version
# Debe mostrar: Python 3.10.x
```

#### Paso 2 — Instalar Git y clonar el repositorio

```powershell
# Si no tienes Git: https://git-scm.com/download/win
cd $HOME\Desktop
git clone https://github.com/vtnehu121/adaptive-cycle-engine-drl.git
cd adaptive-cycle-engine-drl
```

#### Paso 3 — Crear y activar el entorno virtual

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> Si PowerShell bloquea el script con *"la ejecución de scripts está deshabilitada"*, ejecuta una sola vez:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Con el símbolo del sistema (`cmd.exe`) la activación es `venv\Scripts\activate.bat`.

Sabrás que está activo porque el prompt empieza por `(venv)`.

#### Paso 4 — Actualizar pip e instalar dependencias

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Paso 5 — Comprobar el dispositivo de cómputo

El proyecto está pensado para ejecutarse en **CPU**, que es como se han generado todos los resultados. La build por defecto de PyTorch que instala `requirements.txt` es exactamente la que necesitas: **no hay ningún paso adicional que dar**.

```powershell
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA disponible:', torch.cuda.is_available())"
# Salida esperada en el equipo de referencia: PyTorch 2.12.1 | CUDA disponible: False
```

`CUDA disponible: False` es el resultado correcto y esperado. Los módulos detectan el dispositivo automáticamente y recaen en CPU.

<details>
<summary><b>Opcional</b> — acelerar el entrenamiento si dispones de una GPU NVIDIA</summary>

No es necesario para reproducir el proyecto, pero acorta el entrenamiento del PINN. Requiere una tarjeta NVIDIA con drivers instalados:

```powershell
pip uninstall -y torch
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"   # debe imprimir True
```

No hay que tocar el código: la selección de dispositivo (`device='auto'`) pasa sola a CUDA. Ten en cuenta que las cifras de latencia de §8.2 están medidas en CPU; sobre GPU serán distintas.
</details>

#### Paso 6 — (Solo para regenerar el corpus) MATLAB R2026a + Simulink

1. Instala MATLAB R2026a con el *toolbox* **Simulink** desde <https://www.mathworks.com/downloads/>.
2. Comprueba la licencia de Simulink abriendo MATLAB y ejecutando `license('test','Simulink')` — debe devolver `1`.

#### Paso 7 — (Solo para regenerar el corpus) T-MATS y MATLAB Engine

T-MATS debe quedar **junto al repositorio, en la misma carpeta padre**, con el nombre exacto `T-MATS-master`. Es la ruta que `models/init_ACE.m` espera:

```
Desktop\
├── T-MATS-master\
│   └── Trunk\
└── adaptive-cycle-engine-drl\
```

```powershell
cd $HOME\Desktop
git clone https://github.com/nasa/T-MATS.git T-MATS-master

# Instalar el MATLAB Engine para Python (con el venv activo)
cd adaptive-cycle-engine-drl
pip install matlabengine==26.1.12
```

Verifica el puente:

```powershell
python -c "import matlab.engine; print('MATLAB Engine OK')"
```

#### Paso 8 — Verificar la instalación

```powershell
python -c "import torch, stable_baselines3, gymnasium, sklearn, pandas; print('Dependencias OK')"
python -c "from src.environments.ace_env import ACEEnv; print('Entorno OK')"
pytest -m "not slow"
```

---

### 6.2 Instalación en Linux (Ubuntu 22.04 / 24.04)

#### Paso 1 — Instalar Python 3.10 y herramientas del sistema

En Ubuntu 22.04 Python 3.10 es la versión del sistema. En 24.04 hay que añadir el PPA de deadsnakes:

```bash
sudo apt update
sudo apt install -y git build-essential

# Ubuntu 22.04
sudo apt install -y python3.10 python3.10-venv python3-pip

# Ubuntu 24.04 (Python 3.10 no viene por defecto)
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev

python3.10 --version
# Debe mostrar: Python 3.10.x
```

#### Paso 2 — Clonar el repositorio

```bash
cd ~
git clone https://github.com/vtnehu121/adaptive-cycle-engine-drl.git
cd adaptive-cycle-engine-drl
```

#### Paso 3 — Crear y activar el entorno virtual

```bash
python3.10 -m venv venv
source venv/bin/activate
```

El prompt debe empezar por `(venv)`.

#### Paso 4 — Actualizar pip e instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Paso 5 — Comprobar el dispositivo de cómputo

El proyecto se ejecuta en **CPU**, que es como se han generado todos los resultados. La build por defecto de PyTorch que instala `requirements.txt` es la correcta: **no hay ningún paso adicional que dar**.

```bash
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA disponible:', torch.cuda.is_available())"
# Salida esperada en el equipo de referencia: PyTorch 2.12.1 | CUDA disponible: False
```

`CUDA disponible: False` es el resultado correcto y esperado.

<details>
<summary><b>Opcional</b> — acelerar el entrenamiento si dispones de una GPU NVIDIA</summary>

```bash
nvidia-smi   # si el comando no existe: sudo ubuntu-drivers autoinstall && reiniciar

pip uninstall -y torch
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"   # debe imprimir True
```

No hay que tocar el código: la selección de dispositivo (`device='auto'`) pasa sola a CUDA. Las cifras de latencia de §8.2 están medidas en CPU; sobre GPU serán distintas.
</details>

#### Paso 6 — (Solo para regenerar el corpus) MATLAB R2026a + Simulink

```bash
# Descarga el instalador desde mathworks.com y descomprímelo
unzip matlab_R2026a_glnxa64.zip -d matlab_installer
cd matlab_installer
sudo ./install     # selecciona MATLAB + Simulink

# Añade MATLAB al PATH (ajusta la ruta si instalaste en otro sitio)
echo 'export PATH=$PATH:/usr/local/MATLAB/R2026a/bin' >> ~/.bashrc
source ~/.bashrc
matlab -batch "disp(license('test','Simulink'))"   # debe imprimir 1
```

#### Paso 7 — (Solo para regenerar el corpus) T-MATS y MATLAB Engine

Misma disposición de carpetas que en Windows: `T-MATS-master` como hermana del repositorio.

```bash
cd ~
git clone https://github.com/nasa/T-MATS.git T-MATS-master

cd ~/adaptive-cycle-engine-drl
source venv/bin/activate
pip install matlabengine==26.1.12
python -c "import matlab.engine; print('MATLAB Engine OK')"
```

> Si `pip install matlabengine` falla, instálalo desde la propia distribución de MATLAB:
> ```bash
> cd /usr/local/MATLAB/R2026a/extern/engines/python
> python setup.py install
> ```

#### Paso 8 — Verificar la instalación

```bash
python -c "import torch, stable_baselines3, gymnasium, sklearn, pandas; print('Dependencias OK')"
python -c "from src.environments.ace_env import ACEEnv; print('Entorno OK')"
pytest -m "not slow"
```

---

## 7. Uso: cómo ejecutar el proyecto

> **Regla importante:** todos los comandos deben ejecutarse **desde la raíz del repositorio** (`adaptive-cycle-engine-drl/`). La mayoría de los scripts resuelve checkpoints y corpus mediante rutas relativas al directorio de trabajo (`checkpoints/digital_twin/pinn.pt`, `data/synthetic/…`), así que ejecutarlos desde otra carpeta hará que no encuentren los artefactos. Solo `plot_rnn_comparison.py` resuelve sus rutas contra la raíz del proyecto y funciona desde cualquier directorio; no conviene apoyarse en esa excepción.

### Paso 0 — Activar el entorno virtual

**Windows (PowerShell):**
```powershell
cd $HOME\Desktop\adaptive-cycle-engine-drl
.\venv\Scripts\Activate.ps1
```

**Linux (Ubuntu):**
```bash
cd ~/adaptive-cycle-engine-drl
source venv/bin/activate
```

A partir de aquí **los comandos son idénticos en ambos sistemas**. Solo se indica la diferencia cuando existe.

---

### 7.1 Ruta rápida: reproducir los resultados sin reentrenar nada

Los checkpoints están incluidos en el repositorio. Esta ruta tarda unos minutos y regenera todas las métricas y figuras del TFG.

#### Paso 1 — Evaluar el gemelo digital

```bash
python scripts/evaluation/eval_pinn.py
```
Imprime el MAPE por variable sobre el conjunto de test (splits 80/15/5 con semilla 42) y una medida rápida de latencia. La arquitectura se detecta automáticamente del `state_dict`, así que no hay que declararla.

```bash
python scripts/evaluation/eval_pinn.py --compare-train-test   # detección de overfitting
python scripts/evaluation/eval_pinn.py --subset val           # evaluar sobre validación
python scripts/evaluation/eval_pinn.py --checkpoints \
    checkpoints/digital_twin/pinn.pt \
    checkpoints/digital_twin/pinn_no_cl.pt \
    checkpoints/digital_twin/pinn_no_physics.pt   # los tres del ablation de una vez
python scripts/evaluation/eval_pde_residuals.py               # consistencia física fuera de train
```

> La latencia que imprime `eval_pinn.py` (500 iteraciones **sin *warm-up***) es solo indicativa y sale más alta que la cifra oficial. El perfilado SWaP-C de §8.2 lo hace `plot_section6.py` con 200 iteraciones de calentamiento y 2 000 cronometradas.

> En `eval_pde_residuals.py`, un residuo alto en `pde_thrust` **no** indica un modelo defectuoso: ese residuo integra efectos globales del ciclo y es 1–2 órdenes de magnitud mayor que los locales (masa, energía, isentrópico, combustión), sobre todo en régimen supersónico, donde T-MATS extrapola fuera de calibración.

#### Paso 2 — Evaluar el monitor de salud

```bash
python scripts/evaluation/eval_rnn_compare.py
```
Compara LSTM-128, LSTM-64 y GRU-64 sobre la validación de C-MAPSS FD001 por evaluación manual batch a batch, contrastándola con `HealthMonitor.evaluate()` como comprobación cruzada.

> **Cuál es la comparativa oficial.** `eval_rnn_compare.py` evalúa cada modelo con **su propio** `val_loader`, lo que introduce ligera variabilidad en las coberturas IC 95%. La comparativa reportada en §8.4 es la de `plot_rnn_comparison.py`, que usa un **`val_loader` único compartido** por los tres modelos. La conclusión cualitativa —GRU-64 como el modelo mejor calibrado— es la misma en ambos.

#### Paso 3 — Generar las figuras del gemelo digital (fig28–fig30)

```bash
python scripts/plotting/plot_pinn_evaluation.py
```

#### Paso 4 — Generar las figuras del monitor de salud (fig15–fig18 y fig31–fig32)

```bash
python scripts/plotting/plot_rnn_comparison.py
python scripts/plotting/plot_rnn_single.py --rnn gru --hidden 64
```

#### Paso 5 — Evaluar los controladores y generar las figuras de control (fig19–fig23)

```bash
python scripts/plotting/plot_drl_section5.py
```
Ejecuta 50 episodios deterministas por controlador (SAC, TD3, TD3+BC, FADEC — ese es el orden de columnas de todas las figuras) y escribe `results/section_results/RESULTADOS_SECCION_5.json`, que es **el resultado oficial del TFG** (§8.5). Carga de cada agente el `best_model.zip`, no el `{algo}_final.zip`.

> Este script **no fija semilla** deliberadamente: con una semilla concreta el baseline FADEC caía siempre en las mismas misiones desfavorables y exageraba la ventaja del DRL. Las cifras varían ligeramente entre ejecuciones; el orden entre controladores es estable.

#### Paso 6 — Validación crítica: ablación, OOD, sensibilidad y latencia (fig24–fig27)

```bash
python scripts/plotting/plot_section6.py
```
Escribe `results/section_results/RESULTADOS_SECCION_6.json`. Cada uno de los cuatro análisis se omite con un aviso si le falta algún checkpoint, sin abortar el resto.

#### Paso 7 — Análisis del corpus (fig01–fig14)

```bash
python notebooks/corpus_analysis.py
python notebooks/cmapss_degradation_analysis.py
python notebooks/aetp_validation.py
```
Son scripts Python normales con celdas `# %%`; también pueden abrirse como cuadernos en VS Code o Jupyter.

---

### 7.2 Ruta completa: reentrenar todo desde cero

#### Paso 1 — (Opcional) Regenerar el corpus sintético con T-MATS

Requiere MATLAB, Simulink y T-MATS instalados según los pasos 6 y 7 de la instalación. **Es la etapa más costosa** — miles de simulaciones de Simulink encadenadas, varias horas.

```bash
python src/pipeline.py                    # pipeline completo, 5 000 muestras
python src/pipeline.py --n-samples 100    # prueba rápida → ace_dataset_100.csv
python src/pipeline.py --seed 7           # otra semilla (se propaga a envelope y degradación)
python src/pipeline.py --skip-matlab      # reutiliza el corpus existente
```

Las cinco etapas son: muestreo del envelope → simulación T-MATS → preprocesado de C-MAPSS → inyección de degradación → validación físico-estadística (9 comprobaciones). El script termina con código de salida 1 si alguna de las comprobaciones físicas falla, de forma que pueda encadenarse en integración continua.

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--n-samples` | 5000 | Muestras a generar; también nombra los CSV de salida |
| `--seed` | 42 | Semilla única, propagada al muestreo del envelope y a la inyección de degradación |
| `--skip-matlab` | — | Reutiliza `ace_dataset_{n}.csv` sin invocar MATLAB |

Antes de la primera ejecución, prepara el entorno de MATLAB:

```matlab
% En MATLAB, desde la carpeta models/ del repositorio
run('init_ACE.m')          % añade T-MATS al path y carga los mapas del JT9D
open('ACE_3stream_brayton.slx')
run('setup_ACE_params.m')  % define la estructura ACE.* de parámetros
```

> `modify_JT9D_to_ACE.m` **no forma parte de este flujo**: es el script de construcción única que insertó los bloques del tercer flujo (splitter, duct, tobera variable y sangrado del HPC) sobre una copia del JT9D base. El modelo ya construido está versionado como `ACE_3stream_brayton.slx`, así que solo hace falta volver a ejecutarlo si se quiere reconstruir el modelo desde cero — y exige tener el `.slx` abierto y conectar después las líneas de señal manualmente en Simulink.

#### Paso 2 — Entrenar el gemelo digital PINN

```bash
python src/training/train_pinn.py --epochs 1200 --patience 200
```

Los valores por defecto de arquitectura y learning rate **ya son los del checkpoint activo** (224 × 8, `lr` 1e-3); solo hay que subir el presupuesto de épocas y la paciencia. Mejor época del checkpoint: 1145.

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--epochs` | 600 | Máximo de épocas |
| `--patience` | 80 | Paciencia del early stopping |
| `--hidden` | 224 | Dimensión oculta de los bloques residuales (reproduce el checkpoint activo) |
| `--layers` | 8 | Número de bloques residuales (reproduce el checkpoint activo) |
| `--lr` | 1e-3 | Learning rate inicial de AdamW (reproduce el checkpoint activo) |
| `--dataset` | `data/synthetic/ace_dataset_5000.csv` | Ruta al corpus |
| `--checkpoint` | `checkpoints/digital_twin/pinn.pt` | Ruta de salida |

> El CLI **no expone la semilla**: `ACEDigitalTwin.__init__` fija `torch.manual_seed(42)` y `np.random.seed(42)` antes de instanciar la red, para que el modelo completo y las dos variantes ablacionadas partan de la misma inicialización y la comparación sea justa. Cambiarla rompería esa equivalencia.

#### Paso 3 — Entrenar las variantes ablacionadas del PINN

```bash
python src/training/train_pinn_ablation.py --epochs 1200 --patience 200
python src/training/train_pinn_ablation.py --only no_cl        # solo sin constraint layers
python src/training/train_pinn_ablation.py --only no_physics   # solo sin pérdida física
```

Mismos valores por defecto de arquitectura que el modelo completo (224 × 8, `lr` 1e-3). Genera `pinn_no_cl.pt` y `pinn_no_physics.pt` en `checkpoints/digital_twin/`.

#### Paso 4 — Entrenar el monitor de salud

```bash
python src/training/train_rnn.py --rnn gru  --hidden 64  --epochs 100   # modelo elegido
python src/training/train_rnn.py --rnn lstm --hidden 64  --epochs 100
python src/training/train_rnn.py --rnn lstm --hidden 128 --epochs 100
```

Argumentos: `--rnn {lstm,gru}` (def. `lstm`), `--hidden` (def. 128), `--layers` (def. 2), `--seq-len` (def. 50), `--subset {FD001…FD004}` (def. FD001), `--epochs` (def. 100), `--patience` (def. 20), `--alpha` (def. 0,3), `--checkpoint`.

Si no se indica `--checkpoint`, el nombre se construye solo a partir de la arquitectura: `checkpoints/health_monitoring/rnn_{lstm|gru}_{hidden}.pt`, que es exactamente lo que buscan los scripts de figuras.

#### Paso 5 — Entrenar los agentes DRL

```bash
python src/training/train_drl.py --algo sac --timesteps 500000 --mission mixed
python src/training/train_drl.py --algo td3 --timesteps 500000 --mission mixed
python src/training/train_drl.py --algo both --timesteps 500000   # ambos + comparativa FADEC
```

Argumentos: `--algo {sac,td3,both}` (def. `sac`), `--timesteps` (def. **500 000**), `--mission {mixed,cruise,combat,full}` (def. `mixed`), `--seed` (def. 42).

> **Sobre el presupuesto de pasos.** 500 000 es el valor con el que se entrenaron los checkpoints activos (registrado en `logs/{sac,td3}_mixed/results.json`) y es el valor por defecto del CLI: `python src/training/train_drl.py --algo sac` los reproduce sin pasar `--timesteps`. Los comandos de arriba lo indican de forma explícita solo por claridad. Lo mismo aplica a `train_bc_drl.py`, cuyo `--rl-timesteps` ya venía por defecto a 500 000.

Los checkpoints se escriben en `checkpoints/control/{algo}_{mission}/` y los logs en `logs/{algo}_{mission}/`, **sobrescribiendo el run anterior con la misma combinación** — es una convención deliberada para que los scripts de figuras encuentren siempre el modelo activo sin editar rutas.

#### Paso 6 — Entrenar el pipeline híbrido TD3+BC (mejor política)

```bash
python src/training/train_bc_drl.py --expert-episodes 200 --bc-epochs 100 --rl-timesteps 500000
```

Esos son además los valores por defecto. Ejecuta las tres fases en orden: generación de 40 000 demostraciones con el FADEC (200 ep × 200 pasos) → behavioral cloning del actor por MSE → fine-tuning por refuerzo con el replay buffer pre-poblado. Otros argumentos: `--bc-batch-size` (def. 256), `--bc-lr` (def. 1e-3), `--seed` (def. 42).

#### Paso 7 — Regenerar métricas y figuras

Repite los pasos 1 a 7 de la [ruta rápida](#71-ruta-rápida-reproducir-los-resultados-sin-reentrenar-nada).

---

### 7.3 Monitorizar el entrenamiento con TensorBoard

```bash
tensorboard --logdir logs/
```
Abre <http://localhost:6006> en el navegador.

### 7.4 Ejecutar los tests

```bash
pytest                    # toda la suite: 14 tests
pytest -m "not slow"      # omite los 2 tests que levantan MATLAB
pytest --cov=src          # con informe de cobertura
pytest unit_test/test_flight_envelope.py -v
```

La suite son 14 tests repartidos en tres archivos: `test_cmapss_loader.py` (5), `test_flight_envelope.py` (4) y `test_tmats_interface.py` (5). Estos últimos se saltan automáticamente si el MATLAB Engine no está instalado, y dos de ellos —marcados como `slow`, unos dos minutos entre ambos— son tests de regresión de fallos silenciosos corregidos durante la auditoría del modelo Simulink: el desacoplamiento del splitter del tercer flujo respecto a `bpr_ts`, y la activación del *customer bleed* del HPC. Antes de esas correcciones, dos de las cinco variables de control no tenían ningún efecto sobre el ciclo termodinámico y las simulaciones terminaban sin error.

### 7.5 Resolución de problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Estás ejecutando desde otra carpeta. Sitúate en la raíz del repositorio. |
| `PINN no encontrado en checkpoints/digital_twin/pinn.pt` | Misma causa: ruta relativa a la raíz. El entorno degrada a su modelo simplificado interno y avisa por log. |
| Caracteres corruptos en la consola de Windows | Los scripts de plotting ya fuerzan UTF-8. Para el resto: `chcp 65001` antes de ejecutar. |
| `matlab.engine` no se importa | El MATLAB Engine no está instalado, o su versión no corresponde con la release de MATLAB. Solo afecta a la regeneración del corpus. |
| `T-MATS no encontrado en …/T-MATS-master/Trunk` | T-MATS debe estar en la carpeta **hermana** del repositorio y llamarse exactamente `T-MATS-master`. |
| `torch.cuda.is_available()` devuelve `False` | **Es el comportamiento esperado.** El proyecto se ejecuta en CPU y todos los resultados se han generado así; no hay que hacer nada. Solo si tienes una GPU NVIDIA y quieres acelerar el entrenamiento, reinstala torch desde el índice `cu121` (paso 5 de la instalación). |
| El entrenamiento va lento | Es normal en CPU, sobre todo el PINN a 1 200 épocas. Usa los checkpoints incluidos y la [ruta rápida](#71-ruta-rápida-reproducir-los-resultados-sin-reentrenar-nada), o reduce `--epochs` para una prueba. La inferencia, que es lo que importa para el control, es de milisegundos. |

---

## 8. Resultados principales

Todos los datos de esta sección proceden de `results/`: los JSON de `results/section_results/`, las tablas de `results/notebooks/` y las 32 figuras de `results/figures/`.

### 8.1 Corpus sintético

| Métrica | Valor |
|---|---|
| Muestras | 5 000 |
| Variables termodinámicas | 27 |
| Convergencia del solver | 100% |
| Regímenes cubiertos | 5 (takeoff, climb, cruise, combat, descent) |
| Rango de altitud | 2 – 41 984 ft |
| Rango de Mach | 0,10 – 1,80 |
| Rango de empuje | 2 275 – 217 771 lbf (media 29 860) |
| Rango de SFC | 0,198 – 2,11 lbm/lbf·h (media 0,682) |
| Rango de T₄ | 1 686 – 3 727 °R (media 2 948) |
| Semilla | 42 |

**Comportamiento adaptativo verificado** (`results/notebooks/regime_statistics.csv` y `aetp_qualitative_validation.csv`): el corpus reproduce la reconfiguración característica de un ciclo adaptativo, el ratio de empuje combate/crucero es de **5,85×**, el diferencial de SFC entre ambos regímenes es del **−41,6%** con una amplitud operativa de Nf del **50,7%**, y el BPR total varía entre 0,04 y 0,25 en fuerte coherencia con `bpr_ts` (correlación r = 0,982), confirmando el control activo del tercer flujo.

> **Matiz sobre el 5,85×.** Ese ratio queda algo por encima del valor típico militar (~3–4×) porque los edge cases supersónicos a baja altitud (Mach > 1,7) llevan a T-MATS fuera de su ventana de calibración e inflan el empuje medio de combate. La reconfiguración cualitativa es válida; la magnitud absoluta no es directamente comparable con AETP.

### 8.2 Gemelo digital PINN

Arquitectura 224×8, 838 443 parámetros, 3,35 MB. Entrenado 1 200 épocas con paciencia 200; mejor época 1145.

**MAPE por variable sobre el conjunto de test — MAPE medio: 4,64%**

| Variable | MAPE (%) | Variable | MAPE (%) |
|---|---|---|---|
| T₄ | **0,50** | Nf | 0,98 |
| T₅₀ | 0,58 | thrust | **2,36** |
| T₂ | 0,59 | W_inlet | 2,59 |
| P₂ | 0,63 | P₃₀ | 4,43 |
| Nc | 0,75 | wf | 7,86 |
| T₃₀ | 0,97 | SFC | 8,34 |
| farB | ~0,00 | BPR | 34,36 |

Cifras tomadas de `results/section_results/RESULTADOS_SECCION_6.json` (bloque `ablation → PINN_completo`). `farB`, `P2` y `SFC` se calculan por *constraint layers*: el error residual de las dos primeras es el del redondeo en punto flotante. El BPR es la variable con mayor error del conjunto. Ejecutar `eval_pinn.py` sobre el split de test (250 muestras) devuelve el mismo promedio de **4,64%** con diferencias de décimas por variable, propias del subconjunto evaluado.

**Latencia de inferencia (perfilado SWaP-C, `fig27`) — medida en CPU, sin aceleración por GPU**

Medida con 200 iteraciones de calentamiento y 2 000 cronometradas, sobre consultas **individuales** (batch = 1), que es como el entorno de control interroga al gemelo digital en cada paso:

| Métrica | Valor | Umbral | ¿Cumple? |
|---|---|---|---|
| Media | 1,206 ms | — | — |
| P50 | 1,231 ms | — | — |
| P95 | 1,556 ms | — | — |
| **P99** | **1,840 ms** | 5 ms (DO-178C nivel A) | **Sí** |
| Máximo | 2,956 ms | — | — |
| Memoria | 3,35 MB | < 10 MB | Sí |

El JSON registra además el escalado por lote: con batch 128 la latencia total sube a 2,72 ms de media, pero el coste **por muestra** cae a 0,021 ms, un factor 57× frente a la consulta individual.

Tres matices importantes sobre esta medida:

- **El criterio de aceptación es el P99, no la media.** Es la cola de la distribución la que compromete el determinismo temporal de un lazo de control: una media excelente con una cola larga sigue siendo inaceptable en tiempo real.
- **Está medida íntegramente en CPU** (Intel Core i7-12650H), sin GPU. El gemelo digital cumple el presupuesto temporal con un margen de 2,7× **sin hardware acelerador**, que es la condición realista de un sistema embarcado: un módulo de control de motor no lleva una GPU de escritorio a bordo. Sustituir T-MATS por el PINN convierte una consulta de segundos en una de menos de 2 ms sobre el mismo procesador.
- **Es la única medida de latencia que debe citarse.** Otros scripts (`eval_pinn.py`, `train_pinn.py`) imprimen cifras de latencia propias, más altas, porque miden con menos iteraciones y sin fase de calentamiento; son diagnósticos rápidos, no el perfilado SWaP-C.

### 8.3 Estudio de ablación del PINN (`fig24`)

| Configuración | MAPE medio | Parámetros | Δ vs. completo |
|---|---|---|---|
| **PINN completo** | **4,64%** | 838 443 | — |
| Sin constraint layers | 7,31% | 838 782 | **+57%** |
| Sin PDEs físicas | 4,62% | 838 443 | −0,4% |

Los tres MAPE se recalculan en cada ejecución de `plot_section6.py` y quedan desglosados por variable en `results/section_results/RESULTADOS_SECCION_6.json`. El efecto de retirar las *constraint layers* se concentra donde cabía esperar: `farB` pasa de ~0% a **19,3%** y `SFC` de 8,34% a **21,45%**, mientras el resto de variables apenas se mueve.

#### Pruebas fuera de la envolvente (`fig26`)

`plot_section6.py` consulta el gemelo digital en tres condiciones nominales y **seis deliberadamente fuera del envelope de entrenamiento** (0 – 42 000 ft, Mach 0,1 – 1,8, TRA 30 – 100%, bpr_ts 0,05 – 0,50), y comprueba que se mantienen las desigualdades básicas del ciclo y el signo de las magnitudes:

| Caso | Condición (alt / Mach / TRA / bpr_ts) | Empuje (lbf) | T₄ (°R) |
|---|---|---|---|
| Crucero nominal | 35 000 / 0,85 / 80 / 0,30 | 9 274 | 2 747 |
| Combate nominal | 15 000 / 1,50 / 100 / 0,10 | 99 244 | 3 564 |
| Takeoff nominal | 0 / 0,30 / 100 / 0,20 | 11 105 | 3 165 |
| **Altitud 55 kft** | 55 000 / 0,85 / 80 / 0,30 | 5 395 | 2 657 |
| **Mach 2,2** | 30 000 / 2,20 / 80 / 0,30 | 36 566 | 2 763 |
| **TRA 10%** | 35 000 / 0,85 / 10 / 0,30 | 14 682 | 2 488 |
| **TRA 110%** | 35 000 / 0,85 / 110 / 0,30 | 10 329 | 2 896 |
| **Combo extremo** | 50 000 / 2,00 / 100 / 0,60 | 17 684 | 2 805 |
| **Nivel del mar, subsónico** | 0 / 0,05 / 50 / 0,50 | 16 543 | 2 774 |

El modelo mantiene el signo y el orden de magnitud de todas las variables en las nueve condiciones, y respeta las desigualdades del ciclo. Fuera del rango de TRA entrenado la respuesta deja de ser monótona: a TRA = 10% predice más empuje (14 682 lbf) que en el crucero nominal a TRA = 80% (9 274 lbf), y a TRA = 110% menos que a TRA = 10%. Es el comportamiento de extrapolación característico que estas pruebas están diseñadas para exponer.

#### Hallazgo clave: las PDEs son un regularizador *out-of-distribution*

La ablación *in-distribution* sugiere que la pérdida física no aporta nada — 4,62% sin PDEs frente a 4,64% con ellas. La comparación del empuje predicho por el modelo completo y por el modelo sin PDEs en las mismas condiciones demuestra lo contrario:

| Régimen | Divergencia media en empuje (completo vs. sin PDEs) |
|---|---|
| Dentro de la envolvente | **1,1%** |
| Fuera de la envolvente | **23,5%** (máximo 53,5%) |

Casos extremos: en ralentí a TRA = 10% (fuera del rango de entrenamiento, 30–100%) la divergencia era del **41,9%**; a 50 000 ft y Mach 2,0, del **53,5%**.

> **Conclusión.** Con un corpus denso de 5 000 muestras el gradiente de datos domina dentro de la envolvente y el término físico es indistinguible. Fuera de ella, las PDEs son lo único que mantiene coherente la extrapolación — coherente además con que el dominio de los puntos de colocación sea deliberadamente más ancho que el del corpus (§2.1). Un ablation medido solo *in-distribution* habría llevado a descartar erróneamente el componente físico.

### 8.4 Health Monitoring (`fig15` – `fig18`, `fig31` – `fig32`)

Protocolo: 80 motores de entrenamiento y 20 de validación de C-MAPSS FD001 (3 711 secuencias), sin fuga de datos entre motores. Pérdida `L = L_RUL + 0,3·L_deg`, MC-Dropout con N = 50.

| Modelo | Parámetros | RUL RMSE | RUL MAE | Deg. RMSE | Cobertura IC 95% |
|---|---|---|---|---|---|
| LSTM-128 | 299 909 | 15,15 | 11,39 | 0,4395 | 69,0% |
| LSTM-64 | 76 229 | 14,47 | 11,00 | 0,4450 | 85,5% |
| **GRU-64 (elegido)** | **59 589** | **13,94** | **10,98** | 0,4398 | **88,4%** |

**GRU-64 supera a LSTM-128 con 5× menos parámetros** y ofrece la calibración del intervalo de confianza más próxima al 95% nominal (88,4%). LSTM-64 alcanza 85,5%, dentro del rango aceptable pero más alejada del nominal; LSTM-128 queda muy por debajo (69,0%), lo que indica exceso de confianza propio de un modelo sobreparametrizado para el problema (los intervalos son demasiado estrechos y no cubren la incertidumbre real).

Estas cifras son las de `plot_rnn_comparison.py`, que evalúa los tres modelos sobre un **`val_loader` único compartido**; es la comparativa oficial. `eval_rnn_compare.py` da coberturas ligeramente distintas por usar un cargador por modelo (§7.1, paso 2).

### 8.5 Control: DRL frente a FADEC (`fig19` – `fig23`)

Evaluación sobre 50 episodios por controlador, perfil de misión `mixed`, política determinista (`results/section_results/RESULTADOS_SECCION_5.json`):

| Controlador | Reward medio | σ | Episodios con violación | Tasa de episodios limpios |
|---|---|---|---|---|
| **TD3 + BC** | **249,94** | **18,95** | **8 / 50** | **84%** |
| SAC | 209,59 | 46,23 | 20 / 50 | 60% |
| TD3 | 205,41 | 40,50 | 22 / 50 | 56% |
| FADEC (baseline) | −1 947,28 | 1 533,57 | 44 / 50 | 12% |

> **Nota metodológica: qué JSON es el oficial.** Existen dos evaluaciones del mismo entrenamiento y **no son intercambiables**:
>
> | | `results/section_results/RESULTADOS_SECCION_5.json` | `logs/{algo}_{mission}/results.json` |
> |---|---|---|
> | Quién lo genera | `plot_drl_section5.py`, a posteriori | El propio `train_drl.py`, al terminar `learn()` |
> | Checkpoint evaluado | **`best_model.zip`** (mejor reward durante el entrenamiento) | `{algo}_final.zip` (modelo al agotar los pasos) |
> | Entorno | `ACEEnv` instanciado con condiciones de misión controladas | El `eval_env` vectorizado del entrenamiento |
> | Definición de seguridad | `violations` = episodios con ≥ 1 paso inseguro; `safety_rate` = fracción de episodios completamente limpios | Violaciones **por paso** sobre el total ejecutado (SAC 98,82%; TD3 79,8%; TD3+BC 100%) |
>
> **Los hallazgos oficiales del TFG son los de `RESULTADOS_SECCION_5.json`** — los de la tabla de arriba, y los que se citan en la memoria. Los `logs/*/results.json` quedan como registro secundario de la evaluación *end-of-training* de Stable-Baselines3; por eso sus cifras no coinciden (p. ej. SAC: 190,56 ± 49,94 allí frente a 209,59 ± 46,23 aquí). Al citar una tasa de seguridad conviene indicar siempre cuál de las dos definiciones se está usando.

**Consumo específico por fase (lbm/lbf·h)**

| Fase | SAC | TD3 | TD3+BC | FADEC |
|---|---|---|---|---|
| Takeoff | 1,360 | 1,518 | **0,795** | 0,942 |
| Cruise | 0,539 | 0,551 | **0,519** | 0,640 |
| Combat | 0,489 | 0,480 | 0,483 | 0,231* |

\* El FADEC obtiene el mejor SFC nominal en combate, pero lo hace operando de forma insegura — véase abajo.

#### Hallazgo clave: el compromiso empuje–temperatura en combate

Análisis específico sobre 20 episodios en régimen de combate (4 000 pasos):

| Métrica | FADEC | TD3 + BC |
|---|---|---|
| Empuje medio | 125 364 lbf | 72 496 lbf |
| T₄ media | 3 620,8 °R | **3 113,7 °R** |
| Margen de bombeo del fan | 6,54% | **16,48%** |
| Postcombustión media | 0,90 | 0,50 |
| Violaciones | **3 980 / 4 000** | **0 / 4 000** |
| Reward medio por episodio | −9 967,33 | **241,02** |

El FADEC maximiza empuje ignorando el límite de turbina (3 200 °R): opera 420 °R por encima del máximo admisible durante el 99,5% del tiempo. TD3+BC renuncia al 42% del empuje y opera dentro de la envolvente, con más del doble de margen de bombeo. El dato es coherente con el corpus: las 637 muestras de régimen de combate tienen una T₄ media de 3 552 °R (rango 3 340 – 3 727), es decir, un combate con postcombustión máxima está físicamente por encima del límite y evitarlo exige renunciar a empuje.

#### Hallazgo clave: la fidelidad del modelo determina la conclusión

El mismo FADEC, evaluado contra tres modelos del motor:

| Modelo del motor | Reward medio del FADEC | Seguridad |
|---|---|---|
| Modelo simplificado (analítico) | **+288,34** | 100% |
| PINN histórico 128×6 | −2 548,99 | 62,1% |
| PINN definitivo 224×8 | −2 364,38 | 74,5% |

Contra un modelo simplificado, el FADEC parece un controlador excelente. La conclusión se invierte por completo al evaluarlo contra un gemelo digital fiel. **Este resultado justifica por sí solo la existencia del módulo PINN**: sin él, todo el estudio comparativo habría concluido lo contrario. El modelo simplificado es el fallback analítico interno de `ACEEnv`, el que se usa al instanciar el entorno con `pinn_model=None`.

### 8.6 Análisis de sensibilidad de la recompensa (`fig25`)

El agente **TD3 tabula rasa** (`checkpoints/control/td3_mixed/best_model.zip`) evaluado bajo cinco ponderaciones distintas de la función de recompensa —10 episodios de 200 pasos por configuración, **con semilla fijada por episodio** para que todas se enfrenten a las mismas misiones—, para comprobar que la política no depende de la calibración heurística de los coeficientes:

| Configuración | Reward | σ | Empuje (lbf) | SFC | Seguridad |
|---|---|---|---|---|---|
| SFC-heavy | **224,85** | 43,40 | 20 825 | 0,875 | 99,8% |
| Balanced | 215,74 | 45,68 | 20 468 | 0,921 | 99,8% |
| Nominal | 197,77 | 35,45 | 20 331 | 0,839 | 99,6% |
| Thrust-heavy | 183,74 | 42,00 | 20 937 | 0,924 | 99,6% |
| Safety-heavy | 145,81 | 46,02 | 21 192 | 0,933 | 99,6% |

La tasa de seguridad —aquí medida **por paso**, no por episodio— se mantiene **por encima del 99,5% en las cinco configuraciones**: el comportamiento seguro aprendido es robusto y no un artefacto de una ponderación concreta. La política no se reentrena en ningún caso; lo que se mide es la robustez del comportamiento ya aprendido, no su capacidad de readaptarse.

### 8.7 Índice de figuras

| Figuras | Contenido | Generadas por |
|---|---|---|
| `fig01` – `fig06` | Distribuciones, rendimiento por régimen, impacto del tercer flujo, correlaciones, casos extremos, sensibilidad | `notebooks/corpus_analysis.py` |
| `fig07` – `fig12` | RUL por sub-dataset, trayectorias de degradación, health index, inyección estocástica, C-MAPSS vs. ACE, correlación degradación–HI | `notebooks/cmapss_degradation_analysis.py` |
| `fig13` – `fig14` | Tendencias adaptativas, envolvente de vuelo | `notebooks/aetp_validation.py` |
| `fig15` – `fig18` | Comparativa de métricas RNN, dispersión de RUL, seguimiento de degradación, calibración de incertidumbre | `scripts/plotting/plot_rnn_comparison.py` |
| `fig19` – `fig23` | Curvas de convergencia, DRL vs. FADEC, perfil de misión, reward por fase, trayectorias de actuadores | `scripts/plotting/plot_drl_section5.py` |
| `fig24` – `fig27` | Ablación, sensibilidad, pruebas OOD, perfilado de latencia SWaP-C | `scripts/plotting/plot_section6.py` |
| `fig28` – `fig30` | Curvas de entrenamiento del PINN, fidelidad frente a T-MATS, cumplimiento de la relación isentrópica | `scripts/plotting/plot_pinn_evaluation.py` |
| `fig31` – `fig32` | MC-Dropout sobre un motor concreto, seguimiento de degradación por componente | `scripts/plotting/plot_rnn_single.py` |

### 8.8 Reproducibilidad

- Semilla global **42** en muestreo, particiones, inicialización de redes y agentes.
- Particiones deterministas por permutación con semilla fija (`np.random.RandomState(42)`).
- `ACEDigitalTwin` y `HealthMonitor` fijan `torch.manual_seed(42)` y `np.random.seed(42)` **antes** de instanciar el modelo, para que la inicialización de pesos sea reproducible y las variantes comparadas partan de la misma semilla base.
- `ACEEnv` sortea la secuencia de fases y el *health index* inicial con `self.np_random` (el generador de Gymnasium), no con `np.random` global: fijar `env.reset(seed=42)` basta para reproducir un episodio.
- Hash SHA-256 del checkpoint activo del PINN, verificado sobre `checkpoints/digital_twin/pinn.pt`: `1CF7C686F5D0DAFBDF471FCE34563878223E8D204AE9BDFC07FE156DF30ACF23`.
- Reproducibilidad de la RNN confirmada como exacta entre ejecuciones.
- Versiones fijadas con `==` en `requirements.txt` para las dependencias que determinan resultados numéricos.
- **Excepciones documentadas:**
  - `plot_drl_section5.py` no fija semilla a propósito (véase §7.1, paso 5), de modo que sus cifras varían ligeramente entre ejecuciones.
  - `FADECBaseline` acepta un argumento `seed` para su ruido de actuación, pero los scripts lo instancian sin él (`FADECBaseline()`), así que ese ruido **no** es reproducible entre ejecuciones. Pasar `seed=42` fijaría también el baseline.
  - El entrenamiento sigue siendo sensible a la variabilidad de PyTorch en GPU; la reproducibilidad exacta se garantiza en CPU, que es como se generaron todos los resultados.

---

## 9. Referencias académicas

### Physics-Informed Neural Networks

- **Raissi, M., Perdikaris, P., Karniadakis, G. E.** (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. **Journal of Computational Physics**, 378, 686–707. — Formulación base de las PINN y de la pérdida por residuo PDE sobre puntos de colocación.

- **Wang, S., Teng, Y., Perdikaris, P.** (2021). *Understanding and mitigating gradient flow pathologies in physics-informed neural networks*. **SIAM Journal on Scientific Computing**, 43(5), A3055–A3081. — Origen del currículo progresivo de los pesos ω₂ y ω₃ implementado en `PINNLoss.update_weights()`.

- **Cuomo, S., Di Cola, V. S., Giampaolo, F., Rozza, G., Raissi, M., Piccialli, F.** (2022). *Scientific Machine Learning through Physics-Informed Neural Networks: Where we are and What's next*. arXiv:2201.05624. — Revisión del estado del arte y justificación del ajuste empírico de los pesos físicos.

### Deep Reinforcement Learning

- **Haarnoja, T., Zhou, A., Abbeel, P., Levine, S.** (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. **ICML 2018**. — Algoritmo SAC.

- **Fujimoto, S., van Hoof, H., Meger, D.** (2018). *Addressing Function Approximation Error in Actor-Critic Methods*. **ICML 2018**. — Algoritmo TD3.

- **Fujimoto, S., Gu, S. S.** (2021). *A Minimalist Approach to Offline Reinforcement Learning*. **NeurIPS 2021**. — Base del esquema TD3+BC de inicialización por clonación de comportamiento.

- **Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., Dormann, N.** (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. **JMLR**, 22(268), 1–8. — Implementación de referencia de SAC y TD3.

### Safe Reinforcement Learning

- **Ames, A. D., Xu, X., Grizzle, J. W., Tabuada, P.** (2017). *Control Barrier Function Based Quadratic Programs for Safety Critical Systems*. **IEEE Transactions on Automatic Control**, 62(8), 3861–3876. — Fundamento teórico de la capa de seguridad.

- **Brunke, L., Greeff, M., Hall, A. W., Yuan, Z., Zhou, S., Panerati, J., Schoellig, A. P.** (2022). *Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning*. **Annual Review of Control, Robotics, and Autonomous Systems**, 5, 411–444. — Taxonomía de métodos de RL seguro y formulación como CMDP.

- **Altman, E.** (1999). *Constrained Markov Decision Processes*. Chapman & Hall/CRC. — Formalización del CMDP sobre el que se plantea el problema de control.

### Pronóstico y monitorización de salud

- **Saxena, A., Goebel, K., Simon, D., Eklund, N.** (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. **International Conference on Prognostics and Health Management (PHM)**, 1–9. DOI: 10.1109/PHM.2008.4711414. — Dataset C-MAPSS, Table 3 de desgaste natural y modelo exponencial `h(t) = 1 − exp(−a·t^b)` implementados en `degradation.py`.

- **Gal, Y., Ghahramani, Z.** (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. **ICML 2016**. — MC-Dropout para la incertidumbre epistémica del RUL.

- **Heimes, F. O.** (2008). *Recurrent Neural Networks for Remaining Useful Life Estimation*. **PHM 2008**. — Origen de la práctica de saturar la RUL (*clipping*) en 125 ciclos.

- **Zheng, S., Ristovski, K., Farahat, A., Gupta, C.** (2017). *Long Short-Term Memory Network for Remaining Useful Life Estimation*. **IEEE ICPHM 2017**. — Arquitectura recurrente de referencia sobre C-MAPSS.

- **Hochreiter, S., Schmidhuber, J.** (1997). *Long Short-Term Memory*. **Neural Computation**, 9(8), 1735–1780.

- **Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., Bengio, Y.** (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*. **EMNLP 2014**. — Arquitectura GRU, la finalmente seleccionada.

### Arquitecturas de red neuronal

- **He, K., Zhang, X., Ren, S., Sun, J.** (2016). *Deep Residual Learning for Image Recognition*. **IEEE Conference on Computer Vision and Pattern Recognition (CVPR)**, 770–778. — Bloques residuales que constituyen el bloque básico de la arquitectura del PINN.

- **Hendrycks, D., Gimpel, K.** (2016). *Gaussian Error Linear Units (GELUs)*. arXiv:1606.08415. — Función de activación GELU utilizada en los bloques residuales del PINN.

### Propulsión, simulación y control de motores

- **Chapman, J. W., Lavelle, T. M., May, R. D., Litt, J. S., Guo, T.-H.** (2014). *Toolbox for the Modeling and Analysis of Thermodynamic Systems (T-MATS) User's Guide*. **NASA/TM-2014-216638**. — Simulador termodinámico y mapas del JT9D empleados como base del modelo ACE.

- **Jaw, L. C., Mattingly, J. D.** (2009). *Aircraft Engine Controls: Design, System Analysis, and Health Monitoring*. AIAA Education Series. — Capítulos 5–6: *gain scheduling* del FADEC implementado como baseline.

- **Mattingly, J. D., Heiser, W. H., Pratt, D. T.** (2002). *Aircraft Engine Design* (2ª ed.). AIAA Education Series. — Ciclo Brayton y relaciones termodinámicas de los residuos PDE.

- **Walsh, P. P., Fletcher, P.** (2004). *Gas Turbine Performance* (2ª ed.). Blackwell Science. — Rendimiento off-design y márgenes de bombeo.

- **GE Aerospace** (2022). *XA100 Adaptive Cycle Engine: Program Results*. — Objetivos públicos del programa AETP usados en la validación cualitativa del corpus.

- **RTCA** (2011). *DO-178C: Software Considerations in Airborne Systems and Equipment Certification*. — Niveles de criticidad usados como umbrales del perfilado de latencia.

- **ICAO** (1993). *Manual of the ICAO Standard Atmosphere*, Doc 7488/3. — Modelo atmosférico ISA de `tmats_interface.py` y de la *constraint layer* de P₂.

### Muestreo estadístico

- **McKay, M. D., Beckman, R. J., Conover, W. J.** (1979). *A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code*. **Technometrics**, 21(2), 239–245. — Muestreo estratificado (LHS) del envelope de vuelo por hiperrectángulos de misión.

### Técnicas implementadas

| Técnica | Módulo | Referencia |
|---|---|---|
| PINN con residuos PDE y colocación | `src/models/pinn.py` | Raissi et al. (2019) |
| Currículo progresivo de pesos físicos | `src/models/pinn.py` | Wang et al. (2021) |
| Bloques residuales + LayerNorm + GELU | `src/models/pinn.py` | He et al. (2016); Hendrycks & Gimpel (2016) |
| Soft Actor-Critic | `src/training/train_drl.py` | Haarnoja et al. (2018) |
| Twin Delayed DDPG | `src/training/train_drl.py` | Fujimoto et al. (2018) |
| Behavioral Cloning + fine-tuning | `src/training/train_bc_drl.py` | Fujimoto & Gu (2021) |
| Barreras logarítmicas de seguridad | `src/agents/safe_rl.py` | Ames et al. (2017) |
| MC-Dropout para incertidumbre | `src/models/rnn_health.py` | Gal & Ghahramani (2016) |
| Modelo de degradación exponencial | `src/data_gen/degradation.py` | Saxena et al. (2008) |
| Muestreo estratificado por fase (uniforme dentro de cada hiperrectángulo) | `src/data_gen/flight_envelope.py` | McKay et al. (1979) |

---

## 10. Autora y contacto

**Beatriz Nevado Huertas**
Grado en Computación e Inteligencia Artificial
Universidad Alfonso X el Sabio (UAX), Madrid

- Correo académico: **bnevahue@myuax.com**
- Correo personal: **beatriznevado63@gmail.com**
- GitHub: [@vtnehu121](https://github.com/vtnehu121)

### Cita

```bibtex
@thesis{nevado2026ace,
  author  = {Nevado Huertas, Beatriz},
  title   = {Control Multiobjetivo de Motores de Ciclo Adaptativo de Tercer
             Flujo mediante Arquitectura Híbrida PINN--DRL},
  school  = {Universidad Alfonso X el Sabio},
  year    = {2026},
  type    = {Trabajo de Fin de Grado},
  address = {Madrid, España},
  url     = {https://github.com/vtnehu121/adaptive-cycle-engine-drl}
}
```

### Licencia

Distribuido bajo licencia Apache License 2.0. Véase [LICENSE](LICENSE).

El dataset NASA C-MAPSS es de dominio público (NASA Prognostics Center of Excellence). T-MATS se distribuye bajo la NASA Open Source Agreement y **no** está incluido en este repositorio: debe descargarse por separado según el paso 7 de la instalación.
