%% Última Fecha de Modificación: 08/Aug/2026
%
%  Descripción setup_ACE_params.m: Define la estructura ACE.* con todos los
%  parámetros físicos y operacionales del motor de ciclo adaptativo: constantes
%  físicas (termodinámicas e ISA), envelope de vuelo, fases de misión, parámetros 
%  de diseño de cada componente, rangos de los actuadores de geometría variable, 
%  límites de seguridad, modelo de degradación C-MAPSS y lista de sensores. 
%
%  Centraliza aquí los valores para que MATLAB y el código Python partan de una única fuente común.
%  Los valores están respaldados por:
%    - Estándares ICAO/ISA para constantes atmosféricas (T, P, ρ, gradiente)
%    - T-MATS JT9D NPSS (Chapman et al. 2014) para parámetros de componentes
%    - ASTM D1655 (combustible Jet-A1)
%    - Saxena, Goebel, Simon PHM 2008 (modelo de degradación C-MAPSS)
%    - Coherencia con ace_env.py (safety limits y actuadores del entorno DRL)
%  Los rangos operativos específicos del ACE (actuadores, márgenes surge)
%  son decisiones de diseño del proyecto.

if exist('ACE', 'var')
    warning('La estructura ACE ya existe en el workspace. Se sobrescribirá.');
end

%% [1] Constantes físicas - Atmósfera Estándar Internacional (ICAO)

ACE.const.R_air      = 287.058;   % [J/(kg·K)] Constante gas ideal aire seco
ACE.const.R_air_imp  = 1716;      % [ft·lbf/(slug·°R)] Sistema imperial

ACE.const.gamma_c    = 1.4;       % [-] Ratio cp/cv aire frío
ACE.const.gamma_h    = 1.33;      % [-] Ratio cp/cv productos combustión
ACE.const.cp_c       = 1004;      % [J/(kg·K)] cp aire frío
ACE.const.cp_h       = 1148;      % [J/(kg·K)] cp productos combustión

ACE.const.g          = 9.81;      % [m/s²] Gravedad estándar
ACE.const.T0_sl      = 288.15;    % [K] Temperatura sea level ISA
ACE.const.T0_sl_R    = 518.67;    % [°R] Temperatura sea level ISA
ACE.const.P0_sl      = 101325;    % [Pa] Presión sea level ISA
ACE.const.P0_sl_psi  = 14.696;    % [psia] Presión sea level ISA
ACE.const.rho0_sl    = 1.225;     % [kg/m³] Densidad sea level ISA
ACE.const.lapse_rate = 0.0065;    % [K/m] Gradiente térmico troposférico


%% [2] Envolvente de vuelo del motor ACE
%
% Distinguimos dos rangos con propósitos distintos:
%   - Envelope teórico de diseño: límites operacionales del envelope
%     del motor (usado como factores de normalización DRL en ace_env.py).
%   - Rango experimental muestreado: valores efectivamente muestreados
%     por muestreo estratificado uniforme en el corpus ace_dataset_5000.csv.
% 
% El corpus muestrea aproximadamente el 85% del envelope teórico en cada
% dimensión (altitud, mach y tra), dejando margen para condiciones no
% exploradas experimentalmente (out-of-distribution, OOD).

% Envelope teórico de diseño (coherente con ace_env.py factores DRL)
ACE.flight.alt_min_design   = 0;       % [ft] Nivel del mar
ACE.flight.alt_max_design   = 50000;   % [ft] Techo operacional teórico
ACE.flight.mach_min_design  = 0.0;     % [-]  Estacionario en tierra
ACE.flight.mach_max_design  = 2.0;     % [-]  Mach teórico máximo
ACE.flight.tra_min_design   = 20;      % [%]  TRA mínimo diseño (idle)
ACE.flight.tra_max_design   = 100;     % [%]  TRA máximo (WOT)

% Rango experimental muestreado (corpus ace_dataset_5000.csv, redondeado)
ACE.flight.alt_min_corpus   = 2;       % [ft] Mínimo corpus
ACE.flight.alt_max_corpus   = 42000;   % [ft] Máximo corpus 
ACE.flight.mach_min_corpus  = 0.10;    % [-]  Mínimo corpus
ACE.flight.mach_max_corpus  = 1.80;    % [-]  Máximo corpus
ACE.flight.tra_min_corpus   = 30;      % [%]  Mínimo corpus
ACE.flight.tra_max_corpus   = 100;     % [%]  Máximo corpus

%% [3] Fases de misión operativas
%
% Perfiles alt/mach/tra basados en operativa estándar comercial y militar:
%   - Cruise (35000 ft, M=0.85): Estándar comercial widebody (B787, A350, A380)
%   - Combat (15000 ft, M=1.6): Fighter dash supersónico típico (F-15/F-16/Eurofighter)
%   - Transiciones alt-mach: coherentes con límite universal 250 kt < 10000 ft
%     (FAA 14 CFR 91.117 y equivalentes internacionales)

ACE.mission.takeoff  = struct('alt', 0,     'mach', 0.30, 'tra', 100, 'duration', 20);
ACE.mission.climb    = struct('alt', 20000, 'mach', 0.60, 'tra', 90,  'duration', 40);
ACE.mission.cruise   = struct('alt', 35000, 'mach', 0.85, 'tra', 80,  'duration', 100);
ACE.mission.combat   = struct('alt', 15000, 'mach', 1.60, 'tra', 100, 'duration', 30);
ACE.mission.descent  = struct('alt', 10000, 'mach', 0.50, 'tra', 40,  'duration', 30);

%% [4] Componentes del motor - Parámetros de diseño verificables
%
% Valores derivados de T-MATS JT9D NPSS (Chapman et al. 2014):
%   github.com/nasa/T-MATS/blob/master/Resources/JT9D_Public_NPSSv241/JT9D.mdl
% JT9D real (JT9D-7A): OPR ≈ 22-27, BPR 4.8:1 (P&W spec sheet, dependiendo de la variante)
% Eficiencias y coeficientes en rangos típicos de la industria de propulsión.


% [4.1] Inlet (admisión)
ACE.inlet.PR_nominal   = 1.0;     % [-]  Sin pérdidas ideales
ACE.inlet.eta_ram      = 0.98;    % [-]  T-MATS: 0.995-0.998

% [4.2] Fan (ventilador)
ACE.fan.PR_design      = 1.6;     % [-]      
ACE.fan.eta_isen       = 0.93;    % [-]      
ACE.fan.W_design       = 1200;    % [lbm/s]  
ACE.fan.Rline_design   = 2.0;     % [-]     

% [4.3] LPC (compresor baja presión)
ACE.lpc.eta_isen       = 0.88;    % [-]      
ACE.lpc.Rline_design   = 2.0;     % [-]      

% [4.4] HPC (compresor alta presión)
% PR calculado: OPR JT9D (26.7) / (fan.PR · lpc.PR) = 26.7 / (1.6 · 2.5) ≈ 6.7
ACE.hpc.PR_design      = 6.7;     % [-]      OPR JT9D consistente 
ACE.hpc.eta_isen       = 0.86;    % [-]      
ACE.hpc.Rline_design   = 2.0;     % [-]      

% [4.5] Combustor (cámara de combustión)
ACE.burner.eta_comb    = 0.995;   % [-]     Combustor moderno (>0.99)
ACE.burner.dP_ratio    = 0.04;    % [-]     Pérdida carga típica ~4%
ACE.burner.LHV         = 43100;   % [kJ/kg] Jet-A1 típico (ASTM D1655: 42800 kJ/kg mínimo)
ACE.burner.FAR_design  = 0.025;   % [-]     Jet-A1 punto diseño
ACE.burner.T4_max      = 3200;    % [°R]    

% [4.6] HPT (turbina alta presión)
ACE.hpt.eta_isen       = 0.91;    % [-]  
ACE.hpt.cooling_frac   = 0.05;    % [-]  Extracción refrigeración 5%

% [4.7] LPT (turbina baja presión)
ACE.lpt.eta_isen       = 0.92;    % [-]  

% [4.8] Toberas (nozzles)
ACE.nozzle_core.Cd     = 0.98;    % [-]  Coef. descarga 
ACE.nozzle_core.Cv     = 0.99;    % [-]  Coef. velocidad 
ACE.nozzle_bypass.Cd   = 0.98;    % [-]  
ACE.nozzle_bypass.Cv   = 0.99;    % [-]  
ACE.nozzle_ts.Cd       = 0.97;    % [-]  Tercer flujo (mixing losses)
ACE.nozzle_ts.Cv       = 0.98;    % [-]  

% [4.9] Splitter (bifurcador tres flujos)
% BPR nominal JT9D real: 4.8:1 (P&W JT9D-7A specification)
ACE.splitter.BPR_design = 4.8;    % [-]  JT9D real 

%% [5] Actuadores de geometría variable
%
% Rangos de los 4 actuadores del ACE, coherentes con el espacio 
% de acción del entorno DRL.
%
% Los rangos son simétricos respecto al ángulo nominal 0° para 
% simplificar el mapeo entre las acciones normalizadas [-1, 1] 
% del DRL y los ángulos físicos.

% [5.1] VFGV (Variable Fan Guide Vanes)
ACE.act.vfgv.angle_min     = -20;    % [deg]   
ACE.act.vfgv.angle_max     = +20;    % [deg]   
ACE.act.vfgv.angle_nominal = 0;      % [deg]
ACE.act.vfgv.rate_max      = 10;     % [deg/s]

% [5.2] VGV-T (Variable Guide Vanes Turbine)
ACE.act.vgvt.angle_min     = -15;    % [deg]   
ACE.act.vgvt.angle_max     = +15;    % [deg]   
ACE.act.vgvt.angle_nominal = 0;      % [deg]
ACE.act.vgvt.rate_max      = 8;      % [deg/s]
ACE.act.vgvt.eta_factor    = 0.02;   % [1/deg]

% [5.3] Afterburner (postcombustor)
ACE.act.afterburner.T_max        = 3200;   % [°R]
ACE.act.afterburner.FAR_max      = 0.06;   % [-]
ACE.act.afterburner.eta_comb     = 0.90;   % [-]
ACE.act.afterburner.thrust_mult  = 1.5;    % [-]
ACE.act.afterburner.active       = false;  % Estado por defecto

% [5.4] Bleed (extracción de aire para HPT/LPT cooling)
ACE.act.bleed.W31_frac      = 0.05;  % [-] HPT cooling fracción diseño
ACE.act.bleed.W32_frac      = 0.03;  % [-] LPT cooling fracción diseño
ACE.act.bleed.customer_frac = 0.02;  % [-] Customer bleed
ACE.act.bleed.W31_min       = 0.01;  % [-]
ACE.act.bleed.W31_max       = 0.10;  % [-]
ACE.act.bleed.W32_min       = 0.01;  % [-]
ACE.act.bleed.W32_max       = 0.08;  % [-]

%% [6] Límites operacionales de seguridad (Safety limits)
%
% Los márgenes surge SmFan y SmHPC se aproximan mediante la distancia
% relativa a la velocidad máxima del eje correspondiente:
%   SmFan ≈ (Nf_max - Nf) / Nf_max × 100
%   SmHPC ≈ (Nc_max - Nc) / Nc_max × 100
% Esta simplificación es necesaria porque el PINN no incluye los mapas
% surge/flow del compresor entre sus outputs.
% (ver método _check_safety en src/environments/ace_env.py)

% Temperaturas
ACE.limits.T4_max      = 3200;    % [°R]   
ACE.limits.T4_warning  = 3000;    % [°R]   Umbral pre-alarma
ACE.limits.EGT_max     = 1800;    % [°R]   Límite operativo ACE

% Márgenes surge 
ACE.limits.SmFan_min   = 8.0;     % [%]   
ACE.limits.SmHPC_min   = 5.0;     % [%]   

% Velocidades máximas
ACE.limits.Nf_max      = 6000;    % [rpm]  
ACE.limits.Nc_max      = 12000;   % [rpm]  

% Empuje y consumo (referencia, no aplicado como safety limit en DRL)
ACE.limits.thrust_max  = 90000;   % [lbf] Máximo esperable del ACE
ACE.limits.thrust_min  = 0;       % [lbf]
ACE.limits.sfc_max     = 1.2;     % [lbm/lbf/hr] 

%% [7] Modelo de degradación C-MAPSS
%
% Parámetros base del modelo exponencial de Saxena, Goebel, Simon (PHM 2008):
% h(t) = 1 - exp(-a·t^b)

ACE.degrad.model.a_range = [0.001, 0.003];  % [-] 
ACE.degrad.model.b_range = [1.4, 1.6];      % [-] 

% Umbrales de fallo (thresholds Saxena PHM 2008)
ACE.degrad.fail.SmFan = 15.0;   % [%] 
ACE.degrad.fail.SmLPC = 15.0;   % [%] 
ACE.degrad.fail.SmHPC = 15.0;   % [%] 
ACE.degrad.fail.EGT   = 2.0;    % [%] 

% Tabla 3 de Saxena, Goebel, Simon (PHM 2008): 
% valores de referencia de desgaste natural de fondo por componente 
% Aplica a Fan, LPC, HPT y LPT.
%
% Nota: HPC no aparece en Table 3 porque es el fault mode principal del
% dataset FD001 y se modela con la ecuación exponencial ya declarada 
% (ver clase DegradationInjector en src/data_gen/degradation.py).
%
% Cada componente sigue trayectoria exponencial entre Initial y 6000 ciclos.

ACE.degrad.wear.fan_eff  = struct('initial', -0.18, 'c3000', -1.50, 'c6000', -2.85);  % [%]
ACE.degrad.wear.fan_flow = struct('initial', -0.26, 'c3000', -2.04, 'c6000', -3.65);  % [%]

ACE.degrad.wear.lpc_eff  = struct('initial', -0.62, 'c3000', -1.46, 'c6000', -2.61);  % [%]
ACE.degrad.wear.lpc_flow = struct('initial', -1.01, 'c3000', -2.08, 'c6000', -4.00);  % [%]

ACE.degrad.wear.hpt_eff  = struct('initial', -0.48, 'c3000', -2.63, 'c6000', -3.81);  % [%]
ACE.degrad.wear.hpt_flow = struct('initial', +0.08, 'c3000', +1.76, 'c6000', +2.57);  % [%]

ACE.degrad.wear.lpt_eff  = struct('initial', -0.10, 'c3000', -0.54, 'c6000', -1.08);  % [%]
ACE.degrad.wear.lpt_flow = struct('initial', +0.08, 'c3000', +0.26, 'c6000', +0.42);  % [%]


%% [8] Sensores C-MAPSS (Saxena, Goebel, Simon PHM 2008 Table 2)

ACE.sensors.names = { ...
    'T2',    ...  %  1 Total temperature at fan inlet [°R]
    'T24',   ...  %  2 Total temperature at LPC outlet [°R]
    'T30',   ...  %  3 Total temperature at HPC outlet [°R]
    'T50',   ...  %  4 Total temperature at LPT outlet [°R]
    'P2',    ...  %  5 Pressure at fan inlet [psia]
    'P15',   ...  %  6 Total pressure in bypass-duct [psia]
    'P30',   ...  %  7 Total pressure at HPC outlet [psia]
    'Nf',    ...  %  8 Physical fan speed [rpm]
    'Nc',    ...  %  9 Physical core speed [rpm]
    'epr',   ...  % 10 Engine pressure ratio (P50/P2) [-]
    'Ps30',  ...  % 11 Static pressure at HPC outlet [psia]
    'phi',   ...  % 12 Ratio of fuel flow to Ps30 [pps/psi]
    'NRf',   ...  % 13 Corrected fan speed [rpm]
    'NRc',   ...  % 14 Corrected core speed [rpm]
    'BPR',   ...  % 15 Bypass ratio [-]
    'farB',  ...  % 16 Burner fuel-air ratio [-]
    'htBleed',... % 17 Bleed enthalpy [-]
    'Nf_dmd',...  % 18 Demanded fan speed [rpm]
    'PCNfR_dmd',...% 19 Demanded corrected fan speed [rpm]
    'W31',   ...  % 20 HPT coolant bleed [lbm/s]
    'W32'    ...  % 21 LPT coolant bleed [lbm/s]
};

ACE.sim.n_samples = 5000;   % Muestras corpus (pipeline.py)
ACE.sim.seed      = 42;     % Semilla reproducibilidad

n_constants = length(fieldnames(ACE.const));
n_missions  = length(fieldnames(ACE.mission));
n_sensors   = length(ACE.sensors.names);
n_wear      = length(fieldnames(ACE.degrad.wear));

fprintf(['Parametros ACE cargados:\n' ...
         '  - %d constantes físicas\n' ...
         '  - %d fases de misión\n' ...
         '  - %d sensores C-MAPSS\n' ...
         '  - %d componentes con desgaste\n'], ...
         n_constants, n_missions, n_sensors, n_wear);