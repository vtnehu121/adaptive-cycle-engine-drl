%% Última Fecha de Modificación: 08/Aug/2026
%  Descripción modify_JT9D_to_ACE.m: Añade al modelo ACE_3stream_brayton
%  (previamente creado como copia del JT9D base) los bloques Simulink de los
%  subsistemas del tercer flujo del motor ACE: splitter, conducto (duct),
%  tobera variable (NozzleVar) y sangrado del HPC (Bleed). Los bloques se
%  posicionan relativos a la topología del JT9D y se añaden las señales de
%  control necesarias (bloques From/Goto para Splitter_TS_BPR, Noz_ts_NErr y
%  Pamb). Las CONEXIONES entre bloques se establecen manualmente en Simulink
%  tras ejecutar este script; el modelo modificado se guarda con el mismo
%  nombre (ACE_3stream_brayton.slx).
%
%  La inserción se automatiza para garantizar que el posicionamiento, el
%  nombrado y la configuración de los bloques sean idénticos entre
%  ejecuciones y queden documentados en código versionable.
%
%  Requiere:
%    - init_ACE.m ejecutado
%    - ACE_3stream_brayton.slx abierto (creado como copia del JT9D base)

% Dimensiones de los bloques
BLOCK_WIDTH         = 80;    % Ancho estándar de los bloques T-MATS
BLOCK_HEIGHT_STD    = 60;    % Alto de bloques principales (Splitter, NozzleVar)
BLOCK_HEIGHT_SLIM   = 40;    % Alto de bloques delgados (Duct, Bleed)
BLOCK_HEIGHT_TAG    = 14;    % Alto de bloques de tag (From/Goto)

% Offsets del Splitter_TS respecto al Splitter base del JT9D
SPLITTER_TS_DX      = 100;   % Desplazamiento horizontal del Splitter_TS
SPLITTER_TS_DY      = 150;   % Desplazamiento vertical del Splitter_TS

% Layout del tercer flujo (Splitter_TS → TS_Duct → Noz_ts)
COMPONENT_STRIDE_X  = 150;   % Distancia horizontal entre componentes del tercer flujo
DUCT_DY_UNDER_SPLIT = 80;    % Desplazamiento vertical del duct bajo el splitter

% Offsets del sangrado (Bleed_ACE) respecto al HPC
BLEED_DX_FROM_HPC   = 50;    % Desplazamiento horizontal del bleed
BLEED_DY_FROM_HPC   = 80;    % Desplazamiento vertical del bleed (bajo el HPC)

% Offsets de las señales de control (bloques From/Goto)
FROM_TAG_DX_LEFT    = 120;   % Desplazamiento del From respecto al bloque destino
FROM_TAG_WIDTH      = 60;    % Ancho del bloque From/Goto
GOTO_TAG_DX_RIGHT   = 120;   % Desplazamiento del Goto respecto al bloque origen
GOTO_TAG_DX_END     = 200;   % Desplazamiento final del Goto
TAG_DY_OFFSET       = 20;    % Desplazamiento vertical de los tags respecto al bloque
PAMB_TAG_DX_LEFT    = 80;    % Desplazamiento del Pamb From respecto al Noz_ts
PAMB_TAG_DX_END     = 20;    % Desplazamiento final del Pamb From
PAMB_DY_OFFSET      = 40;    % Desplazamiento vertical del Pamb From

% Precondiciones
model = 'ACE_3stream_brayton';
plant = [model '/Plant model'];

if ~bdIsLoaded(model)
    error('Modelo %s no cargado. Ejecutar open_system(''%s'') antes.', model, model);
end

% Verificar que el bloque base (Splitter del JT9D) existe
try
    splitter_pos = get_param([plant '/Splitter'], 'Position');
catch
    error(['Bloque Splitter no encontrado en %s.\n' ...
           'Verifique que el modelo cargado corresponde al JT9D base ' ...
           'antes de aplicar la modificación ACE.'], plant);
end

% Verificar idempotencia: si los bloques del ACE ya existen, abortar
if ~isempty(find_system(model, 'Name', 'Splitter_TS'))
    warning(['El bloque Splitter_TS ya existe en %s. ' ...
             'Para re-ejecutar el script, recargue primero el JT9D base ' ...
             '(close_system(''%s'', 0); open_system(''JT9D_base'')).'], ...
            model, model);
    return;
end

% Cargar librerías T-MATS necesarias:
%   Splitter -> para el bloque Splitter_TS del tercer flujo
%   Duct     -> para TS_Duct
%   Turbo    -> para NozzleVar y Bleed
load_system('Lib_Turbo_Splitter_TMATS');
load_system('Lib_Turbo_Duct_TMATS');
load_system('TMATS_Turbo');

% Splitter del tercer flujo
ts_splitter_x = splitter_pos(3) + SPLITTER_TS_DX;
ts_splitter_y = splitter_pos(2) + SPLITTER_TS_DY;
add_block('Lib_Turbo_Splitter_TMATS/Splitter', [plant '/Splitter_TS'], ...
    'Position', [ts_splitter_x, ts_splitter_y, ...
                 ts_splitter_x + BLOCK_WIDTH, ts_splitter_y + BLOCK_HEIGHT_STD]);

% Conducto (Duct) del tercer flujo
ts_duct_x = ts_splitter_x + COMPONENT_STRIDE_X;
ts_duct_y = ts_splitter_y + DUCT_DY_UNDER_SPLIT;
add_block('Lib_Turbo_Duct_TMATS/Duct', [plant '/TS_Duct'], ...
    'Position', [ts_duct_x, ts_duct_y, ...
                 ts_duct_x + BLOCK_WIDTH, ts_duct_y + BLOCK_HEIGHT_SLIM]);

% Tobera variable del tercer flujo
noz_ts_x = ts_duct_x + COMPONENT_STRIDE_X;
noz_ts_y = ts_duct_y;
add_block('TMATS_Turbo/NozzleVar', [plant '/Noz_ts'], ...
    'Position', [noz_ts_x, noz_ts_y, ...
                 noz_ts_x + BLOCK_WIDTH, noz_ts_y + BLOCK_HEIGHT_STD]);

% Sangrado (Bleed) del HPC
hpc_pos = get_param([plant '/HPC'], 'Position');
bleed_x = hpc_pos(3) + BLEED_DX_FROM_HPC;
bleed_y = hpc_pos(4) + BLEED_DY_FROM_HPC;
add_block('TMATS_Turbo/Bleed', [plant '/Bleed_ACE'], ...
    'Position', [bleed_x, bleed_y, ...
                 bleed_x + BLOCK_WIDTH, bleed_y + BLOCK_HEIGHT_SLIM]);

% Etiquetas de señal (bloques From/Goto de Simulink)
% From del BPR del Splitter_TS
add_block('simulink/Signal Routing/From', [plant '/Splitter_TS_BPRI'], ...
    'Position', [ts_splitter_x - FROM_TAG_DX_LEFT, ts_splitter_y + TAG_DY_OFFSET, ...
                 ts_splitter_x - FROM_TAG_DX_LEFT + FROM_TAG_WIDTH, ...
                 ts_splitter_y + TAG_DY_OFFSET + BLOCK_HEIGHT_TAG], ...
    'GotoTag', 'Splitter_TS_BPR');

% Goto del error de la tobera Noz_ts
add_block('simulink/Signal Routing/Goto', [plant '/Noz_ts_NErrD'], ...
    'Position', [noz_ts_x + GOTO_TAG_DX_RIGHT, noz_ts_y + TAG_DY_OFFSET, ...
                 noz_ts_x + GOTO_TAG_DX_END, ...
                 noz_ts_y + TAG_DY_OFFSET + BLOCK_HEIGHT_TAG], ...
    'GotoTag', 'Noz_ts_NErr');

% From de la presión ambiente para Noz_ts
add_block('simulink/Signal Routing/From', [plant '/Noz_ts_Pamb1'], ...
    'Position', [noz_ts_x - PAMB_TAG_DX_LEFT, noz_ts_y + PAMB_DY_OFFSET, ...
                 noz_ts_x - PAMB_TAG_DX_END, ...
                 noz_ts_y + PAMB_DY_OFFSET + BLOCK_HEIGHT_TAG], ...
    'GotoTag', 'Pamb');


models_dir  = fileparts(mfilename('fullpath'));
output_path = fullfile(models_dir, [model '.slx']);
save_system(model, output_path);

fprintf('Bloques añadidos: Splitter_TS, TS_Duct, Noz_ts, Bleed_ACE\n');
fprintf('Etiquetas de señal (From/Goto): Splitter_TS_BPR, Noz_ts_NErr, Pamb\n');
fprintf('Modelo guardado en: %s\n', output_path);
fprintf('\nPróximo paso: conectar manualmente las líneas de señal en Simulink\n');
fprintf('              y ejecutar setup_ACE_params.m para inicializar parámetros.\n');