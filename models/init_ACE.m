%% Última Fecha de Modificación: 08/Aug/2026
%  Descripción init_ACE.m: Inicializa el entorno de simulación del motor ACE.
%  Resuelve las rutas del proyecto y de T-MATS, verifica que la librería esté
%  instalada donde se espera, la añade al path de la sesión de MATLAB y carga los
%  mapas de turbomaquinaria del JT9D en el Model WorkSpace. Es el primer script que
%  debe ejecutarse: sin él ni el modelo Simulink ni el resto de scripts encuentran
%  los bloques ni los mapas de componentes de los que dependen.
%
%  Dependencia externa: T-MATS (NASA Glenn Research Center)
%    https://github.com/nasa/T-MATS
%    Ubicación esperada: ../T-MATS-master (mismo nivel que la carpeta del proyecto)

% Verificar licencia de Simulink
if ~license('test', 'Simulink')
    error(['Este script requiere MATLAB con Simulink instalado.\n' ...
           'Verifique que su licencia incluye Simulink antes de continuar.']);
end

% Rutas del proyecto y T-MATS
models_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(models_dir);
tmats_root   = fullfile(fileparts(project_root), 'T-MATS-master', 'Trunk');

if ~isfolder(tmats_root)
    error(['T-MATS no encontrado en: %s\n\n' ...
           'INSTALACION DE T-MATS:\n' ...
           '  1. Descargue desde: https://github.com/nasa/T-MATS\n' ...
           '  2. Descomprima junto al proyecto (misma carpeta padre):\n\n' ...
           '     <padre>/\n' ...
           '     |-- T-MATS-master/\n' ...
           '     +-- adaptive-cycle-engine-drl/\n\n' ...
           '  3. Vuelva a ejecutar init_ACE.m'], tmats_root);
end

% Añadir T-MATS al path (evitando duplicados)
if ~contains(path, tmats_root)
    addpath(genpath(tmats_root));
    fprintf('T-MATS anadido al path\n');
else
    fprintf('T-MATS ya estaba en el path\n');
end

% Cargar mapas del JT9D en el Model WorkSpace
jt9d_setup = fullfile(tmats_root, 'TMATS_Examples', 'Example_JT9D', ...
                      'JT9D_setup_everything_Dyn.m');

if ~isfile(jt9d_setup)
    error(['Script de setup JT9D no encontrado: %s\n' ...
           'Verifique que la instalación de T-MATS incluye TMATS_Examples/Example_JT9D/'], ...
          jt9d_setup);
end

run(jt9d_setup);

fprintf('T-MATS cargado desde: %s\n', tmats_root);
fprintf('Mapas JT9D cargados en Model Workspace\n');
fprintf('Entorno ACE inicializado correctamente\n');
fprintf('\nPróximo paso: abrir ACE_3stream_brayton.slx y ejecutar setup_ACE_params.m\n');