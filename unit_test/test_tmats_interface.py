"""
Última Fecha de Modificación: 09/Aug/2026
Descripción test_tmats_interface.py: Tests unitarios de
src/simulators/tmats_interface.py. Cubre dos bloques independientes: la
validación de rutas y del modelo Simulink que TMATSSimulator hace al construirse,
y los tests de regresión de los dos fixes estructurales aplicados al modelo
ACE_3stream_brayton.slx durante la auditoría del proyecto: (i) el desacoplamiento
del Splitter de tercer flujo respecto al parámetro `bpr_ts` y (ii) la activación
del customer bleed nativo del HPC.

Ambos fixes corrigen fallos silenciosos: antes de ellos las simulaciones
terminaban sin error, pero dos de las cinco variables de control no tenían efecto
alguno sobre el ciclo termodinámico, de modo que el corpus resultante era
físicamente inconsistente. Los tests de regresión existen para que ninguna
edición futura del .slx los reintroduzca sin que la suite lo detecte.

Los tests requieren el MATLAB Engine for Python: si el runtime no está
disponible, ambas clases se marcan como skip. Los dos tests de regresión levantan
una sesión de MATLAB y ejecutan simulaciones completas del ciclo, por lo que van
marcados como `slow` y pueden deseleccionarse con -m "not slow".
"""

import importlib.util
from pathlib import Path

import pytest


def matlab_engine_available() -> bool:
    """Detecta si el MATLAB Engine for Python está instalado en el entorno.

    Usa importlib.util.find_spec en lugar de un import dentro de un try/except
    porque importar matlab.engine tiene efectos secundarios costosos —localiza la
    instalación de MATLAB y carga bibliotecas nativas— que no interesan cuando
    solo se quiere decidir si los tests deben ejecutarse o saltarse.

    return: True si el paquete matlab.engine es importable en este entorno;
        False en caso contrario.
    """
    return importlib.util.find_spec('matlab.engine') is not None

# Marcador compartido por ambas clases: sin MATLAB Engine no hay nada que probar,
# ni siquiera la validación de rutas, porque tmats_interface.py importa
# matlab.engine a nivel de módulo.
requires_matlab = pytest.mark.skipif(
    not matlab_engine_available(),
    reason="MATLAB Engine for Python no está disponible en este entorno",
)

@requires_matlab
class TestTMATSSimulatorInit:
    """Tests de la validación de rutas en la construcción del simulador.

    TMATSSimulator comprueba la carpeta y el archivo .slx en __init__ sin arrancar
    todavía el Engine, y esa validación temprana es justamente lo que se verifica
    aquí. Por eso estos tests son baratos: operan sobre directorios temporales y
    no llegan a levantar MATLAB en ningún momento.
    """

    def test_init_valid_path(self, tmp_path):
        """Comprueba que la inicialización acepta un directorio con el .slx.

        Verifica también el estado inicial del objeto: `eng` a None y
        `simulation_count` a cero. Es necesario porque el arranque de MATLAB debe
        quedar diferido hasta start(); si __init__ abriera ya la sesión, cada
        validación de ruta costaría decenas de segundos.

        param tmp_path: Fixture de pytest con un directorio temporal donde se crea
            un ACE_3stream_brayton.slx vacío que hace de modelo simulado.
        return: None; falla si la construcción lanza o si el estado inicial no es
            el esperado.
        """
        model_file = tmp_path / 'ACE_3stream_brayton.slx'
        model_file.touch()

        from src.simulators.tmats_interface import TMATSSimulator
        sim = TMATSSimulator(models_path=str(tmp_path))
        assert sim.models_path == tmp_path
        assert sim.eng is None
        assert sim.simulation_count == 0

    def test_init_invalid_path(self, tmp_path):
        """Comprueba que la inicialización falla si el directorio no existe.

        Se exige FileNotFoundError en la construcción, y no un fallo posterior al
        simular, porque un error de ruta detectado tras levantar el Engine
        costaría el arranque completo de MATLAB antes de dar el diagnóstico.

        param tmp_path: Fixture de pytest usada como raíz temporal sobre la que se
            construye una ruta hija inexistente.
        return: None; falla si la construcción no lanza FileNotFoundError.
        """
        from src.simulators.tmats_interface import TMATSSimulator
        with pytest.raises(FileNotFoundError):
            TMATSSimulator(models_path=str(tmp_path / 'ruta_inexistente'))

    def test_init_missing_model(self, tmp_path):
        """Comprueba que la inicialización falla si falta el archivo .slx.

        Complementa al test anterior cubriendo el caso más traicionero: la carpeta
        models/ existe pero el modelo Simulink no está —repositorio clonado sin
        LFS, archivo renombrado—. Sin esta comprobación el error aparecería dentro
        de MATLAB, con un mensaje mucho más difícil de interpretar.

        param tmp_path: Fixture de pytest con un directorio temporal vacío, sin el
            archivo ACE_3stream_brayton.slx.
        return: None; falla si la construcción no lanza FileNotFoundError.
        """
        from src.simulators.tmats_interface import TMATSSimulator
        with pytest.raises(FileNotFoundError):
            TMATSSimulator(models_path=str(tmp_path))


@pytest.fixture(scope='class')
def simulator():
    """Arranca una sesión de MATLAB compartida por toda la clase de regresión.

    El ámbito de clase es obligado aquí: levantar el Engine y cargar T-MATS cuesta
    decenas de segundos, y cada test de regresión ejecuta dos simulaciones. El
    yield garantiza que la sesión se cierra con stop() aunque algún test falle, de
    modo que no queden procesos de MATLAB huérfanos tras la suite.

    return: Generador que cede la instancia de TMATSSimulator ya arrancada, o skip
        de la clase completa si models/ACE_3stream_brayton.slx no está presente.
    """
    from src.simulators.tmats_interface import TMATSSimulator
    models_dir = Path('models').resolve()
    if not (models_dir / 'ACE_3stream_brayton.slx').exists():
        pytest.skip("Modelo Simulink no disponible en models/")
    sim = TMATSSimulator(models_path=str(models_dir))
    sim.start()
    yield sim
    sim.stop()

@requires_matlab
class TestFixesRegression:
    """Tests de regresión de los dos fixes estructurales del modelo Simulink.

    Su propósito es garantizar que ningún cambio futuro sobre el .slx reintroduzca
    los bugs corregidos en la auditoría. Ambos se comprueban por sensibilidad: se
    simulan dos condiciones idénticas salvo en el parámetro bajo estudio y se
    exige que la magnitud afectada responda. Es la única forma de detectar un
    parámetro desconectado, porque un bloque constante en lugar de una entrada
    dinámica no genera error, solo un resultado que ignora la consigna.
    """

    @pytest.mark.slow
    def test_bpr_ts_drives_bpr(self, simulator):
        """Fix regression: el Splitter debe consumir dinámicamente sim_bpr_ts.

        Antes del fix, `Splitter_TS_BPR_Const` estaba fijado a 0.3 y el parámetro
        escrito desde Python era ignorado, resultando en Corr(bpr_ts, BPR) ≈ 0
        sobre el corpus. Tras el fix, dos simulaciones con bpr_ts distintos deben
        producir BPR distintos y monotónicamente crecientes.

        param simulator: Fixture con la sesión de TMATSSimulator ya arrancada.
        return: None; falla si el BPR total no crece al aumentar bpr_ts, señal de
            que el Splitter ha vuelto a quedar desacoplado del parámetro.
        """
        cond_low = {'altitude': 35000, 'mach': 0.85, 'tra': 100,
                    'bpr_ts': 0.10, 'bleed_fraction': 0.03}
        cond_high = {'altitude': 35000, 'mach': 0.85, 'tra': 100,
                     'bpr_ts': 0.50, 'bleed_fraction': 0.03}

        result_low = simulator.run(**cond_low)
        result_high = simulator.run(**cond_high)

        assert result_high['BPR'] > result_low['BPR'], (
            "El BPR total no responde a bpr_ts; el Splitter podría "
            "estar de nuevo desacoplado del parámetro.")

    @pytest.mark.slow
    def test_bleed_fraction_produces_flow(self, simulator):
        """Fix regression: el customer bleed del HPC debe estar activo.

        Antes del fix, `CBLDEN_M` estaba desactivado y bleed_fraction no producía
        flujo real. Tras el fix, W_bleed debe crecer linealmente con
        bleed_fraction manteniendo el resto de las condiciones constantes.

        param simulator: Fixture con la sesión de TMATSSimulator ya arrancada.
        return: None; falla si W_bleed no crece al aumentar bleed_fraction, lo que
            apuntaría a que el customer bleed del HPC vuelve a estar inactivo.
        """
        cond_low = {'altitude': 30000, 'mach': 0.80, 'tra': 90,
                    'bpr_ts': 0.30, 'bleed_fraction': 0.02}
        cond_high = {'altitude': 30000, 'mach': 0.80, 'tra': 90,
                     'bpr_ts': 0.30, 'bleed_fraction': 0.08}

        result_low = simulator.run(**cond_low)
        result_high = simulator.run(**cond_high)

        assert result_high['W_bleed'] > result_low['W_bleed'], (
            "El customer bleed no responde a bleed_fraction; verificar "
            "que CBLDEN_M='on' y C_CBD_M='sim_bleed_fraction' en el HPC.")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
