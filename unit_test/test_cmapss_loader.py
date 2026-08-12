"""
Última Fecha de Modificación: 09/Aug/2026
Descripción test_cmapss_loader.py: Tests unitarios de
src/preprocessing/cmapss_loader.py. Verifica que la clase CMAPSSLoader carga los
sub-datasets NASA C-MAPSS con las dimensiones publicadas por Saxena, Goebel y
Simon (PHM 2008), que el esquema de columnas coincide con la Table 2 de ese
trabajo, que los sensores de FD001 caen en sus rangos físicos típicos y que tanto
el health index como la normalización MinMax quedan acotados en el intervalo
unidad.

Estas comprobaciones son necesarias porque los archivos originales de C-MAPSS son
texto plano sin cabecera: los nombres de las 26 columnas los impone el loader, de
modo que un solo nombre desplazado asignaría en silencio los valores de un sensor
a otro. El corpus se cargaría, los modelos entrenarían y los resultados serían
falsos sin que ninguna excepción lo delatase.

Los tests se saltan automáticamente si el directorio data/CMAPSS_DATA no está
presente, permitiendo ejecutar la suite en entornos donde el dataset no se ha
descargado.
"""

from pathlib import Path

import pytest

from src.preprocessing.cmapss_loader import CMAPSSLoader


@pytest.fixture(scope='class')
def loader():
    """Instancia el cargador de C-MAPSS una única vez para toda la clase.

    El ámbito de clase evita reconstruir el loader —y con él revalidar la ruta y
    vaciar las cachés de sub-datasets— en cada test, de forma que los archivos de
    FD001 se leen una sola vez para las cinco comprobaciones. Si el dataset no
    está descargado la fixture llama a pytest.skip en lugar de fallar, porque la
    ausencia de datos no es un defecto del código bajo prueba.

    return: Instancia de CMAPSSLoader apuntando a data/CMAPSS_DATA, o skip de la
        clase completa si el directorio no existe.
    """
    data_path = Path('data/CMAPSS_DATA')
    if not data_path.exists():
        pytest.skip("Dataset C-MAPSS no disponible en data/CMAPSS_DATA")
    return CMAPSSLoader(data_path=str(data_path))


class TestCMAPSSLoader:
    """Tests del cargador y preprocesador de los sub-datasets C-MAPSS.

    Agrupa las comprobaciones que comparten la fixture `loader` de ámbito de
    clase. Van juntas porque todas validan el mismo contrato: que lo que el
    loader entrega al resto del pipeline es el dataset publicado por la NASA y no
    una reinterpretación silenciosa de sus columnas.
    """

    def test_load_fd001_dimensions(self, loader):
        """Comprueba que FD001 se carga con las dimensiones publicadas.

        FD001 contiene 20 631 ciclos de 100 motores según Saxena et al. (2008).
        Cualquier desviación indica que el parser ha perdido filas, ha leído otro
        sub-dataset o ha interpretado mal el separador de columnas. Se verifica
        además que aparece la columna RUL, que no existe en el archivo original y
        de la que dependen el health index y el entrenamiento de la RNN de salud.

        param loader: Fixture con la instancia de CMAPSSLoader.
        return: None; falla si el número de filas, el número de motores o la
            columna RUL no coinciden con lo esperado.
        """
        df = loader.load_subset('FD001')
        assert len(df) == 20631
        assert df['unit_id'].nunique() == 100
        assert 'RUL' in df.columns

    def test_column_names_match_saxena(self, loader):
        """Comprueba que los 21 sensores coinciden con la Table 2 de Saxena et al. (2008).

        Es la comprobación que sostiene a todas las demás: al no venir cabecera en
        el archivo, el mapeo columna-sensor lo fija COLUMN_NAMES en el loader y
        nada en tiempo de ejecución lo contrasta contra la fuente. Este test hace
        explícita esa lista frente al paper original.

        param loader: Fixture con la instancia de CMAPSSLoader.
        return: None; falla enumerando los sensores ausentes del DataFrame.
        """
        df = loader.load_subset('FD001')
        expected_sensors = ['T2', 'T24', 'T30', 'T50',
                            'P2', 'P15', 'P30',
                            'Nf', 'Nc', 'epr', 'Ps30', 'phi',
                            'NRf', 'NRc', 'BPR', 'farB',
                            'htBleed', 'Nf_dmd', 'PCNfR_dmd',
                            'W31', 'W32']
        missing = [s for s in expected_sensors if s not in df.columns]
        assert not missing, f"Faltan sensores: {missing}"

    def test_sensor_ranges_fd001(self, loader):
        """Comprueba que los sensores de FD001 caen en sus rangos típicos.

        Contrasta las medias de T2 (temperatura total a la entrada del fan), Nf y
        Nc (velocidades de eje bajo y alto) y BPR (bypass ratio) contra los
        órdenes de magnitud de la Table 2 de Saxena et al. (2008). Complementa al
        test de nombres: detecta el caso en que las columnas están correctamente
        etiquetadas pero leídas con la escala o las unidades equivocadas, algo que
        una comprobación de nombres no puede ver.

        param loader: Fixture con la instancia de CMAPSSLoader.
        return: None; falla si alguna media se sale de su rango de referencia.
        """
        df = loader.load_subset('FD001')
        assert 480 < df['T2'].mean() < 520
        assert 2300 < df['Nf'].mean() < 2500
        assert 8900 < df['Nc'].mean() < 9200
        assert 8 < df['BPR'].mean() < 9

    def test_health_index_bounds(self, loader):
        """Comprueba que el health index derivado del RUL queda acotado en [0, 1].

        HI es la etiqueta objetivo de la RNN de salud y el entorno de RL lo
        interpreta como fracción de vida remanente. Si se saliera del intervalo
        unidad —por un RUL negativo o por una normalización mal referenciada—, la
        recompensa y el criterio de parada del entorno quedarían mal escalados sin
        ningún síntoma visible durante el entrenamiento.

        param loader: Fixture con la instancia de CMAPSSLoader.
        return: None; falla si la columna HI no se genera o excede [0, 1].
        """
        df = loader.load_subset('FD001')
        df = loader.compute_health_index(df)
        assert 'HI' in df.columns
        assert df['HI'].min() >= 0
        assert df['HI'].max() <= 1

    def test_minmax_normalization(self, loader):
        """Comprueba que la normalización MinMax acota los sensores informativos.

        Recorre los sensores de INFORMATIVE_SENSORS admitiendo una tolerancia de
        ±0.01 que absorbe el error de coma flotante del escalado. Es necesario
        porque las redes del proyecto asumen entradas en el intervalo unidad: un
        sensor fuera de rango dominaría los gradientes y desequilibraría el
        entrenamiento sin producir ningún error.

        param loader: Fixture con la instancia de CMAPSSLoader.
        return: None; falla si algún sensor informativo queda fuera de [0, 1] tras
            aplicar la tolerancia numérica.
        """
        df = loader.load_subset('FD001')
        df_norm = loader.normalize(df, method='minmax')
        for sensor in loader.INFORMATIVE_SENSORS:
            if sensor in df_norm.columns:
                assert df_norm[sensor].min() >= -0.01
                assert df_norm[sensor].max() <= 1.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
