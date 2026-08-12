"""
Última Fecha de Modificación: 09/Aug/2026
Descripción test_flight_envelope.py: Tests unitarios de
src/data_gen/flight_envelope.py. Verifica que el muestreador estratificado del
envelope de vuelo del motor ACE devuelve exactamente el número de muestras
pedido, con las columnas que consume el resto del pipeline, valores dentro de los
rangos físicos del envelope operativo, el reparto por fase de misión declarado en
REGIMES.

FlightEnvelope es el primer eslabón del pipeline de datos: define el dominio
sobre el que se generan el corpus sintético y el gemelo digital. Un sesgo en el
reparto por fase o un rango mal fijado no produciría ningún error visible, pero
desplazaría la distribución de entrenamiento de todos los modelos aguas abajo y
solo se manifestaría como un mal comportamiento del agente en operación.
"""

import pandas as pd
import pytest

from src.data_gen.flight_envelope import FlightEnvelope

# Distribución objetivo por fase de misión (fracciones sobre el total).
# Debe coincidir con la implementada en FlightEnvelope.
REGIME_FRACTIONS = {
    'takeoff': 0.15,
    'climb':   0.20,
    'cruise':  0.25,
    'combat':  0.25,
    'descent': 0.15,
}

class TestFlightEnvelope:
    """Tests del muestreador estratificado del envelope de vuelo.

    Cada test construye su propia instancia de FlightEnvelope con el tamaño de
    corpus mínimo que necesita, en lugar de compartir una fixture. La generación
    es puramente numérica y barata, y así queda explícito en cada caso qué
    cardinalidad exige la comprobación: 5000 muestras para el reparto por fase,
    unas pocas decenas para la estructura del DataFrame.
    """

    def test_generate_correct_size(self):
        """Comprueba que generate() devuelve exactamente n_samples filas.

        El reparto por fase es ``n_samples * weight`` truncado a entero, de modo
        que la suma de las cinco fases queda por debajo del total pedido;
        FlightEnvelope compensa ese déficit replicando filas al azar. Este test
        fija ese contrato: quien pide 100 muestras recibe 100, sin depender de que
        los pesos dividan exactamente el total.

        return: None; falla si el corpus generado no tiene la cardinalidad pedida.
        """
        env = FlightEnvelope(n_samples=100, seed=42)
        df = env.generate()
        assert len(df) == 100

    def test_generate_required_columns(self):
        """Comprueba que el corpus expone las columnas que consume el pipeline.

        Las cinco variables de condición (altitude, mach, tra, bpr_ts,
        bleed_fraction) son las que el simulador T-MATS inyecta en el workspace de
        MATLAB, y `regime` es la etiqueta de fase sobre la que se agrupan los
        análisis por régimen. Renombrar cualquiera de ellas rompería a esos
        consumidores lejos del punto donde se hizo el cambio.

        return: None; falla enumerando las columnas ausentes del DataFrame.
        """
        env = FlightEnvelope(n_samples=10, seed=42)
        df = env.generate()
        required = ['altitude', 'mach', 'tra',
                    'bpr_ts', 'bleed_fraction', 'regime']
        missing = [c for c in required if c not in df.columns]
        assert not missing, f"Faltan columnas: {missing}"

    def test_generate_physical_ranges(self):
        """Comprueba que los valores muestreados respetan el envelope del ACE.

        Los límites verificados son la envolvente global que resulta de unir los
        cinco hiperrectángulos de REGIMES definidos en flight_envelope.py:
            - altitude: [0, 42000] ft (unión de takeoff, climb, cruise, combat, descent)
            - mach:     [0, 2.0]   (rango subsónico hasta supersónico moderado)
            - tra:      [20, 100]  (Throttle Resolver Angle, %)
            - bpr_ts:   [0, 1.0]   (Third Stream BPR, fracción)
            - bleed_fraction: [0.01, 0.10] (customer bleed HPC, fracción)
        Es necesario porque el simulador T-MATS no tiene modelo válido fuera de ese
        dominio: una condición extrapolada no falla, converge a un punto de operación
        sin sentido físico que acabaría incorporado al corpus como si fuera legítimo.

        return: None; falla si altitude, mach, tra, bpr_ts o bleed_fraction se
            salen de sus límites operativos.
        """
        env = FlightEnvelope(n_samples=500, seed=42)
        df = env.generate()

        # altitude: unión de las 5 fases → [0, 42000] ft (cruise es el techo)
        assert df['altitude'].min() >= 0, \
            f"altitude min = {df['altitude'].min()}, esperado >= 0 ft"
        assert df['altitude'].max() <= 42000, \
            f"altitude max = {df['altitude'].max()}, esperado <= 42000 ft"

        assert df['mach'].min() >= 0, \
            f"mach min = {df['mach'].min()}, esperado >= 0"
        assert df['mach'].max() <= 2.0, \
            f"mach max = {df['mach'].max()}, esperado <= 2.0"

        assert df['tra'].min() >= 20, \
            f"tra min = {df['tra'].min()}, esperado >= 20 %"
        assert df['tra'].max() <= 100, \
            f"tra max = {df['tra'].max()}, esperado <= 100 %"

        assert df['bpr_ts'].min() >= 0, \
            f"bpr_ts min = {df['bpr_ts'].min()}, esperado >= 0"
        assert df['bpr_ts'].max() <= 1.0, \
            f"bpr_ts max = {df['bpr_ts'].max()}, esperado <= 1.0"

        # bleed_fraction acotado en [0.01, 0.10] según flight_envelope.py
        assert df['bleed_fraction'].min() >= 0.01, \
            f"bleed_fraction min = {df['bleed_fraction'].min()}, esperado >= 0.01"
        assert df['bleed_fraction'].max() <= 0.10, \
            f"bleed_fraction max = {df['bleed_fraction'].max()}, esperado <= 0.10"

    def test_regime_distribution(self):
        """Comprueba que el reparto por fase cumple las fracciones objetivo.

        Los pesos de REGIMES son lo que hace estratificado al muestreo: sin ellos
        el corpus sobrerrepresentaría regiones irrelevantes del envelope y dejaría
        sin cubrir los puntos donde el motor pasa realmente la misión. Se usan
        5000 muestras, la misma cardinalidad que el corpus final, para que el
        truncamiento entero del reparto no distorsione las fases de fracción baja.

        return: None; falla indicando la fase, el recuento esperado y el obtenido.
        """
        env = FlightEnvelope(n_samples=5000, seed=42)
        df = env.generate()
        counts = df['regime'].value_counts()

        for regime, fraction in REGIME_FRACTIONS.items():
            expected = int(5000 * fraction)
            assert counts.get(regime, 0) == expected, (
                f"Fase '{regime}': esperado {expected}, "
                f"obtenido {counts.get(regime, 0)}"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
