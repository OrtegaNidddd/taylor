"""Módulo de componentes de interfaz, controles y presets didácticos para Streamlit."""

from dataclasses import dataclass
import streamlit as st

PRESETS = {
    "Personalizada": "",
    "Ondulatoria (sin(x) * cos(y))": "sin(x) * cos(y)",
    "Paraboloide Elíptico (x^2 + y^2)": "x^2 + y^2",
    "Silla de Montar (x^2 - y^2)": "x^2 - y^2",
    "Campana Gaussiana (exp(-(x^2 + y^2)))": "exp(-(x^2 + y^2))",
    "Superficie Mono-cuadrática (x^2 + 2*y)": "x^2 + 2*y",
    "Cúbica de Mono (x^3 - 3*x*y^2)": "x^3 - 3*x*y^2",
    "Logarítmica Suave (log(1 + x^2 + y^2))": "log(1 + x^2 + y^2)",
}


@dataclass
class UIParameters:
    """Parámetros seleccionados por el usuario en la interfaz."""
    func_str: str
    x0: float
    y0: float
    degree: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    tolerance: float
    show_real: bool
    show_taylor: bool
    show_anchor: bool
    show_tolerance_mask: bool
    real_opacity: float
    taylor_opacity: float
    show_error_surface: bool


def render_sidebar_controls() -> UIParameters:
    """Renderiza todos los controles de la barra lateral y retorna los parámetros configurados."""
    st.sidebar.header("Parámetros del Modelo")

    # 1. Preset o función personalizada
    selected_preset = st.sidebar.selectbox(
        "Catálogo de Ejemplos Didácticos",
        options=list(PRESETS.keys()),
        index=1,  # Default: sin(x) * cos(y)
    )

    default_func = PRESETS[selected_preset] if selected_preset != "Personalizada" else "sin(x) + cos(y)"
    func_str = st.sidebar.text_input(
        "Función f(x, y)",
        value=default_func,
        help="Ingresa cualquier función analítica. Ej: sin(x)*cos(y), x^2 + y^2, exp(-(x^2+y^2))",
    )

    # 2. Punto de expansión (x0, y0)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Punto de Apoyo $(x_0, y_0)$")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x0 = float(st.number_input("x₀", value=0.0, step=0.1, format="%.2f"))
    with col2:
        y0 = float(st.number_input("y₀", value=0.0, step=0.1, format="%.2f"))

    # 3. Grado del polinomio
    degree = int(
        st.sidebar.slider(
            "Grado del Polinomio (n)",
            min_value=1,
            max_value=5,
            value=1,
            help="Grado 1 = Plano Tangente. Grados mayores incorporan curvatura y órdenes superiores.",
        )
    )

    # 4. Ventana de visualización (dominio)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ventana de Visualización")
    x_range = st.sidebar.slider(
        "Rango en X",
        min_value=-10.0,
        max_value=10.0,
        value=(max(-5.0, x0 - 3.0), min(5.0, x0 + 3.0)),
        step=0.5,
    )
    y_range = st.sidebar.slider(
        "Rango en Y",
        min_value=-10.0,
        max_value=10.0,
        value=(max(-5.0, y0 - 3.0), min(5.0, y0 + 3.0)),
        step=0.5,
    )

    # 5. Tolerancia del error
    st.sidebar.markdown("---")
    st.sidebar.subheader("Análisis del Residuo")
    tolerance = float(
        st.sidebar.number_input(
            "Umbral de Tolerancia (|Error| ≤ ε)",
            min_value=0.001,
            max_value=5.0,
            value=0.20,
            step=0.05,
            format="%.3f",
            help="Define la región de convergencia o validez donde la aproximación difiere menos de ε de la función original.",
        )
    )

    # 6. Opciones de visualización
    st.sidebar.markdown("---")
    st.sidebar.subheader("Capas y Visibilidad 3D")
    show_real = st.sidebar.checkbox("Superficie Original f(x, y)", value=True)
    show_taylor = st.sidebar.checkbox("Aproximación de Taylor P_n(x, y)", value=True)
    show_anchor = st.sidebar.checkbox("Punto de Anclaje (x₀, y₀)", value=True)
    show_tolerance_mask = st.sidebar.checkbox("Zona de Tolerancia (|Error| ≤ ε)", value=True)
    show_error_surface = st.sidebar.checkbox("Ver Superficie de Error Dedicada", value=False)

    with st.sidebar.expander("Ajustes de Opacidad"):
        real_opacity = st.slider("Opacidad de f(x, y)", 0.1, 1.0, 0.85, 0.05)
        taylor_opacity = st.slider("Opacidad de Taylor", 0.1, 1.0, 0.70, 0.05)

    return UIParameters(
        func_str=func_str,
        x0=x0,
        y0=y0,
        degree=degree,
        x_range=x_range,
        y_range=y_range,
        tolerance=tolerance,
        show_real=show_real,
        show_taylor=show_taylor,
        show_anchor=show_anchor,
        show_tolerance_mask=show_tolerance_mask,
        real_opacity=real_opacity,
        taylor_opacity=taylor_opacity,
        show_error_surface=show_error_surface,
    )
