"""Controles de la barra lateral, catálogo de funciones y experimentos guiados (Streamlit)."""

from dataclasses import dataclass
import streamlit as st
from core.taylor_math import MIN_DEGREE, MAX_DEGREE
from ui.explanations import EXPERIMENTS, GLOSSARY, Experiment
from ui.tabs import TAB_3D

# Nombre amigable -> fórmula
PRESETS = {
    "Olas (seno × coseno)": "sin(x) * cos(y)",
    "Tazón (paraboloide)": "x^2 + y^2",
    "Silla de montar": "x^2 - y^2",
    "Colina suave (campana)": "exp(-(x^2 + y^2))",
    "Canal inclinado": "x^2 + 2*y",
    "Silla de mono (3 valles)": "x^3 - 3*x*y^2",
    "Embudo logarítmico": "log(1 + x^2 + y^2)",
    "Escribir mi propia función": "",
}
CUSTOM_PRESET = "Escribir mi propia función"

# Valores iniciales de los widgets (claves de st.session_state)
DEFAULTS = {
    "preset": "Olas (seno × coseno)",
    "func_str": PRESETS["Olas (seno × coseno)"],
    "degree": 1,
    "x0": 0.0,
    "y0": 0.0,
    "radius": 3.0,
    "resolution": 40,
    "tolerance": 0.20,
    "taylor_opacity": 0.70,
}


@dataclass
class UIParameters:
    """Parámetros seleccionados por el usuario en la interfaz."""
    func_str: str
    x0: float
    y0: float
    degree: int
    radius: float        # Radio de visualización (vecindad) alrededor de (x0, y0)
    resolution: int      # Puntos por eje de la malla
    tolerance: float
    taylor_opacity: float

    @property
    def x_range(self) -> tuple[float, float]:
        return (self.x0 - self.radius, self.x0 + self.radius)

    @property
    def y_range(self) -> tuple[float, float]:
        return (self.y0 - self.radius, self.y0 + self.radius)


def init_session_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def _on_preset_change() -> None:
    formula = PRESETS[st.session_state["preset"]]
    if formula:
        st.session_state["func_str"] = formula


def load_experiment(exp: Experiment) -> None:
    """Carga un experimento guiado en los controles (se usa como callback de botón)."""
    st.session_state.update(
        preset=exp.preset,
        func_str=PRESETS[exp.preset],
        degree=exp.degree,
        x0=exp.x0,
        y0=exp.y0,
        radius=exp.radius,
        active_experiment=exp.title,
        main_tab=TAB_3D,
    )


def reset_controls() -> None:
    st.session_state.update(DEFAULTS)
    st.session_state.pop("active_experiment", None)


def render_sidebar_controls() -> UIParameters:
    """Renderiza los controles de la barra lateral y retorna los parámetros configurados."""
    init_session_state()
    sb = st.sidebar

    # 0. Experimentos guiados
    sb.header("Experimentos guiados", icon=":material/school:")
    sb.caption("¿No sabes por dónde empezar? Haz clic en orden y observa el visor 3D.")
    for exp in EXPERIMENTS:
        sb.button(exp.title, on_click=load_experiment, args=(exp,), width="stretch", key=f"exp_{exp.title}")

    sb.markdown("---")
    sb.header("Controles", icon=":material/tune:")

    # 1. Función
    sb.selectbox(
        "1 · Elige una superficie",
        options=list(PRESETS.keys()),
        key="preset",
        on_change=_on_preset_change,
        help="Cada opción es una 'forma de terreno' distinta. La fórmula aparece en el campo de abajo.",
    )
    func_str = sb.text_input(
        "Fórmula f(x, y)",
        key="func_str",
        help=(
            "Puedes editarla. Ejemplos: sin(x)*cos(y), x^2 + y^2, exp(-(x^2+y^2)). "
            "Usa ^ para potencias y * para multiplicar."
        ),
    )

    # 2. Grado
    degree = int(
        sb.slider(
            "2 · Grado de la copia (n)",
            min_value=MIN_DEGREE,
            max_value=MAX_DEGREE,
            key="degree",
            help=(
                "Qué tanto detalle copia el polinomio. 0 = solo la altura (un piso plano), "
                "1 = también la inclinación (plano tangente), 2 = también la curvatura, 3-6 = detalles más finos."
            ),
        )
    )
    sb.caption(
        {
            0: "📏 Grado 0: un piso plano.",
            1: "📐 Grado 1: plano inclinado (tangente).",
            2: "🥣 Grado 2: copia la curvatura.",
        }.get(degree, f"🔬 Grado {degree}: detalles finos.")
    )

    # 3. Punto de apoyo
    sb.subheader("3 · Punto de apoyo (x₀, y₀)", icon=":material/my_location:")
    sb.caption("Dónde se 'pega' la copia a la superficie (el rombo amarillo).")
    x0 = round(float(sb.slider("x₀", min_value=-5.0, max_value=5.0, step=0.1, key="x0")), 4)
    y0 = round(float(sb.slider("y₀", min_value=-5.0, max_value=5.0, step=0.1, key="y0")), 4)

    # 4. Tolerancia
    sb.subheader("4 · Error aceptable (ε)", icon=":material/rule:")
    tolerance = float(
        sb.number_input(
            "¿Cuánto error aceptas?",
            min_value=0.001,
            max_value=5.0,
            step=0.05,
            format="%.3f",
            key="tolerance",
            help="Si la copia se equivoca menos que este valor en un lugar, decimos que 'sirve' ahí.",
        )
    )

    # 5. Opciones de visualización (avanzadas)
    with sb.expander("Opciones de visualización", icon=":material/visibility:"):
        radius = float(
            st.slider(
                "Tamaño de la ventana (radio r)",
                min_value=0.5,
                max_value=10.0,
                step=0.5,
                key="radius",
                help="Se dibuja el cuadrado [x₀ − r, x₀ + r] × [y₀ − r, y₀ + r].",
            )
        )
        resolution = int(
            st.slider(
                "Resolución (puntos por eje)",
                min_value=20,
                max_value=100,
                step=10,
                key="resolution",
                help="Más puntos = gráficas más suaves pero más lentas.",
            )
        )
        taylor_opacity = float(
            st.slider("Opacidad de la copia naranja", 0.1, 1.0, step=0.05, key="taylor_opacity")
        )

    sb.button("Restablecer todo", icon=":material/restart_alt:", on_click=reset_controls, width="stretch")

    # 6. Glosario
    sb.markdown("---")
    with sb.expander("Glosario: ¿qué significa…?", icon=":material/menu_book:"):
        for term, meaning in GLOSSARY.items():
            st.markdown(f"**{term}** — {meaning}")

    return UIParameters(
        func_str=func_str,
        x0=x0,
        y0=y0,
        degree=degree,
        radius=radius,
        resolution=resolution,
        tolerance=tolerance,
        taylor_opacity=taylor_opacity,
    )
