"""Visualizador Interactivo del Polinomio de Taylor en Dos Variables.

Herramienta educativa para la materia de Cálculo Multivariado que conecta
la formulación algebraica analítica con la intuición geométrica tridimensional
e integra un Tutor Inteligente con IA (Gemini / DeepSeek).

La interfaz está organizada como un recorrido de aprendizaje:
  Empieza aquí (idea en 1D) → Visor 3D → ¿Dónde funciona? → Las matemáticas → Tutor IA
"""

import streamlit as st
import numpy as np
import sympy as sp

from core.taylor_math import (
    MathParserError,
    NumericEvaluationResult,
    PointComparisonResult,
    TaylorExpansionResult,
    compare_at_point,
    compute_taylor_polynomial,
    evaluate_surfaces_and_error,
    parse_function_2d,
)
from core.ai_tutor import explain_current_state_with_ai
from ui.components import render_sidebar_controls
from ui.explanations import EXPERIMENTS, describe_current_state
from ui.intro_1d import render_intro_tab
from ui.tabs import TAB_3D, TAB_AI, TAB_ERROR, TAB_INTRO, TAB_MATH
from ui.plotting_2d import build_error_heatmap_figure
from ui.plotting_3d import build_taylor_3d_figure, build_error_surface_figure
from ui.latex_display import render_latex_mathematical_breakdown, taylor_polynomial_latex

# Configuración principal de la aplicación Streamlit
st.set_page_config(
    page_title="Visualizador 3D: Polinomio de Taylor Multivariable",
    page_icon=":material/functions:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Caché inteligente: el cálculo algebraico y numérico solo se repite cuando cambian
# sus parámetros matemáticos. Mover controles puramente visuales (transparencia de P_n)
# reutiliza los resultados almacenados.
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=64)
def cached_taylor_expansion(func_str: str, x0: float, y0: float, degree: int) -> TaylorExpansionResult:
    """Parseo seguro + polinomio de Taylor simbólico (la etapa más costosa)."""
    func_expr, x_sym, y_sym = parse_function_2d(func_str)
    return compute_taylor_polynomial(func_expr, x_sym, y_sym, x0=x0, y0=y0, degree=degree)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_surface_evaluation(
    func_str: str,
    x0: float,
    y0: float,
    degree: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    tolerance: float,
    num_points: int,
) -> NumericEvaluationResult:
    """Malla NumPy, recorte por percentiles y residuo |f - P_n|."""
    taylor = cached_taylor_expansion(func_str, x0, y0, degree)
    return evaluate_surfaces_and_error(
        func_expr=taylor.func_expr,
        poly_expr=taylor.polynomial_symbolic,
        x_sym=taylor.x_sym,
        y_sym=taylor.y_sym,
        x_range=tuple(x_range),
        y_range=tuple(y_range),
        x0=x0,
        y0=y0,
        tolerance=tolerance,
        num_points=num_points,
    )


@st.cache_data(show_spinner=False, max_entries=256)
def cached_point_comparison(
    func_str: str, x0: float, y0: float, degree: int, x1: float, y1: float
) -> PointComparisonResult:
    """Comparador puntual f(x1, y1) vs P_n(x1, y1)."""
    taylor = cached_taylor_expansion(func_str, x0, y0, degree)
    return compare_at_point(
        taylor.func_expr, taylor.polynomial_symbolic, taylor.x_sym, taylor.y_sym, x1, y1
    )


def _fmt(value: float, spec: str = ".6g") -> str:
    return format(value, spec) if np.isfinite(value) else "N/A"


# Encabezado principal
st.title("Visualizador del Polinomio de Taylor en 2 Variables", icon=":material/calculate:")
st.markdown(
    "Una función complicada se puede **imitar con un polinomio sencillo** cerca de un punto. "
    "Esta app te deja **ver** esa imitación en 3D y **medir dónde funciona y dónde falla**. "
    "Si es tu primera vez, empieza por la pestaña **Empieza aquí** o por los "
    "**Experimentos guiados** de la barra lateral."
)

# 1. Controles de la barra lateral
ui_params = render_sidebar_controls()

tab_intro, tab_3d, tab_err, tab_math, tab_ai = st.tabs(
    [TAB_INTRO, TAB_3D, TAB_ERROR, TAB_MATH, TAB_AI],
    key="main_tab",
    on_change="rerun",
)

with tab_intro:
    render_intro_tab()

# 2. Pipeline de cálculo simbólico y numérico con manejo robusto de excepciones
try:
    # A + B. Parseo seguro y motor simbólico (derivadas parciales y polinomio de Taylor)
    with st.spinner("Calculando derivadas parciales y polinomio de Taylor..."):
        taylor_res = cached_taylor_expansion(
            ui_params.func_str, ui_params.x0, ui_params.y0, ui_params.degree
        )

    # C. Motor numérico: mallas, discretización, recorte por percentiles y residuo |f - P_n|
    eval_res = cached_surface_evaluation(
        ui_params.func_str,
        ui_params.x0,
        ui_params.y0,
        ui_params.degree,
        tuple(ui_params.x_range),
        tuple(ui_params.y_range),
        ui_params.tolerance,
        ui_params.resolution,
    )

    # Valor numérico en el punto central para el marcador
    try:
        z0_val = float(taylor_res.f_at_point)
    except Exception:
        z0_val = np.nan

    # ---------------------------------------------------------------------
    # Pestaña: Visor 3D
    # ---------------------------------------------------------------------
    with tab_3d:
        active = st.session_state.get("active_experiment")
        exp = next((e for e in EXPERIMENTS if e.title == active), None)
        if exp is not None:
            st.info(f"**Experimento {exp.title}** — {exp.what_to_look}", icon=":material/science:")

        col_vis, col_side = st.columns([1.35, 0.65], gap="medium")

        with col_side:
            st.markdown("##### Punto de prueba")
            st.caption("Elige un lugar cualquiera para comparar la función real con la copia (línea roja).")
            cin1, cin2 = st.columns(2)
            with cin1:
                x1 = float(st.number_input("x₁", value=ui_params.x0 + 0.5, step=0.1, format="%.3f", key="cmp_x1"))
            with cin2:
                y1 = float(st.number_input("y₁", value=ui_params.y0 + 0.5, step=0.1, format="%.3f", key="cmp_y1"))

            point_cmp = cached_point_comparison(
                ui_params.func_str, ui_params.x0, ui_params.y0, ui_params.degree, x1, y1
            )

            st.markdown("##### ¿Qué estás viendo?")
            with st.container(border=True):
                st.markdown(
                    describe_current_state(
                        degree=ui_params.degree,
                        x0=ui_params.x0,
                        y0=ui_params.y0,
                        f0=z0_val,
                        radius=ui_params.radius,
                        tolerance=ui_params.tolerance,
                        valid_pct=eval_res.valid_percentage,
                        max_error=eval_res.max_error,
                        point_x1=x1,
                        point_y1=y1,
                        point_f=point_cmp.f_exact,
                        point_p=point_cmp.p_approx,
                        point_err=point_cmp.abs_error,
                    )
                )
            if not np.isfinite(point_cmp.f_exact):
                st.warning(f"La función no está definida en ({x1}, {y1}).", icon=":material/warning:")

        with col_vis:
            fig_3d = build_taylor_3d_figure(
                eval_res=eval_res,
                x0=ui_params.x0,
                y0=ui_params.y0,
                z0=z0_val,
                func_str=ui_params.func_str,
                degree=ui_params.degree,
                taylor_opacity=ui_params.taylor_opacity,
                point_cmp=point_cmp,
            )
            st.plotly_chart(fig_3d, width="stretch")
            st.caption(
                "🖱️ Arrastra para girar, usa la rueda para hacer zoom y pasa el cursor para leer valores. "
                "Clic en un elemento de la leyenda para ocultarlo."
            )

        with st.expander("Números exactos del punto de prueba", icon=":material/pin:"):
            cmp_m1, cmp_m2, cmp_m3, cmp_m4 = st.columns(4)
            with cmp_m1:
                st.metric("Función real f(x₁, y₁)", _fmt(point_cmp.f_exact))
            with cmp_m2:
                st.metric(f"Copia P{ui_params.degree}(x₁, y₁)", _fmt(point_cmp.p_approx))
            with cmp_m3:
                has_err = bool(np.isfinite(point_cmp.abs_error))
                within = has_err and point_cmp.abs_error <= ui_params.tolerance
                st.metric(
                    "Error absoluto E",
                    _fmt(point_cmp.abs_error, ".3e"),
                    delta=("aceptable (≤ ε)" if within else "excesivo (> ε)") if has_err else None,
                    delta_color="normal" if within else "inverse",
                    help="E = |f(x₁, y₁) − Pₙ(x₁, y₁)|: cuánto se equivoca la copia en ese punto.",
                )
            with cmp_m4:
                st.metric(
                    "Error relativo",
                    f"{_fmt(point_cmp.rel_error_pct, '.4g')} %" if np.isfinite(point_cmp.rel_error_pct) else "N/A",
                    help="El error como porcentaje del valor real. No está definido cuando f(x₁, y₁) = 0.",
                )

    # ---------------------------------------------------------------------
    # Pestaña: ¿Dónde funciona?
    # ---------------------------------------------------------------------
    with tab_err:
        st.markdown(
            f"La copia nunca es perfecta en todas partes. Aquí ves **dónde se equivoca menos de "
            f"{ui_params.tolerance:g}** (el error que aceptaste en la barra lateral)."
        )
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                "Zona donde sirve",
                f"{eval_res.valid_percentage:.1f} %",
                help="Porcentaje del área mostrada donde el error es menor o igual que ε.",
                border=True,
            )
        with m2:
            st.metric(
                "Radio de confianza",
                f"≈ {eval_res.valid_radius_est:.2f}",
                help="Hasta qué distancia del punto de apoyo la copia sigue sirviendo (aproximado).",
                border=True,
            )
        with m3:
            st.metric(
                "Peor error",
                _fmt(eval_res.max_error, ".4g"),
                help="El error más grande dentro de la ventana mostrada.",
                border=True,
            )
        with m4:
            st.metric(
                "Error promedio (RMSE)",
                _fmt(eval_res.rmse, ".4g"),
                help=f"Error cuadrático medio en la ventana. Error absoluto medio (MAE): {_fmt(eval_res.mae, '.4g')}.",
                border=True,
            )

        col_heat, col_surf = st.columns(2, gap="medium")
        with col_heat:
            st.markdown("#### Mapa del error (vista desde arriba)")
            st.caption(
                "🟩 **Verde** = la copia sirve · 🟥 **Rojo** = se equivoca demasiado · "
                "La **línea negra** es la frontera. El rombo es el punto de apoyo y la ✕ el punto de prueba."
            )
            fig_heat = build_error_heatmap_figure(
                eval_res=eval_res,
                x0=ui_params.x0,
                y0=ui_params.y0,
                tolerance=ui_params.tolerance,
                point_cmp=point_cmp,
            )
            st.plotly_chart(fig_heat, width="stretch")
        with col_surf:
            st.markdown("#### El error como montaña")
            st.caption(
                "La altura es cuánto se equivoca la copia. El **techo gris** es el error aceptable ε: "
                "donde la montaña lo atraviesa, la copia deja de servir."
            )
            fig_err = build_error_surface_figure(
                eval_res=eval_res,
                x0=ui_params.x0,
                y0=ui_params.y0,
                tolerance=ui_params.tolerance,
            )
            st.plotly_chart(fig_err, width="stretch")

        st.info(
            "**Patrón a notar:** el error es casi cero junto al punto de apoyo y crece al alejarse. "
            "Subir el grado suele agrandar la zona verde.",
            icon=":material/lightbulb:",
        )

    # ---------------------------------------------------------------------
    # Pestaña: Las matemáticas
    # ---------------------------------------------------------------------
    with tab_math:
        render_latex_mathematical_breakdown(taylor_res)

    # ---------------------------------------------------------------------
    # Pestaña: Tutor IA
    # ---------------------------------------------------------------------
    with tab_ai:
        st.subheader("Tutor Inteligente de Cálculo Multivariado", icon=":material/smart_toy:")
        st.caption(
            "Pide una explicación de la configuración actual o haz cualquier pregunta. "
            "Sin clave de API se genera un diagnóstico automático."
        )

        btn_explain = st.button("Explícame lo que estoy viendo", icon=":material/auto_awesome:")
        user_question = st.text_input(
            "O hazle una pregunta al tutor:",
            placeholder="Ej: ¿Por qué la copia falla más rápido en la dirección Y?",
        )
        btn_ask = st.button("Enviar pregunta", icon=":material/send:") if user_question else False

        fx_str = sp.latex(taylor_res.gradient[0])
        fy_str = sp.latex(taylor_res.gradient[1])
        det_h_str = None
        if taylor_res.hessian_matrix is not None:
            H = taylor_res.hessian_matrix
            det_h_str = sp.latex(sp.simplify(H[0][0] * H[1][1] - H[0][1] * H[1][0]))

        if btn_explain or btn_ask:
            with st.spinner("El Tutor IA está analizando la superficie y el polinomio..."):
                try:
                    explanation = explain_current_state_with_ai(
                        func_str=ui_params.func_str,
                        x0=ui_params.x0,
                        y0=ui_params.y0,
                        f0=z0_val,
                        degree=ui_params.degree,
                        poly_str=taylor_polynomial_latex(taylor_res),
                        gradient_str=(fx_str, fy_str),
                        hessian_det_str=det_h_str,
                        valid_percentage=eval_res.valid_percentage,
                        rmse=eval_res.rmse,
                        max_error=eval_res.max_error,
                        valid_radius=eval_res.valid_radius_est,
                        tolerance=ui_params.tolerance,
                        custom_question=user_question if btn_ask else None,
                    )
                    st.markdown(explanation)
                except Exception as e_ai:
                    st.error(f"Error al consultar el Tutor IA: {e_ai}", icon=":material/error:")

except MathParserError as mpe:
    st.error(f"**No pude entender la fórmula:** {mpe}", icon=":material/error:")
    st.info(
        "**Cómo escribirla:** usa `sin(x)`, `cos(y)`, `exp(x)`, `log(x)`, `sqrt(x)`; `^` para potencias "
        "(`x^2`) y `*` para multiplicar (`2*x*y`). Ejemplo: `exp(-(x^2+y^2))`.",
        icon=":material/lightbulb:",
    )

except ValueError as ve:
    st.warning(f"**Atención:** {ve}", icon=":material/warning:")

except Exception as ex:
    st.error(f"**Error inesperado en el cálculo:** {ex}", icon=":material/error:")
