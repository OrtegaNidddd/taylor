"""Visualizador Interactivo del Polinomio de Taylor en Dos Variables.

Herramienta educativa para la materia de Cálculo Multivariado que conecta
la formulación algebraica analítica con la intuición geométrica tridimensional
e integra un Tutor Inteligente con IA (Gemini / DeepSeek).
"""

import streamlit as st
import numpy as np
import sympy as sp

from core.parser import parse_function_2d, MathParserError
from core.taylor_engine import compute_taylor_polynomial
from core.numeric_eval import evaluate_surfaces_and_error
from core.ai_tutor import explain_current_state_with_ai
from ui.components import render_sidebar_controls
from ui.plotting_3d import build_taylor_3d_figure, build_error_surface_figure
from ui.latex_display import render_latex_mathematical_breakdown

# Configuración principal de la aplicación Streamlit
st.set_page_config(
    page_title="Visualizador 3D: Polinomio de Taylor Multivariable",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Encabezado principal y descripción didáctica
st.title("📐 Visualizador Interactivo del Polinomio de Taylor en 2 Variables")
st.markdown(
    """
    Esta herramienta explora el comportamiento del **Polinomio de Taylor multivariable** alrededor de un punto de apoyo $(x_0, y_0)$.
    Permite contrastar analíticamente el desarrollo algebraico (derivadas parciales, plano tangente y matriz Hessiana)
    con la geometría tridimensional y la **región de validez del residuo** en tiempo real.
    """
)

# 1. Renderizar controles de la barra lateral
ui_params = render_sidebar_controls()

# 2. Pipeline de cálculo simbólico y numérico con manejo robusto de excepciones
try:
    # A. Parseo seguro de la expresión matemática
    func_expr, x_sym, y_sym = parse_function_2d(ui_params.func_str)

    # B. Motor simbólico: derivadas parciales y polinomio de Taylor
    taylor_res = compute_taylor_polynomial(
        func_expr=func_expr,
        x_sym=x_sym,
        y_sym=y_sym,
        x0=ui_params.x0,
        y0=ui_params.y0,
        degree=ui_params.degree,
    )

    # C. Motor numérico: mallas, discretización y residuo |f - P_n|
    eval_res = evaluate_surfaces_and_error(
        func_expr=func_expr,
        poly_expr=taylor_res.polynomial_symbolic,
        x_sym=x_sym,
        y_sym=y_sym,
        x_range=ui_params.x_range,
        y_range=ui_params.y_range,
        x0=ui_params.x0,
        y0=ui_params.y0,
        tolerance=ui_params.tolerance,
        num_points=40,
    )

    # Valor numérico en el punto central para el marcador
    try:
        z0_val = float(taylor_res.f_at_point)
    except Exception:
        z0_val = np.nan

    # 3. Métricas clave en tarjetas superiores
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Punto (x₀, y₀)", f"({ui_params.x0:.2f}, {ui_params.y0:.2f})")
    with m2:
        val_str = f"{z0_val:.4f}" if np.isfinite(z0_val) else "N/A"
        st.metric("f(x₀, y₀)", val_str)
    with m3:
        st.metric("Grado n", f"{ui_params.degree}")
    with m4:
        st.metric(
            f"Zona Válida (|Error| ≤ {ui_params.tolerance})",
            f"{eval_res.valid_percentage:.1f} %",
            help="Porcentaje del área de la ventana donde la aproximación cumple con la tolerancia fijada.",
        )
    with m5:
        rmse_str = f"{eval_res.rmse:.4f}" if np.isfinite(eval_res.rmse) else "N/A"
        st.metric("RMSE en Ventana", rmse_str, help="Error cuadrático medio global en la ventana evaluada.")

    # 4. Distribución en columnas principales (Visor 3D y Panel Analítico)
    col_vis, col_math = st.columns([1.2, 0.8], gap="medium")

    with col_vis:
        st.subheader("🌐 Visor Espacial 3D Interactivo")

        # Construcción de la figura tridimensional principal
        fig_3d = build_taylor_3d_figure(
            eval_res=eval_res,
            x0=ui_params.x0,
            y0=ui_params.y0,
            z0=z0_val,
            func_str=ui_params.func_str,
            degree=ui_params.degree,
            show_real=ui_params.show_real,
            show_taylor=ui_params.show_taylor,
            show_anchor=ui_params.show_anchor,
            show_tolerance_mask=ui_params.show_tolerance_mask,
            real_opacity=ui_params.real_opacity,
            taylor_opacity=ui_params.taylor_opacity,
        )
        st.plotly_chart(fig_3d, width="stretch")

        # Superficie de error dedicada (opcional según toggle)
        if ui_params.show_error_surface:
            st.markdown("#### Superficie del Residuo $|f(x, y) - P_n(x, y)|$")
            fig_err = build_error_surface_figure(
                eval_res=eval_res,
                x0=ui_params.x0,
                y0=ui_params.y0,
                tolerance=ui_params.tolerance,
            )
            st.plotly_chart(fig_err, width="stretch")

    with col_math:
        # Renderizado de ecuaciones formales y desglose pedagógico
        render_latex_mathematical_breakdown(taylor_res)

        # Diagnóstico del residuo
        st.subheader("Diagnóstico del Residuo (Error)")
        st.markdown(
            fr"""
            - **Tolerancia configurada ($\epsilon$):** `{ui_params.tolerance}`
            - **Error máximo en la ventana:** `{eval_res.max_error:.4f}`
            - **Error absoluto medio (MAE):** `{eval_res.mae:.4f}`
            - **Radio estimado de validez centrado en $(x_0, y_0)$:** `~ {eval_res.valid_radius_est:.2f}` unidades.
            """
        )

    # 5. Sección de Tutor Inteligente con IA (Gemini / DeepSeek)
    st.markdown("---")
    st.subheader("🤖 Tutor Inteligente de Cálculo Multivariado")
    st.caption("Recibe explicaciones pedagógicas generadas por IA sobre la aproximación y el residuo actual.")

    col_tutor_btn, col_tutor_input = st.columns([0.35, 0.65])
    with col_tutor_btn:
        btn_explain = st.button("✨ Explicar Geometría y Aproximación con IA", width="stretch")
    with col_tutor_input:
        user_question = st.text_input(
            "O hazle una pregunta específica al tutor:",
            placeholder="Ej: ¿Por qué la aproximación diverge más rápido en la dirección Y?",
            label_visibility="collapsed",
        )
        btn_ask = st.button("Enviar Pregunta al Tutor", width="stretch") if user_question else False

    # Preparar datos formateados para el tutor
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
                    poly_str=sp.latex(taylor_res.polynomial_simplified),
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
                st.error(f"Error al consultar el Tutor IA: {e_ai}")

    # 6. Guía Pedagógica y Preguntas de Reflexión para el Estudiante
    st.markdown("---")
    with st.expander("📚 Guía Pedagógica y Preguntas de Reflexión para el Estudiante"):
        st.markdown(
            r"""
            ### ¿Cómo usar este simulador para aprender Cálculo Multivariado?
            1. **Grado $n=1$ (Plano Tangente):** Observa cómo para $n=1$, la aproximación es un plano que roza la superficie en $(x_0, y_0)$. Nota cómo a medida que te alejas del punto, la curvatura de la función original se separa rápidamente del plano.
            2. **Grado $n=2$ (Aproximación Cuadrática):** Al subir a grado 2, la aproximación se convierte en un paraboloide (elíptico o hiperbólico). Observa cómo "copia" la curvatura de la superficie original alrededor del punto.
            3. **La Zona Verde de Tolerancia:** Los puntos verdes representan las coordenadas $(x, y)$ donde la diferencia entre la función y el polinomio es menor o igual a $\epsilon$.
               - ¿Qué ocurre con la zona verde cuando aumentas el grado $n$? *(R: El área válida se expande).*
               - ¿Qué ocurre si reduces la tolerancia $\epsilon$? *(R: La zona verde se contrae hacia el punto de anclaje).*
            4. **Puntos Singulares o Discontinuos:** Prueba funciones como $1/(x^2 + y^2)$ y observa qué sucede si colocas $(x_0, y_0) = (0, 0)$ frente a un punto regular como $(1, 1)$.
            """
        )

except MathParserError as mpe:
    st.error(f"⚠️ **Error en la entrada matemática:** {mpe}")
    st.info("💡 **Consejo de sintaxis:** Usa notación estándar: `sin(x)`, `cos(y)`, `x^2`, `exp(-(x^2+y^2))`, `2*x*y`.")

except ValueError as ve:
    st.warning(f"⚠️ **Atención:** {ve}")

except Exception as ex:
    st.error(f"❌ **Error inesperado en el cálculo:** {ex}")
