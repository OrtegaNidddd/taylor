"""Pestaña introductoria: la idea de Taylor explicada con una curva (una sola variable)."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

_x = sp.Symbol("x")

CURVES_1D = {
    "sen(x) — una ola": sp.sin(_x),
    "eˣ — crecimiento exponencial": sp.exp(_x),
    "cos(x) — otra ola": sp.cos(_x),
    "1 / (1 + x²) — una colina": 1 / (1 + _x**2),
}


@st.cache_data(show_spinner=False)
def _taylor_1d(name: str, a: float, n: int) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    f = CURVES_1D[name]
    a_s = sp.Rational(str(a))
    poly = sum(sp.diff(f, _x, k).subs(_x, a_s) / sp.factorial(k) * (_x - a_s) ** k for k in range(n + 1))
    xs = np.linspace(a - 6, a + 6, 600)
    f_np = sp.lambdify(_x, f, "numpy")
    p_np = sp.lambdify(_x, poly, "numpy")
    ys_f = np.broadcast_to(f_np(xs), xs.shape).astype(float)
    ys_p = np.broadcast_to(p_np(xs), xs.shape).astype(float)
    return sp.latex(sp.expand(poly)), xs, ys_f, ys_p


def render_intro_tab() -> None:
    st.markdown(
        """
        ### ¿Qué hace esta aplicación?

        Muchas funciones matemáticas (senos, exponenciales, logaritmos…) son **difíciles de calcular**.
        Los polinomios (expresiones con solo sumas y potencias, como $3 + 2x - x^2$) son **muy fáciles**.

        El **polinomio de Taylor** es una forma de construir un polinomio que **imita** a una función difícil
        **cerca de un punto** que tú eliges. Es como describir una montaña a alguien por teléfono:
        """
    )
    c1, c2, c3 = st.columns(3)
    c1.info("**Grado 0:** \"estoy a 1 200 m de altura\". Solo la altura.", icon=":material/horizontal_rule:")
    c2.info("**Grado 1:** \"…y el terreno sube hacia el norte\". Altura + inclinación.", icon=":material/trending_up:")
    c3.info("**Grado 2:** \"…y estoy en un valle\". Altura + inclinación + curvatura.", icon=":material/landscape:")
    st.markdown(
        "Cada dato extra hace la descripción más precisa, pero **siempre es mejor cerca de donde estás** "
        "que lejos. Esa es toda la idea."
    )

    st.markdown("---")
    st.markdown("### Pruébalo primero con una curva")
    st.caption("Mueve el grado y mira cómo la línea naranja (la copia) se va pegando a la azul (la función real).")

    col_ctrl, col_plot = st.columns([0.32, 0.68], gap="medium")
    with col_ctrl:
        name = st.selectbox("Curva", list(CURVES_1D.keys()), key="intro_curve")
        n = st.slider("Grado de la copia", 0, 10, 1, key="intro_degree")
        a = st.slider("Punto de apoyo", -3.0, 3.0, 0.0, 0.5, key="intro_center")
        poly_ltx, xs, ys_f, ys_p = _taylor_1d(name, a, n)
        st.markdown("**La copia es:**")
        st.latex(rf"P_{{{n}}}(x) = {poly_ltx}")

    with col_plot:
        finite = ys_f[np.isfinite(ys_f)]
        lo, hi = float(finite.min()), float(finite.max())
        pad = max(1.0, 0.4 * (hi - lo))
        fa = float(sp.N(CURVES_1D[name].subs(_x, a)))

        good = np.abs(ys_f - ys_p) <= 0.1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=np.where(good, ys_f, np.nan), mode="lines",
                                 line=dict(color="rgba(22,163,74,0.25)", width=18),
                                 name="Zona donde la copia sirve", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=ys_f, mode="lines", line=dict(color="#2563EB", width=3),
                                 name="Función real"))
        fig.add_trace(go.Scatter(x=xs, y=ys_p, mode="lines", line=dict(color="#F97316", width=3, dash="dash"),
                                 name=f"Copia de grado {n}"))
        fig.add_trace(go.Scatter(x=[a], y=[fa], mode="markers",
                                 marker=dict(size=14, color="#FACC15", symbol="diamond",
                                             line=dict(color="#000", width=2)),
                                 name="Punto de apoyo"))
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[lo - pad, hi + pad], title="altura"),
            xaxis=dict(title="x"),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        )
        st.plotly_chart(fig, width="stretch")

    st.success(
        "**Lo que acabas de ver es lo mismo que hace el resto de la app, pero en 3D:** "
        "en vez de una curva hay una **superficie** (un terreno con dos direcciones, x e y) "
        "y la copia es también una superficie. Pasa a la pestaña **Visor 3D** o usa los "
        "**Experimentos guiados** de la barra lateral.",
        icon=":material/arrow_forward:",
    )
