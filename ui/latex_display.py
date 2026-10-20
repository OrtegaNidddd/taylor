"""Módulo para el formateo pedagógico y renderizado de fórmulas matemáticas en LaTeX."""

import streamlit as st
import sympy as sp
from core.taylor_engine import TaylorExpansionResult


def render_latex_mathematical_breakdown(res: TaylorExpansionResult) -> None:
    """
    Renderiza de forma clara y pedagógica el desarrollo algebraico del polinomio de Taylor.
    """
    st.subheader("Desarrollo Simbólico y Analítico")

    # 1. Función Original y Punto de Expansión
    st.markdown(f"**Función objetivo:** $f(x, y) = {sp.latex(res.func_expr)}$")
    st.markdown(f"**Punto de expansión:** $(x_0, y_0) = ({res.x0}, {res.y0})$")
    st.markdown(f"**Valor en el punto:** $f({res.x0}, {res.y0}) = {sp.latex(res.f_at_point)}$")

    # 2. Polinomio de Taylor Resultante
    st.markdown(f"#### Polinomio de Taylor $P_{{{res.degree}}}(x, y)$")
    st.latex(f"P_{{{res.degree}}}(x, y) = {sp.latex(res.polynomial_simplified)}")

    # 3. Interpretación de Grado 1: Plano Tangente
    with st.expander("Interpretación Geométrica: Plano Tangente ($n = 1$)", expanded=(res.degree == 1)):
        fx, fy = res.gradient
        st.markdown(
            "El polinomio de primer grado coincide exactamente con el **plano tangente** a la superficie en el punto:"
        )
        st.latex(
            rf"\nabla f({res.x0}, {res.y0}) = \begin{{pmatrix}} f_x \\ f_y \end{{pmatrix}} = "
            rf"\begin{{pmatrix}} {sp.latex(fx)} \\ {sp.latex(fy)} \end{{pmatrix}}"
        )
        st.latex(
            rf"z = f(x_0, y_0) + f_x(x_0, y_0)(x - x_0) + f_y(x_0, y_0)(y - y_0)"
        )
        st.latex(rf"z = {sp.latex(res.tangent_plane)}")

    # 4. Interpretación de Grado 2: Matriz Hessiana y Curvatura
    if res.degree >= 2 and res.hessian_matrix is not None:
        with st.expander("Interpretación de Grado 2: Matriz Hessiana y Términos Cuadráticos", expanded=(res.degree == 2)):
            H = res.hessian_matrix
            fxx, fxy = H[0][0], H[0][1]
            fyx, fyy = H[1][0], H[1][1]
            det_H = sp.simplify(fxx * fyy - fxy * fyx)

            st.markdown("La aproximación cuadrática incorpora la curvatura mediante la **Matriz Hessiana**:")
            st.latex(
                rf"H({res.x0}, {res.y0}) = \begin{{pmatrix}} f_{{xx}} & f_{{xy}} \\ f_{{yx}} & f_{{yy}} \end{{pmatrix}} = "
                rf"\begin{{pmatrix}} {sp.latex(fxx)} & {sp.latex(fxy)} \\ {sp.latex(fyx)} & {sp.latex(fyy)} \end{{pmatrix}}"
            )
            st.markdown(f"**Determinante Hessiano:** $\\det(H) = {sp.latex(det_H)}$")
            if det_H.is_number:
                if det_H > 0 and fxx > 0:
                    st.info("Punto de curvatura convexa local (mínimo local potencial).")
                elif det_H > 0 and fxx < 0:
                    st.info("Punto de curvatura cóncava local (máximo local potencial).")
                elif det_H < 0:
                    st.info("Punto de ensilladura (curvaturas opuestas según la dirección).")

    # 5. Desglose detallado por órdenes
    with st.expander("Desglose término a término por órdenes de derivación"):
        for order, term_list in res.terms_by_order.items():
            st.markdown(f"**Términos de Orden {order}:**")
            cols = st.columns(min(len(term_list), 3) or 1)
            for idx, item in enumerate(term_list):
                with cols[idx % len(cols)]:
                    i, j = item["dx_power"], item["dy_power"]
                    d_val = item["derivative_value"]
                    coeff = item["coefficient"]
                    st.latex(
                        rf"\frac{{\partial^{{{order}}} f}}{{\partial x^{{{i}}} \partial y^{{{j}}}}} = {sp.latex(d_val)}"
                    )
                    st.caption(f"Coeficiente: {sp.latex(coeff)}")
