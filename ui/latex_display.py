"""Módulo para el formateo pedagógico y renderizado de fórmulas matemáticas en LaTeX."""

import streamlit as st
import sympy as sp
from core.taylor_math import TaylorExpansionResult, to_exact_number


GENERAL_TAYLOR_FORMULA = (
    r"P_n(x, y) = \sum_{k=0}^{n} \sum_{j=0}^{k} \frac{1}{j!\,(k-j)!} \cdot "
    r"\frac{\partial^k f(x_0, y_0)}{\partial x^{j}\, \partial y^{k-j}} (x - x_0)^{j} (y - y_0)^{k-j}"
)


def _power_factor_latex(sym: sp.Symbol, center: sp.Expr, power: int) -> str:
    """LaTeX de (sym - center)^power conservando la forma (x - x0) de la fórmula."""
    if power == 0:
        return ""
    base = sp.latex(sym) if center == 0 else rf"\left({sp.latex(sym - center)}\right)"
    return base if power == 1 else f"{base}^{{{power}}}"


def taylor_polynomial_latex(res: TaylorExpansionResult) -> str:
    """LaTeX de P_n en potencias de (x - x0) y (y - y0), ordenado por grado k."""
    x0, y0 = to_exact_number(res.x0), to_exact_number(res.y0)
    parts: list[str] = []
    for k in sorted(res.terms_by_order):
        for term in res.terms_by_order[k]:
            coeff = term["coefficient"]
            if coeff == 0:
                continue
            negative = coeff.could_extract_minus_sign()
            magnitude = -coeff if negative else coeff
            monomial = " ".join(
                f
                for f in (
                    _power_factor_latex(res.x_sym, x0, term["dx_power"]),
                    _power_factor_latex(res.y_sym, y0, term["dy_power"]),
                )
                if f
            )
            if magnitude == 1 and monomial:
                coeff_ltx = ""
            elif isinstance(magnitude, sp.Add):
                coeff_ltx = rf"\left({sp.latex(magnitude)}\right)"
            else:
                coeff_ltx = sp.latex(magnitude)
            body = " ".join(p for p in (coeff_ltx, monomial) if p)
            if not parts:
                parts.append(f"-{body}" if negative else body)
            else:
                parts.append(f"{'-' if negative else '+'} {body}")
    return " ".join(parts) if parts else "0"


def render_latex_mathematical_breakdown(res: TaylorExpansionResult) -> None:
    """
    Renderiza de forma clara y pedagógica el desarrollo algebraico del polinomio de Taylor.
    """
    st.subheader("Desarrollo Simbólico y Analítico", icon=":material/menu_book:")
    st.caption(
        "Aquí está la parte formal: cómo se construye la copia paso a paso. "
        "Cada bloque desplegable trae una explicación en palabras antes de la fórmula."
    )

    # 1. Función Original y Punto de Expansión
    st.markdown(f"**Función objetivo:** $f(x, y) = {sp.latex(res.func_expr)}$")
    st.markdown(f"**Punto de expansión:** $(x_0, y_0) = ({res.x0}, {res.y0})$")
    st.markdown(f"**Valor en el punto:** $f({res.x0}, {res.y0}) = {sp.latex(res.f_at_point)}$")

    # 2. Fórmula general (doble sumatoria) y Polinomio de Taylor resultante (evaluación exacta)
    st.markdown("**Fórmula general de Taylor en dos variables:**")
    st.latex(GENERAL_TAYLOR_FORMULA)
    st.caption(
        "En palabras: se miden las derivadas (cómo cambia la superficie) en el punto de apoyo y cada una se "
        "multiplica por potencias de la distancia a ese punto, (x − x₀) y (y − y₀). Sumar todo da la copia."
    )

    poly_ltx = f"P_{{{res.degree}}}(x, y) = {taylor_polynomial_latex(res)}"
    st.markdown(f"#### Polinomio de Taylor $P_{{{res.degree}}}(x, y)$")
    st.latex(poly_ltx)
    with st.expander("Código LaTeX del polinomio", icon=":material/code:"):
        st.code(poly_ltx, language="latex")

    if res.limit_resolved:
        derivs = ", ".join(
            "f" if (i, j) == (0, 0) else f"∂^{i + j}f/∂x^{i}∂y^{j}" for i, j in res.limit_resolved
        )
        st.info(
            f"Indeterminación en el centro resuelta mediante límites iterados "
            rf"$\lim_{{y \to y_0}} \lim_{{x \to x_0}}$ para: {derivs}.",
            icon=":material/info:",
        )

    # 3. Interpretación de Grado 0: Plano Horizontal
    if res.degree == 0:
        with st.expander("Interpretación Geométrica: Plano Horizontal ($n = 0$)", icon=":material/horizontal_rule:", expanded=True):
            st.markdown(
                "Con grado cero el polinomio solo conserva el término constante: es el **plano horizontal** "
                "que pasa por el punto de apoyo y no captura ninguna pendiente ni curvatura."
            )
            st.latex(rf"P_0(x, y) = f(x_0, y_0) = {sp.latex(res.f_at_point)}")

    # 4. Interpretación de Grado 1: Plano Tangente
    with st.expander("Interpretación Geométrica: Plano Tangente ($n = 1$)", icon=":material/straighten:", expanded=(res.degree == 1)):
        fx, fy = res.gradient
        st.markdown(
            "El polinomio de primer grado coincide exactamente con el **plano tangente** a la superficie en el punto. "
            "El **gradiente** son dos números: la pendiente al caminar en dirección x ($f_x$) y en dirección y ($f_y$)."
        )
        st.latex(
            rf"\nabla f({res.x0}, {res.y0}) = \begin{{pmatrix}} f_x \\ f_y \end{{pmatrix}} = "
            rf"\begin{{pmatrix}} {sp.latex(fx)} \\ {sp.latex(fy)} \end{{pmatrix}}"
        )
        st.latex(
            rf"z = f(x_0, y_0) + f_x(x_0, y_0)(x - x_0) + f_y(x_0, y_0)(y - y_0)"
        )
        st.latex(rf"z = {sp.latex(res.tangent_plane)}")

    # 5. Interpretación de Grado 2: Matriz Hessiana y Curvatura
    if res.degree >= 2 and res.hessian_matrix is not None:
        with st.expander("Interpretación de Grado 2: Matriz Hessiana y Términos Cuadráticos", icon=":material/grid_on:", expanded=(res.degree == 2)):
            H = res.hessian_matrix
            fxx, fxy = H[0][0], H[0][1]
            fyx, fyy = H[1][0], H[1][1]
            det_H = sp.simplify(fxx * fyy - fxy * fyx)

            st.markdown(
                "La aproximación cuadrática incorpora la curvatura mediante la **Matriz Hessiana**: "
                "una tabla con las segundas derivadas, que dicen cómo se *dobla* la superficie en cada dirección."
            )
            st.latex(
                rf"H({res.x0}, {res.y0}) = \begin{{pmatrix}} f_{{xx}} & f_{{xy}} \\ f_{{yx}} & f_{{yy}} \end{{pmatrix}} = "
                rf"\begin{{pmatrix}} {sp.latex(fxx)} & {sp.latex(fxy)} \\ {sp.latex(fyx)} & {sp.latex(fyy)} \end{{pmatrix}}"
            )
            st.markdown(f"**Determinante Hessiano:** $\\det(H) = {sp.latex(det_H)}$")
            if det_H.is_number:
                if det_H > 0 and fxx > 0:
                    st.info("Forma de **tazón** (convexa): el punto podría ser un mínimo local, un fondo de valle.", icon=":material/info:")
                elif det_H > 0 and fxx < 0:
                    st.info("Forma de **cúpula** (cóncava): el punto podría ser un máximo local, una cima.", icon=":material/info:")
                elif det_H < 0:
                    st.info("Forma de **silla de montar**: sube en una dirección y baja en la otra.", icon=":material/info:")

    # 6. Desglose detallado por órdenes
    with st.expander("Desglose término a término por órdenes de derivación", icon=":material/format_list_numbered:"):
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
