"""Motor simbólico para el cálculo y expansión del Polinomio de Taylor en dos variables."""

import math
from dataclasses import dataclass
from typing import Any
import sympy as sp


@dataclass
class TaylorExpansionResult:
    """Contenedor de resultados del cálculo del Polinomio de Taylor."""
    func_expr: sp.Expr
    x_sym: sp.Symbol
    y_sym: sp.Symbol
    x0: float
    y0: float
    degree: int
    polynomial_symbolic: sp.Expr
    polynomial_simplified: sp.Expr
    f_at_point: sp.Expr
    gradient: tuple[sp.Expr, sp.Expr]  # (fx, fy)
    tangent_plane: sp.Expr             # Plano tangente (n=1)
    hessian_matrix: list[list[sp.Expr]] | None  # Matriz Hessiana si n >= 2
    terms_by_order: dict[int, list[dict[str, Any]]]  # Desglose pedagógico por orden


def compute_taylor_polynomial(
    func_expr: sp.Expr,
    x_sym: sp.Symbol,
    y_sym: sp.Symbol,
    x0: float,
    y0: float,
    degree: int,
) -> TaylorExpansionResult:
    """
    Calcula el polinomio de Taylor de dos variables alrededor de (x0, y0) hasta el grado n.

    Utiliza la fórmula multivariable general:
    P_n(x, y) = sum_{k=0}^{n} sum_{i=0}^{k} (1 / (i! * (k-i)!)) * D_x^i D_y^{k-i} f(x0, y0) * (x - x0)^i * (y - y0)^{k-i}
    """
    if degree < 1:
        raise ValueError("El grado del polinomio de Taylor debe ser al menos 1.")

    x0_val = sp.Float(x0)
    y0_val = sp.Float(y0)
    sub_point = {x_sym: x0_val, y_sym: y0_val}

    # Valor de la función en el punto de expansión
    f_at_point = func_expr.subs(sub_point)
    if f_at_point.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        raise ValueError(
            f"La función f(x, y) no está definida o es singular en el punto de apoyo ({x0}, {y0})."
        )

    # Cache de derivadas evaluadas: (i, j) -> valor en (x0, y0)
    deriv_cache: dict[tuple[int, int], sp.Expr] = {}

    def get_eval_derivative(i: int, j: int) -> sp.Expr:
        if (i, j) not in deriv_cache:
            diff_expr = sp.diff(func_expr, x_sym, i, y_sym, j)
            val = diff_expr.subs(sub_point)
            deriv_cache[(i, j)] = val
        return deriv_cache[(i, j)]

    poly_accum = sp.Number(0)
    terms_by_order: dict[int, list[dict[str, Any]]] = {}

    for k in range(degree + 1):
        terms_by_order[k] = []
        for i in range(k + 1):
            j = k - i
            c_fact = sp.Rational(1, math.factorial(i) * math.factorial(j))
            deriv_val = get_eval_derivative(i, j)

            if deriv_val.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                raise ValueError(
                    f"La derivada parcial d^{k}f / (dx^{i} dy^{j}) no existe o es singular en ({x0}, {y0})."
                )

            term_coeff = c_fact * deriv_val
            monomial = (x_sym - x0_val) ** i * (y_sym - y0_val) ** j
            term_expr = term_coeff * monomial

            poly_accum += term_expr

            terms_by_order[k].append({
                "order": k,
                "dx_power": i,
                "dy_power": j,
                "derivative_value": deriv_val,
                "combinatorial_factor": c_fact,
                "coefficient": term_coeff,
                "monomial": monomial,
                "term_expr": term_expr,
            })

    # Plano tangente (n=1)
    fx = get_eval_derivative(1, 0)
    fy = get_eval_derivative(0, 1)
    tangent_plane = f_at_point + fx * (x_sym - x0_val) + fy * (y_sym - y0_val)

    # Matriz Hessiana (si grado >= 2)
    hessian_matrix = None
    if degree >= 2:
        fxx = get_eval_derivative(2, 0)
        fxy = get_eval_derivative(1, 1)
        fyy = get_eval_derivative(0, 2)
        hessian_matrix = [
            [fxx, fxy],
            [fxy, fyy],
        ]

    # Polinomio simplificado
    poly_simplified = sp.simplify(poly_accum)

    return TaylorExpansionResult(
        func_expr=func_expr,
        x_sym=x_sym,
        y_sym=y_sym,
        x0=x0,
        y0=y0,
        degree=degree,
        polynomial_symbolic=poly_accum,
        polynomial_simplified=poly_simplified,
        f_at_point=f_at_point,
        gradient=(fx, fy),
        tangent_plane=tangent_plane,
        hessian_matrix=hessian_matrix,
        terms_by_order=terms_by_order,
    )
