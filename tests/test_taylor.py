"""Pruebas exhaustivas para el motor de Taylor."""

import time
import pytest
import sympy as sp
from core.taylor_math import compute_taylor_polynomial


def test_taylor_tangent_plane_degree_1():
    """Valida que el polinomio de grado 1 coincida con el plano tangente exacto."""
    x, y = sp.symbols("x y", real=True)
    res = compute_taylor_polynomial(x**2 + y**2, x, y, x0=1.0, y0=2.0, degree=1)

    expected_plane = 5 + 2 * (x - 1) + 4 * (y - 2)
    assert sp.simplify(res.polynomial_symbolic - expected_plane) == 0
    assert sp.simplify(res.polynomial_simplified - (2 * x + 4 * y - 5)) == 0
    assert res.degree == 1
    assert res.hessian_matrix is None


def test_taylor_degree_2_and_hessian():
    """Valida la aproximación de segundo grado y el cálculo de la matriz Hessiana."""
    x, y = sp.symbols("x y", real=True)
    res = compute_taylor_polynomial(sp.sin(x) * sp.cos(y), x, y, x0=0.0, y0=0.0, degree=2)

    assert sp.simplify(res.polynomial_simplified - x) == 0
    assert res.hessian_matrix is not None
    assert len(res.hessian_matrix) == 2


def test_taylor_degree_3():
    """Valida una función polinómica exacta de grado 3."""
    x, y = sp.symbols("x y", real=True)
    f = x**3 + y**3 + 2 * x * y
    res = compute_taylor_polynomial(f, x, y, x0=0.0, y0=0.0, degree=3)
    assert sp.simplify(res.polynomial_simplified - f) == 0


def test_taylor_degree_4_and_5():
    """Verifica que el cálculo hasta grado 5 funcione y sea rápido."""
    x, y = sp.symbols("x y", real=True)
    f = sp.exp(x) * sp.cos(y)
    t0 = time.time()
    res = compute_taylor_polynomial(f, x, y, x0=0.0, y0=0.0, degree=4)
    duration = time.time() - t0

    assert res.degree == 4
    assert duration < 1.0  # Menos de 1 segundo
    # Para e^x cos(y), en (0,0) f(0,0)=1
    assert res.f_at_point == 1


def test_taylor_singular_point_raises_value_error():
    """Debe advertir con ValueError si la función o alguna derivada es singular en el punto de apoyo."""
    x, y = sp.symbols("x y", real=True)
    with pytest.raises(ValueError, match="no está definida o es singular"):
        compute_taylor_polynomial(sp.log(x), x, y, x0=0.0, y0=1.0, degree=1)


def test_taylor_degree_0_is_horizontal_plane():
    """Con n = 0 el polinomio es la constante z = f(x0, y0)."""
    x, y = sp.symbols("x y", real=True)
    res = compute_taylor_polynomial(sp.exp(x) * sp.cos(y), x, y, x0=1.0, y0=0.5, degree=0)
    assert res.degree == 0
    assert res.polynomial_simplified.free_symbols == set()
    assert abs(float(res.polynomial_simplified) - float(sp.exp(1.0) * sp.cos(0.5))) < 1e-12
    assert res.hessian_matrix is None
    assert list(res.terms_by_order.keys()) == [0]


def test_taylor_degree_6_exact_and_fast():
    """Grado máximo n = 6: reproduce exactamente un polinomio de grado 6 y es rápido."""
    x, y = sp.symbols("x y", real=True)
    f = x**6 - 3 * x**3 * y**3 + y**6 + x * y
    res = compute_taylor_polynomial(f, x, y, x0=0.5, y0=-1.0, degree=6)
    assert sp.simplify(res.polynomial_simplified - f) == 0
    assert len(res.terms_by_order[6]) == 7

    t0 = time.time()
    compute_taylor_polynomial(sp.sin(x) * sp.exp(y), x, y, x0=0.3, y0=0.2, degree=6)
    assert time.time() - t0 < 2.0


@pytest.mark.parametrize("bad_degree", [-1, 7])
def test_taylor_degree_out_of_range_raises(bad_degree):
    x, y = sp.symbols("x y", real=True)
    with pytest.raises(ValueError, match="entre 0 y 6"):
        compute_taylor_polynomial(x + y, x, y, x0=0.0, y0=0.0, degree=bad_degree)


# -----------------------------------------------------------------------------
# Pruebas Matemáticas de Control (FORMATO_PPA, Resultados 2)
# -----------------------------------------------------------------------------
import numpy as np
from core.taylor_math import evaluate_surfaces_and_error

X, Y = sp.symbols("x y", real=True)


def test_control_sin_degree_3_exact():
    res = compute_taylor_polynomial(sp.sin(X), X, Y, x0=0.0, y0=0.0, degree=3)
    assert sp.expand(res.polynomial_symbolic - (X - X**3 / 6)) == 0  # coeficientes racionales exactos


def test_control_cos_degree_2_exact():
    res = compute_taylor_polynomial(sp.cos(X), X, Y, x0=0.0, y0=0.0, degree=2)
    assert sp.expand(res.polynomial_symbolic - (1 - X**2 / 2)) == 0


def test_control_ln_at_1_0_degree_2_exact():
    res = compute_taylor_polynomial(sp.log(X), X, Y, x0=1.0, y0=0.0, degree=2)
    assert sp.expand(res.polynomial_symbolic - ((X - 1) - (X - 1) ** 2 / 2)) == 0


@pytest.mark.parametrize(
    "f, x0, y0",
    [
        (sp.sin(X) * sp.cos(Y), 0.7, -0.3),
        (sp.exp(-(X**2 + Y**2)), 0.4, 0.9),
        (sp.log(1 + X**2 + Y**2), -1.2, 0.5),
        (X**3 - 3 * X * Y**2, 1.1, 2.0),
        (sp.sqrt(4 + X + Y), 0.3, 0.1),
    ],
)
def test_control_exact_contact_at_center(f, x0, y0):
    """|f(x0, y0) - P_n(x0, y0)| < 1e-15 para todo grado."""
    for n in range(0, 7):
        res = compute_taylor_polynomial(f, X, Y, x0=x0, y0=y0, degree=n)
        sub = {X: sp.Rational(str(x0)), Y: sp.Rational(str(y0))}
        gap = abs(float(f.subs(sub)) - float(res.polynomial_symbolic.subs(sub)))
        assert gap < 1e-15


def test_control_degree_0_horizontal_and_degree_1_tangent():
    f = sp.exp(X) * sp.sin(Y)
    p0 = compute_taylor_polynomial(f, X, Y, x0=0.5, y0=1.0, degree=0).polynomial_symbolic
    assert p0.free_symbols == set()
    assert sp.simplify(p0 - f.subs({X: sp.Rational(1, 2), Y: 1})) == 0

    res1 = compute_taylor_polynomial(f, X, Y, x0=0.5, y0=1.0, degree=1)
    assert sp.simplify(res1.polynomial_symbolic - res1.tangent_plane) == 0


def test_control_convergence_error_decreases_with_degree():
    """Al subir n = 1 -> 2 -> 3 -> 4 el error máximo baja y el área válida crece (no decrece)."""
    f = sp.exp(X + Y / 2)
    max_errors, valid = [], []
    for n in (1, 2, 3, 4):
        poly = compute_taylor_polynomial(f, X, Y, x0=0.0, y0=0.0, degree=n).polynomial_symbolic
        ev = evaluate_surfaces_and_error(f, poly, X, Y, (-1.0, 1.0), (-1.0, 1.0), 0.0, 0.0, tolerance=0.05)
        max_errors.append(ev.max_error)
        valid.append(ev.valid_percentage)
    assert all(a > b for a, b in zip(max_errors, max_errors[1:]))
    assert max_errors[-1] < max_errors[0] / 10
    assert all(a < b for a, b in zip(valid, valid[1:]))


# -----------------------------------------------------------------------------
# Indeterminaciones resueltas mediante límites iterados (Fase 2)
# -----------------------------------------------------------------------------
def test_iterated_limits_resolve_removable_singularity():
    res = compute_taylor_polynomial(sp.sin(X) / X, X, Y, x0=0.0, y0=0.0, degree=4)
    assert sp.expand(res.polynomial_symbolic - (1 - X**2 / 6 + X**4 / 120)) == 0
    assert (0, 0) in res.limit_resolved


def test_iterated_limits_keep_true_singularities():
    with pytest.raises(ValueError, match="no está definida o es singular"):
        compute_taylor_polynomial(1 / (X**2 + Y**2), X, Y, x0=0.0, y0=0.0, degree=1)
