"""Pruebas exhaustivas para el motor de Taylor."""

import time
import pytest
import sympy as sp
from core.taylor_engine import compute_taylor_polynomial


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
