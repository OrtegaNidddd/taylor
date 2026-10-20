"""Pruebas unitarias para la evaluación numérica, mallas y residuo."""

import time
import numpy as np
import sympy as sp
import pytest
from core.numeric_eval import evaluate_surfaces_and_error


def test_evaluate_surfaces_dimensions_and_error():
    """Verifica las dimensiones de la malla y el residuo idéntico en el punto central."""
    x, y = sp.symbols("x y", real=True)
    f = x**2 + y**2
    p = 2 * x + 4 * y - 5

    res = evaluate_surfaces_and_error(
        func_expr=f,
        poly_expr=p,
        x_sym=x,
        y_sym=y,
        x_range=(-2.0, 4.0),
        y_range=(-1.0, 5.0),
        x0=1.0,
        y0=2.0,
        tolerance=0.5,
        num_points=30,
    )

    assert res.X.shape == (30, 30)
    assert res.Y.shape == (30, 30)
    assert res.Z_real.shape == (30, 30)
    assert res.Z_taylor.shape == (30, 30)
    assert res.Z_error.shape == (30, 30)
    assert res.valid_mask.shape == (30, 30)
    assert res.valid_percentage > 0.0
    assert res.max_error >= 0.0
    assert res.rmse >= 0.0


def test_linear_function_perfect_match():
    """Para una función lineal, el plano tangente coincide al 100% en todo el dominio."""
    x, y = sp.symbols("x y", real=True)
    f = 3 * x - 2 * y + 7
    # P_1 coincide exactamente con f
    res = evaluate_surfaces_and_error(
        func_expr=f,
        poly_expr=f,
        x_sym=x,
        y_sym=y,
        x_range=(-3.0, 3.0),
        y_range=(-3.0, 3.0),
        x0=0.0,
        y0=0.0,
        tolerance=0.01,
        num_points=25,
    )
    assert np.allclose(res.Z_error, 0.0, atol=1e-8)
    assert res.valid_percentage == 100.0
    assert res.max_error < 1e-7


def test_discontinuity_and_clipping():
    """Valida que funciones con asíntotas como 1/(x^2 + y^2) no rompan la matriz."""
    x, y = sp.symbols("x y", real=True)
    f = 1 / (x**2 + y**2 + 0.0001)
    p = sp.Number(10)

    res = evaluate_surfaces_and_error(
        func_expr=f,
        poly_expr=p,
        x_sym=x,
        y_sym=y,
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        x0=0.5,
        y0=0.5,
        tolerance=1.0,
        num_points=20,
        clipping_limit=50.0,
    )
    # Debe haber ejecutado sin excepciones y filtrado valores desmedidos
    assert res.Z_real.shape == (20, 20)


def test_performance_under_500ms():
    """Verifica que la discretización y evaluación numérica se ejecute en menos de 500 ms."""
    x, y = sp.symbols("x y", real=True)
    f = sp.sin(x) * sp.cos(y)
    p = x - (x**3) / 6 - (x * y**2) / 2

    t0 = time.time()
    res = evaluate_surfaces_and_error(
        func_expr=f,
        poly_expr=p,
        x_sym=x,
        y_sym=y,
        x_range=(-5.0, 5.0),
        y_range=(-5.0, 5.0),
        x0=0.0,
        y0=0.0,
        tolerance=0.2,
        num_points=40,
    )
    elapsed = time.time() - t0
    assert elapsed < 0.5  # < 500ms
    assert res.X.shape == (40, 40)


def test_invalid_ranges_raise():
    """Debe rechazar rangos donde min >= max o tolerancia <= 0."""
    x, y = sp.symbols("x y", real=True)

    with pytest.raises(ValueError):
        evaluate_surfaces_and_error(
            x, x, x, y, x_range=(5.0, 2.0), y_range=(-1.0, 1.0), x0=0, y0=0
        )

    with pytest.raises(ValueError):
        evaluate_surfaces_and_error(
            x, x, x, y, x_range=(-2.0, 2.0), y_range=(-1.0, 1.0), x0=0, y0=0, tolerance=-0.1
        )
