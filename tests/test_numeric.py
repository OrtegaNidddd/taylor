"""Pruebas unitarias para la evaluación numérica, mallas y residuo."""

import time
import numpy as np
import sympy as sp
import pytest
from core.taylor_math import evaluate_surfaces_and_error


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


def test_percentile_clipping_removes_asymptote_only():
    """El recorte 2–98 % elimina el pico de la asíntota y conserva los extremos de una función suave."""
    from core.taylor_math import robust_clip_bounds

    x, y = sp.symbols("x y", real=True)
    # Asíntota en el origen: los pocos puntos centrales se disparan
    res = evaluate_surfaces_and_error(
        func_expr=1 / (x**2 + y**2 + 1e-4), poly_expr=sp.Number(1), x_sym=x, y_sym=y,
        x_range=(-3.0, 3.0), y_range=(-3.0, 3.0), x0=1.0, y0=1.0, tolerance=0.1, num_points=41,
    )
    assert np.isnan(res.Z_real[20, 20])          # el centro (valor 1e4) queda recortado
    assert np.nanmax(res.Z_real) < 100            # la escala ya no está dominada por la asíntota
    assert np.isnan(res.Z_error[20, 20])          # el error en la asíntota no contamina métricas
    assert res.clip_bounds is not None

    # Función suave: no se pierde ningún punto
    smooth = evaluate_surfaces_and_error(
        func_expr=sp.sin(x) * sp.cos(y), poly_expr=x, x_sym=x, y_sym=y,
        x_range=(-3.0, 3.0), y_range=(-3.0, 3.0), x0=0.0, y0=0.0, tolerance=0.1, num_points=40,
    )
    assert np.all(np.isfinite(smooth.Z_real))

    assert robust_clip_bounds(np.full((3, 3), np.nan)) is None


def test_error_not_hidden_when_taylor_diverges():
    """El residuo usa P_n sin recortar: si el polinomio diverge, el error sigue siendo finito y grande."""
    x, y = sp.symbols("x y", real=True)
    res = evaluate_surfaces_and_error(
        func_expr=sp.sin(x), poly_expr=x**6, x_sym=x, y_sym=y,
        x_range=(-4.0, 4.0), y_range=(-1.0, 1.0), x0=0.0, y0=0.0, tolerance=0.1, num_points=30,
    )
    assert np.all(np.isfinite(res.Z_error))
    assert res.max_error > 1000


def test_compare_at_point_absolute_and_relative_error():
    from core.taylor_math import compare_at_point

    x, y = sp.symbols("x y", real=True)
    f = sp.exp(x + y)
    p = 1 + x + y
    cmp = compare_at_point(f, p, x, y, 0.5, 0.25)
    exact = float(np.exp(0.75))
    assert abs(cmp.f_exact - exact) < 1e-12
    assert abs(cmp.p_approx - 1.75) < 1e-12
    assert abs(cmp.abs_error - abs(exact - 1.75)) < 1e-12
    assert abs(cmp.rel_error_pct - abs(exact - 1.75) / exact * 100) < 1e-9


def test_compare_at_point_undefined_cases():
    from core.taylor_math import compare_at_point

    x, y = sp.symbols("x y", real=True)
    # f(x1, y1) = 0 → error relativo indefinido
    cmp_zero = compare_at_point(sp.sin(x), x, x, y, 0.0, 1.0)
    assert cmp_zero.abs_error == 0.0
    assert np.isnan(cmp_zero.rel_error_pct)
    # f no definida (log de negativo) → NaN
    cmp_undef = compare_at_point(sp.log(x), x - 1, x, y, -1.0, 0.0)
    assert np.isnan(cmp_undef.f_exact) and np.isnan(cmp_undef.abs_error)


def test_error_heatmap_has_tolerance_contour():
    from ui.plotting_2d import build_error_heatmap_figure
    from core.taylor_math import compare_at_point

    x, y = sp.symbols("x y", real=True)
    res = evaluate_surfaces_and_error(
        func_expr=sp.exp(x) * sp.cos(y), poly_expr=1 + x, x_sym=x, y_sym=y,
        x_range=(-2.0, 2.0), y_range=(-2.0, 2.0), x0=0.0, y0=0.0, tolerance=0.2, num_points=30,
    )
    cmp = compare_at_point(sp.exp(x) * sp.cos(y), 1 + x, x, y, 1.0, 1.0)
    fig = build_error_heatmap_figure(res, 0.0, 0.0, 0.2, point_cmp=cmp)
    heat = [t for t in fig.data if t.type == "heatmap"][0]
    assert heat.colorscale[0][1] is not None  # escala verde (sirve) / rojo (falla)
    contours = [t for t in fig.data if t.type == "contour"]
    tol_contour = contours[-1]
    assert abs(tol_contour.contours.start - np.log10(0.2)) < 1e-12
    assert tol_contour.line.color == "#000000"


def test_window_without_finite_values_raises():
    """Validación de dominio finito: si f no es real en toda la ventana, se informa el error."""
    x, y = sp.symbols("x y", real=True)
    with pytest.raises(ValueError, match="valores reales finitos"):
        evaluate_surfaces_and_error(
            sp.log(x), x, x, y, x_range=(-3.0, -1.0), y_range=(-1.0, 1.0), x0=-2.0, y0=0.0
        )
