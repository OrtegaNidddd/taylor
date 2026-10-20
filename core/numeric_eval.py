"""Módulo para la evaluación numérica vectorizada, cálculo de mallas y análisis del residuo."""

from dataclasses import dataclass
import numpy as np
import sympy as sp


@dataclass
class NumericEvaluationResult:
    """Contenedor de arreglos numéricos y métricas de error en la malla."""
    X: np.ndarray
    Y: np.ndarray
    Z_real: np.ndarray
    Z_taylor: np.ndarray
    Z_error: np.ndarray
    tolerance: float
    valid_mask: np.ndarray  # Máscara donde error <= tolerancia
    max_error: float
    mae: float
    rmse: float
    valid_percentage: float
    valid_radius_est: float


def evaluate_surfaces_and_error(
    func_expr: sp.Expr,
    poly_expr: sp.Expr,
    x_sym: sp.Symbol,
    y_sym: sp.Symbol,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    x0: float,
    y0: float,
    tolerance: float = 0.1,
    num_points: int = 40,
    clipping_limit: float = 200.0,
) -> NumericEvaluationResult:
    """
    Evalúa numéricamente f(x, y) y P_n(x, y) sobre una discretización espacial (malla),
    calculando la matriz de error absoluto y las métricas de aproximación.
    """
    xmin, xmax = x_range
    ymin, ymax = y_range

    if xmin >= xmax:
        raise ValueError("El límite inferior x_min debe ser menor que x_max.")
    if ymin >= ymax:
        raise ValueError("El límite inferior y_min debe ser menor que y_max.")
    if tolerance <= 0:
        raise ValueError("La tolerancia de error debe ser un valor positivo.")

    # Generar la malla bidimensional
    x_vals = np.linspace(xmin, xmax, num_points)
    y_vals = np.linspace(ymin, ymax, num_points)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Compilar expresiones simbólicas a funciones vectorizadas con NumPy
    # Usar módulos seguros para evitar desbordamientos no controlados
    f_num = sp.lambdify((x_sym, y_sym), func_expr, modules=["numpy", {"atan2": np.arctan2}])
    p_num = sp.lambdify((x_sym, y_sym), poly_expr, modules=["numpy", {"atan2": np.arctan2}])

    # Evaluación segura con captura de advertencias numéricas
    with np.errstate(all="ignore"):
        try:
            Z_real = f_num(X, Y)
        except Exception:
            # Fallback elemento a elemento si lambdify falla vectorizado
            z_list = []
            for xi, yi in zip(X.ravel(), Y.ravel()):
                try:
                    val = float(func_expr.subs({x_sym: xi, y_sym: yi}))
                except Exception:
                    val = np.nan
                z_list.append(val)
            Z_real = np.array(z_list).reshape(X.shape)

        try:
            Z_taylor = p_num(X, Y)
        except Exception:
            z_list = []
            for xi, yi in zip(X.ravel(), Y.ravel()):
                try:
                    val = float(poly_expr.subs({x_sym: xi, y_sym: yi}))
                except Exception:
                    val = np.nan
                z_list.append(val)
            Z_taylor = np.array(z_list).reshape(X.shape)

    # Convertir a matrices si la evaluación retornó escalares (ej. funciones constantes)
    if np.isscalar(Z_real):
        Z_real = np.full_like(X, float(Z_real), dtype=float)
    else:
        Z_real = np.asarray(Z_real, dtype=float)

    if np.isscalar(Z_taylor):
        Z_taylor = np.full_like(X, float(Z_taylor), dtype=float)
    else:
        Z_taylor = np.asarray(Z_taylor, dtype=float)

    # Filtrar valores extremos/asíntotas que romperían la visualización 3D
    Z_real = np.where(np.abs(Z_real) > clipping_limit, np.nan, Z_real)
    Z_taylor = np.where(np.abs(Z_taylor) > clipping_limit, np.nan, Z_taylor)

    # Cálculo del residuo/error absoluto: |f(x, y) - P_n(x, y)|
    with np.errstate(all="ignore"):
        Z_error = np.abs(Z_real - Z_taylor)

    # Máscara booleana de tolerancia
    valid_mask = (Z_error <= tolerance) & (~np.isnan(Z_error))

    # Cálculo de métricas sobre elementos válidos no nulos
    finite_error = Z_error[np.isfinite(Z_error)]

    if len(finite_error) > 0:
        max_error = float(np.max(finite_error))
        mae = float(np.mean(finite_error))
        rmse = float(np.sqrt(np.mean(finite_error ** 2)))
    else:
        max_error = float("nan")
        mae = float("nan")
        rmse = float("nan")

    total_valid_points = np.count_nonzero(np.isfinite(Z_real) & np.isfinite(Z_taylor))
    if total_valid_points > 0:
        valid_percentage = float(np.count_nonzero(valid_mask) / total_valid_points * 100.0)
    else:
        valid_percentage = 0.0

    # Estimación del radio de validez centrado en (x0, y0)
    # Distancia desde (x0, y0) de todos los puntos dentro de la tolerancia
    dist_from_center = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    valid_dists = dist_from_center[valid_mask]
    if len(valid_dists) > 0:
        # Usamos percentil 90 de las distancias contiguas válidas para evitar puntos aislados
        valid_radius_est = float(np.percentile(valid_dists, 75))
    else:
        valid_radius_est = 0.0

    return NumericEvaluationResult(
        X=X,
        Y=Y,
        Z_real=Z_real,
        Z_taylor=Z_taylor,
        Z_error=Z_error,
        tolerance=tolerance,
        valid_mask=valid_mask,
        max_error=max_error,
        mae=mae,
        rmse=rmse,
        valid_percentage=valid_percentage,
        valid_radius_est=valid_radius_est,
    )
