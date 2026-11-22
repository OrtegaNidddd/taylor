"""Backend matemático del Visualizador de Taylor en dos variables (taylor_math.py).

Reúne en un único módulo las tres capas del motor de cálculo:

1. Parseo seguro de f(x, y) con lista blanca de funciones (SymPy).
2. Motor simbólico: derivadas parciales iteradas, evaluación exacta en (x0, y0)
   (con límites iterados para resolver indeterminaciones) y Polinomio de Taylor
   P_n(x, y) para grados n = 0, 1, ..., 6.
3. Discretización espacial con NumPy (np.meshgrid / np.linspace), compilación con
   sp.lambdify, recorte robusto de asíntotas por percentiles (2–98 %), cálculo del
   residuo E(x, y) = |f(x, y) - P_n(x, y)| y comparador puntual en (x1, y1).

Este módulo no depende de Streamlit; la caché (@st.cache_data) vive en app.py.
"""

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

# Rango de grados admitidos para el polinomio de Taylor
MIN_DEGREE = 0
MAX_DEGREE = 6

# Percentiles del recorte robusto de asíntotas (Fase 3)
CLIP_LOWER_PERCENTILE = 2.0
CLIP_UPPER_PERCENTILE = 98.0
# Holgura añadida a cada lado de la banda [p2, p98], como fracción de su ancho,
# para no recortar los extremos legítimos de funciones suaves.
CLIP_MARGIN_FACTOR = 0.5


# =============================================================================
# 1. PARSEO SEGURO DE EXPRESIONES
# =============================================================================

# Transformaciones seguras para interpretar sintaxis matemática común (ej. 2x, x^2)
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# Diccionario seguro de funciones y constantes matemáticas permitidas
ALLOWED_NAMES = {
    # Tipos numéricos y símbolos necesarios para el árbol sintáctico de SymPy
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Symbol": sp.Symbol,
    "Rational": sp.Rational,
    # Constantes
    "pi": sp.pi,
    "Pi": sp.pi,
    "PI": sp.pi,
    "e": sp.E,
    "E": sp.E,
    # Funciones trigonométricas
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "atan2": sp.atan2,
    "sec": sp.sec,
    "csc": sp.csc,
    "cot": sp.cot,
    # Funciones hiperbólicas
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    # Exponenciales y logarítmicas
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    # Raíces y valor absoluto
    "sqrt": sp.sqrt,
    "cbrt": sp.cbrt,
    "abs": sp.Abs,
    "Abs": sp.Abs,
}

# Palabras clave y nombres prohibidos por motivos de seguridad
FORBIDDEN_PATTERNS = [
    r"__",
    r"\bimport\b",
    r"\beval\b",
    r"\bexec\b",
    r"\bcompile\b",
    r"\bopen\b",
    r"\bos\b",
    r"\bsys\b",
    r"\bsubprocess\b",
    r"\bgetattr\b",
    r"\bsetattr\b",
    r"\bdelattr\b",
    r"\bglobals\b",
    r"\blocals\b",
    r"\bvars\b",
    r"\bdir\b",
]


class MathParserError(Exception):
    """Excepción base para errores de parseo matemático."""
    pass


class InvalidIdentifierError(MathParserError):
    """Lanzada cuando la expresión contiene funciones o variables no autorizadas."""
    pass


class MathSyntaxError(MathParserError):
    """Lanzada cuando la sintaxis matemática es inválida."""
    pass


def sanitize_input(expr_str: str) -> str:
    """Limpia caracteres extraños y verifica patrones maliciosos en la cadena de entrada."""
    if not expr_str or not expr_str.strip():
        raise MathSyntaxError("La expresión de la función no puede estar vacía.")

    cleaned = expr_str.strip()

    # Chequeo estricto de seguridad contra inyecciones y accesos a dunder/builtins
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            raise InvalidIdentifierError(
                f"Expresión rechazada por motivos de seguridad: contiene patrones prohibidos ('{pattern}')."
            )

    # Normalizar espacios
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def parse_function_2d(expr_str: str) -> tuple[sp.Expr, sp.Symbol, sp.Symbol]:
    """
    Parsea de forma segura una expresión matemática f(x, y).

    Retorna:
        tuple[sp.Expr, sp.Symbol, sp.Symbol]: (expresión simbólica, símbolo x, símbolo y)

    Lanza:
        MathSyntaxError: Si la sintaxis es incorrecta.
        InvalidIdentifierError: Si contiene variables o funciones no permitidas.
    """
    cleaned = sanitize_input(expr_str)

    x, y = sp.symbols("x y", real=True)
    local_dict = {**ALLOWED_NAMES, "x": x, "y": y}
    safe_global_dict = {"__builtins__": {}}

    try:
        parsed_expr = parse_expr(
            cleaned,
            local_dict=local_dict,
            global_dict=safe_global_dict,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )
    except MathParserError:
        raise
    except (SyntaxError, TypeError, ValueError, Exception) as exc:
        raise MathSyntaxError(
            f"Error de sintaxis en la expresión '{expr_str}': {exc}"
        ) from exc

    # Asegurar que sea una expresión de SymPy
    if not isinstance(parsed_expr, sp.Basic):
        parsed_expr = sp.sympify(parsed_expr)

    # Validar que los símbolos libres no excedan 'x' e 'y'
    free_symbols = parsed_expr.free_symbols
    invalid_symbols = [s for s in free_symbols if s not in (x, y)]
    if invalid_symbols:
        nombres = ", ".join(s.name for s in invalid_symbols)
        raise InvalidIdentifierError(
            f"Variables o identificadores no permitidos detectados: {nombres}. "
            "Solo se admiten las variables 'x' e 'y'."
        )

    return parsed_expr, x, y


# =============================================================================
# 2. MOTOR SIMBÓLICO DEL POLINOMIO DE TAYLOR
# =============================================================================

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
    limit_resolved: list[tuple[int, int]]  # Derivadas (i, j) obtenidas por límites iterados


def _is_singular(value: sp.Expr) -> bool:
    return bool(value.has(sp.nan, sp.zoo, sp.oo, -sp.oo))


def to_exact_number(value: float) -> sp.Rational:
    """Convierte un número decimal de la interfaz (ej. 0.1) en el racional exacto (1/10)."""
    return sp.nsimplify(round(float(value), 10), rational=True)


def _iterated_limit(expr: sp.Expr, x_sym: sp.Symbol, y_sym: sp.Symbol, x0: sp.Expr, y0: sp.Expr) -> sp.Expr:
    """Límite iterado lim_{y->y0} lim_{x->x0} expr, usado para resolver indeterminaciones 0/0."""
    try:
        return sp.limit(sp.limit(expr, x_sym, x0), y_sym, y0)
    except Exception:
        return sp.nan


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

    Con n = 0 el resultado es el plano horizontal constante z = f(x0, y0).

    El centro se convierte a números racionales exactos, de modo que las derivadas se evalúan
    de forma exacta (ej. sin(x) en 0 da P_3 = x - x**3/6, sin decimales). Si una derivada
    produce una indeterminación en el centro (ej. sin(x)/x en x = 0), se resuelve mediante
    límites iterados; si el límite no es finito, la función es realmente singular.
    """
    if not MIN_DEGREE <= degree <= MAX_DEGREE:
        raise ValueError(
            f"El grado del polinomio de Taylor debe estar entre {MIN_DEGREE} y {MAX_DEGREE}."
        )

    x0_val = to_exact_number(x0)
    y0_val = to_exact_number(y0)
    sub_point = {x_sym: x0_val, y_sym: y0_val}
    limit_resolved: list[tuple[int, int]] = []

    def evaluate_at_center(expr: sp.Expr, key: tuple[int, int]) -> sp.Expr:
        val = expr.subs(sub_point)
        if _is_singular(val):
            val = _iterated_limit(expr, x_sym, y_sym, x0_val, y0_val)
            if not _is_singular(val):
                limit_resolved.append(key)
        return val

    # Valor de la función en el punto de expansión (validación de dominio finito)
    f_at_point = evaluate_at_center(func_expr, (0, 0))
    if _is_singular(f_at_point):
        raise ValueError(
            f"La función f(x, y) no está definida o es singular en el punto de apoyo ({x0}, {y0})."
        )

    # Derivadas iteradas: D_x^i D_y^j f se obtiene derivando la ya calculada D_x^i D_y^{j-1} f
    # (o D_x^{i-1} f si j = 0), evitando recalcular cada derivada desde cero.
    symbolic_derivs: dict[tuple[int, int], sp.Expr] = {(0, 0): func_expr}
    deriv_cache: dict[tuple[int, int], sp.Expr] = {(0, 0): f_at_point}

    def get_symbolic_derivative(i: int, j: int) -> sp.Expr:
        if (i, j) not in symbolic_derivs:
            if j > 0:
                symbolic_derivs[(i, j)] = sp.diff(get_symbolic_derivative(i, j - 1), y_sym)
            else:
                symbolic_derivs[(i, j)] = sp.diff(get_symbolic_derivative(i - 1, 0), x_sym)
        return symbolic_derivs[(i, j)]

    def get_eval_derivative(i: int, j: int) -> sp.Expr:
        if (i, j) not in deriv_cache:
            deriv_cache[(i, j)] = evaluate_at_center(get_symbolic_derivative(i, j), (i, j))
        return deriv_cache[(i, j)]

    poly_accum = sp.Number(0)
    terms_by_order: dict[int, list[dict[str, Any]]] = {}

    for k in range(degree + 1):
        terms_by_order[k] = []
        for i in range(k + 1):
            j = k - i
            c_fact = sp.Rational(1, math.factorial(i) * math.factorial(j))
            deriv_val = get_eval_derivative(i, j)

            if _is_singular(deriv_val):
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

    # Plano tangente (referencia pedagógica, se calcula para cualquier grado)
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

    # Forma expandida en potencias de x e y (expand basta para un polinomio y es mucho
    # más rápido que simplify en grados altos)
    poly_simplified = sp.expand(poly_accum)

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
        limit_resolved=limit_resolved,
    )


# =============================================================================
# 3. DISCRETIZACIÓN, RECORTE ROBUSTO Y RESIDUO
# =============================================================================

@dataclass
class NumericEvaluationResult:
    """Contenedor de arreglos numéricos y métricas de error en la malla."""
    X: np.ndarray
    Y: np.ndarray
    Z_real: np.ndarray        # f(x, y) recortada para visualización
    Z_taylor: np.ndarray      # P_n(x, y) recortado para visualización
    Z_error: np.ndarray       # Residuo |f - P_n| (sin recortar, NaN solo en asíntotas)
    Z_error_display: np.ndarray  # Residuo recortado por percentiles para la superficie 3D
    tolerance: float
    valid_mask: np.ndarray  # Máscara donde error <= tolerancia
    max_error: float
    mae: float
    rmse: float
    valid_percentage: float
    valid_radius_est: float
    clip_bounds: tuple[float, float] | None  # Banda de recorte aplicada a f y P_n


def robust_clip_bounds(
    Z: np.ndarray,
    lower_pct: float = CLIP_LOWER_PERCENTILE,
    upper_pct: float = CLIP_UPPER_PERCENTILE,
    margin_factor: float = CLIP_MARGIN_FACTOR,
) -> tuple[float, float] | None:
    """
    Calcula la banda de recorte robusta [p_lo - m, p_hi + m] a partir de los percentiles
    (por defecto 2 % y 98 %) de los valores finitos de Z.

    La holgura m = margin_factor * (p_hi - p_lo) preserva los máximos y mínimos legítimos
    de funciones suaves, mientras que los picos de asíntotas (1/x, tan, ...) quedan fuera.
    Retorna None si no hay valores finitos.
    """
    finite = Z[np.isfinite(Z)]
    if finite.size == 0:
        return None
    p_lo, p_hi = np.percentile(finite, [lower_pct, upper_pct])
    span = float(p_hi - p_lo)
    margin = margin_factor * span if span > 0 else max(1.0, abs(float(p_hi)))
    return float(p_lo) - margin, float(p_hi) + margin


def _apply_bounds(Z: np.ndarray, bounds: tuple[float, float] | None) -> np.ndarray:
    if bounds is None:
        return Z
    lo, hi = bounds
    with np.errstate(invalid="ignore"):
        return np.where((Z < lo) | (Z > hi), np.nan, Z)


def _evaluate_on_grid(expr: sp.Expr, x_sym: sp.Symbol, y_sym: sp.Symbol, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Evalúa una expresión sobre la malla; los infinitos y complejos se convierten en NaN."""
    fn = sp.lambdify((x_sym, y_sym), expr, modules=["numpy", {"atan2": np.arctan2}])
    with np.errstate(all="ignore"):
        try:
            Z = fn(X, Y)
        except Exception:
            # Fallback elemento a elemento si lambdify falla vectorizado
            z_list = []
            for xi, yi in zip(X.ravel(), Y.ravel()):
                try:
                    val = float(expr.subs({x_sym: xi, y_sym: yi}))
                except Exception:
                    val = np.nan
                z_list.append(val)
            Z = np.array(z_list).reshape(X.shape)

    # Convertir a matriz si la evaluación retornó un escalar (ej. funciones constantes)
    if np.isscalar(Z) or np.ndim(Z) == 0:
        Z = np.full_like(X, float(np.real(Z)), dtype=float)
    Z = np.asarray(Z)
    if np.iscomplexobj(Z):
        Z = np.where(np.abs(np.imag(Z)) > 1e-12, np.nan, np.real(Z))
    Z = np.asarray(Z, dtype=float)
    return np.where(np.isfinite(Z), Z, np.nan)


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
    clipping_limit: float | None = None,
) -> NumericEvaluationResult:
    """
    Evalúa numéricamente f(x, y) y P_n(x, y) sobre una discretización espacial (malla),
    calculando la matriz de error absoluto y las métricas de aproximación.

    El recorte de asíntotas usa percentiles (2–98 %) de f(x, y) sobre la malla.
    `clipping_limit` es un tope absoluto adicional y opcional (|z| > límite → NaN).
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

    Z_real_raw = _evaluate_on_grid(func_expr, x_sym, y_sym, X, Y)
    if not np.any(np.isfinite(Z_real_raw)):
        raise ValueError("La función f(x, y) no toma valores reales finitos en la ventana de visualización.")
    Z_taylor_raw = _evaluate_on_grid(poly_expr, x_sym, y_sym, X, Y)

    if clipping_limit is not None:
        Z_real_raw = np.where(np.abs(Z_real_raw) > clipping_limit, np.nan, Z_real_raw)
        Z_taylor_raw = np.where(np.abs(Z_taylor_raw) > clipping_limit, np.nan, Z_taylor_raw)

    # Recorte robusto por percentiles (2–98 %) calculado sobre la función original.
    # La misma banda se aplica a P_n para que ambas superficies compartan escala.
    clip_bounds = robust_clip_bounds(Z_real_raw)
    Z_real = _apply_bounds(Z_real_raw, clip_bounds)
    Z_taylor = _apply_bounds(Z_taylor_raw, clip_bounds)

    # Residuo |f(x, y) - P_n(x, y)|: se calcula con los valores sin recortar del polinomio
    # (para no ocultar su divergencia) y se descarta solo donde f está en una asíntota.
    with np.errstate(all="ignore"):
        Z_error = np.abs(Z_real - Z_taylor_raw)
    Z_error_display = _apply_bounds(Z_error, robust_clip_bounds(Z_error))

    # Máscara booleana de tolerancia
    valid_mask = np.isfinite(Z_error) & (np.nan_to_num(Z_error, nan=np.inf) <= tolerance)

    # Métricas sobre los elementos finitos
    finite_error = Z_error[np.isfinite(Z_error)]
    if finite_error.size > 0:
        max_error = float(np.max(finite_error))
        mae = float(np.mean(finite_error))
        rmse = float(np.sqrt(np.mean(finite_error ** 2)))
        valid_percentage = float(np.count_nonzero(valid_mask) / finite_error.size * 100.0)
    else:
        max_error = mae = rmse = float("nan")
        valid_percentage = 0.0

    # Estimación del radio de validez centrado en (x0, y0)
    dist_from_center = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    valid_dists = dist_from_center[valid_mask]
    if valid_dists.size > 0:
        # Percentil 75 de las distancias válidas para atenuar puntos aislados
        valid_radius_est = float(np.percentile(valid_dists, 75))
    else:
        valid_radius_est = 0.0

    return NumericEvaluationResult(
        X=X,
        Y=Y,
        Z_real=Z_real,
        Z_taylor=Z_taylor,
        Z_error=Z_error,
        Z_error_display=Z_error_display,
        tolerance=tolerance,
        valid_mask=valid_mask,
        max_error=max_error,
        mae=mae,
        rmse=rmse,
        valid_percentage=valid_percentage,
        valid_radius_est=valid_radius_est,
        clip_bounds=clip_bounds,
    )


# =============================================================================
# 4. COMPARADOR PUNTUAL EN (x1, y1)
# =============================================================================

@dataclass
class PointComparisonResult:
    """Comparación entre f y P_n en un punto arbitrario (x1, y1)."""
    x1: float
    y1: float
    f_exact: float        # f(x1, y1)
    p_approx: float       # P_n(x1, y1)
    abs_error: float      # |f - P_n|
    rel_error_pct: float  # |f - P_n| / |f| * 100  (NaN si f ≈ 0)


def _eval_point(expr: sp.Expr, x_sym: sp.Symbol, y_sym: sp.Symbol, x1: float, y1: float) -> float:
    try:
        val = complex(expr.subs({x_sym: sp.Float(x1), y_sym: sp.Float(y1)}).evalf())
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")
    if abs(val.imag) > 1e-12 or not math.isfinite(val.real):
        return float("nan")
    return val.real


def compare_at_point(
    func_expr: sp.Expr,
    poly_expr: sp.Expr,
    x_sym: sp.Symbol,
    y_sym: sp.Symbol,
    x1: float,
    y1: float,
    zero_threshold: float = 1e-12,
) -> PointComparisonResult:
    """
    Evalúa f(x1, y1) y P_n(x1, y1) con precisión simbólica y calcula:
      - Error absoluto:  E = |f(x1, y1) - P_n(x1, y1)|
      - Error relativo:  |f - P_n| / |f| * 100 %  (indefinido si |f| < zero_threshold)
    Si f no está definida en (x1, y1), los valores dependientes son NaN.
    """
    f_val = _eval_point(func_expr, x_sym, y_sym, x1, y1)
    p_val = _eval_point(poly_expr, x_sym, y_sym, x1, y1)

    abs_err = abs(f_val - p_val)  # NaN se propaga
    if math.isfinite(f_val) and abs(f_val) >= zero_threshold:
        rel_err = abs_err / abs(f_val) * 100.0
    else:
        rel_err = float("nan")

    return PointComparisonResult(
        x1=x1,
        y1=y1,
        f_exact=f_val,
        p_approx=p_val,
        abs_error=abs_err,
        rel_error_pct=rel_err,
    )
