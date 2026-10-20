"""Módulo de parseo seguro y validación de expresiones matemáticas multivariables."""

import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

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
