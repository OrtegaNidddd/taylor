"""Pruebas unitarias para el módulo de parseo seguro de expresiones matemáticas."""

import pytest
import sympy as sp
from core.parser import (
    parse_function_2d,
    MathParserError,
    MathSyntaxError,
    InvalidIdentifierError,
)


def test_parse_valid_elementary_functions():
    """Valida el parseo de funciones elementales continuas."""
    expr, x, y = parse_function_2d("sin(x) + cos(y)")
    assert expr == sp.sin(x) + sp.cos(y)

    expr, x, y = parse_function_2d("exp(-(x^2 + y^2))")
    assert expr == sp.exp(-(x**2 + y**2))

    expr, x, y = parse_function_2d("log(1 + x^2 + y^2)")
    assert expr == sp.log(1 + x**2 + y**2)


def test_parse_implicit_multiplication_and_xor():
    """Verifica que el parser admita potencias con ^ y multiplicación implícita."""
    expr, x, y = parse_function_2d("2x + 3y")
    assert expr == 2 * x + 3 * y

    expr, x, y = parse_function_2d("x^3 - y^3")
    assert expr == x**3 - y**3


def test_parse_constants():
    """Valida el soporte para constantes matemáticas pi y e."""
    expr, x, y = parse_function_2d("pi * x + e * y")
    assert expr == sp.pi * x + sp.E * y


def test_parse_empty_input():
    """Debe lanzar MathSyntaxError cuando la cadena esté vacía o solo contenga espacios."""
    with pytest.raises(MathSyntaxError):
        parse_function_2d("")

    with pytest.raises(MathSyntaxError):
        parse_function_2d("   ")


def test_parse_invalid_identifiers():
    """Debe rechazar variables no permitidas como z, t, w."""
    with pytest.raises(InvalidIdentifierError):
        parse_function_2d("x^2 + y^2 + z^2")

    with pytest.raises(InvalidIdentifierError):
        parse_function_2d("variable_desconocida * x")


def test_parse_security_blocks():
    """Debe bloquear llamadas sospechosas como eval o importaciones."""
    with pytest.raises(MathParserError):
        parse_function_2d("__import__('os').system('ls')")

    with pytest.raises(MathParserError):
        parse_function_2d("eval('2+2')")
