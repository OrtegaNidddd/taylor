"""Pruebas unitarias para el módulo del Tutor Inteligente con IA."""

import os
import pytest
from core.ai_tutor import (
    generate_heuristic_explanation,
    explain_current_state_with_ai,
    get_ai_credentials,
    call_gemini_api,
)


def test_get_ai_credentials():
    """Valida que la lectura de credenciales desde el entorno funcione sin errores."""
    creds = get_ai_credentials()
    assert "provider" in creds
    assert "gemini_key" in creds
    assert "deepseek_key" in creds


def test_heuristic_fallback_generation():
    """Valida la generación de explicaciones didácticas automáticas sin necesidad de API Key."""
    text = generate_heuristic_explanation(
        func_str="x^2 + y^2",
        x0=1.0,
        y0=2.0,
        f0=5.0,
        degree=1,
        poly_str="2x + 4y - 5",
        fx="2",
        fy="4",
        det_h=None,
        valid_pct=45.0,
        valid_radius=1.8,
        tolerance=0.2,
    )
    assert "Diagnóstico Didáctico" in text
    assert "Plano Tangente" in text
    assert "x_0, y_0" in text


def test_explain_current_state_fallback_when_no_key():
    """Valida que si no se proporciona API key, devuelva el diagnóstico heurístico amigable."""
    explanation = explain_current_state_with_ai(
        func_str="sin(x) * cos(y)",
        x0=0.0,
        y0=0.0,
        f0=0.0,
        degree=2,
        poly_str="x",
        gradient_str=("1", "0"),
        hessian_det_str="0",
        valid_percentage=85.0,
        rmse=0.05,
        max_error=0.15,
        valid_radius=2.5,
        tolerance=0.2,
        override_api_key="",  # Sin clave
        override_provider="gemini",
    )
    assert "Diagnóstico Didáctico" in explanation
    assert "Para habilitar explicaciones generadas en tiempo real por IA" in explanation


def test_live_gemini_generation_with_configured_key():
    """Si hay una clave configurada en el entorno, valida que la API de Gemini responda exitosamente."""
    creds = get_ai_credentials()
    key = creds["gemini_key"]
    if not key:
        pytest.skip("No hay GEMINI_API_KEY configurada para prueba en vivo.")

    resp = call_gemini_api("Explica en 1 oración qué es el plano tangente.", key)
    assert len(resp) > 10
    assert isinstance(resp, str)
