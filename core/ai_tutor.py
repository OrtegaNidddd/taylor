"""Módulo de Tutor Inteligente con IA (Gemini y DeepSeek) para explicaciones pedagógicas."""

import os
import requests
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Cache en memoria de modelos disponibles para evitar llamadas repetidas a ListModels
_CACHED_AVAILABLE_GEMINI_MODELS: list[str] | None = None


def get_ai_credentials() -> dict[str, str]:
    """Obtiene la configuración activa de IA desde variables de entorno."""
    provider = os.getenv("AI_PROVIDER", "gemini").lower().strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

    return {
        "provider": provider,
        "gemini_key": gemini_key,
        "gemini_model": gemini_model,
        "deepseek_key": deepseek_key,
        "deepseek_model": deepseek_model,
    }


def list_supported_gemini_models(api_key: str) -> list[str]:
    """Consulta la API de Google para descubrir los modelos que admiten generateContent para esta clave."""
    global _CACHED_AVAILABLE_GEMINI_MODELS
    if _CACHED_AVAILABLE_GEMINI_MODELS is not None:
        return _CACHED_AVAILABLE_GEMINI_MODELS

    for version in ["v1beta", "v1"]:
        try:
            url = f"https://generativelanguage.googleapis.com/{version}/models?key={api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                valid_models = []
                for m in data.get("models", []):
                    methods = m.get("supportedGenerationMethods", [])
                    if "generateContent" in methods:
                        m_name = m.get("name", "").replace("models/", "")
                        valid_models.append(m_name)
                if valid_models:
                    _CACHED_AVAILABLE_GEMINI_MODELS = valid_models
                    return valid_models
        except Exception:
            continue

    return []


def select_best_gemini_model(api_key: str, requested_model: str) -> tuple[str, str]:
    """
    Determina el mejor modelo disponible y la versión de la API correspondiente.
    Retorna (nombre_modelo, api_version).
    """
    clean_req = requested_model.replace("models/", "").strip()
    available = list_supported_gemini_models(api_key)

    if available:
        # Si el modelo solicitado está disponible, usarlo
        if clean_req in available:
            return clean_req, "v1beta"

        # Buscar por coincidencias de prefijo (ej. gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-flash)
        priority_prefixes = [
            "gemini-3.5-flash",
            "gemini-flash-lite-latest",
            "gemini-3-flash-preview",
            "gemini-flash-latest",
            "gemini-3.7-flash",
            "gemini-3.8-flash",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]
        for pref in priority_prefixes:
            for avail in available:
                if avail.startswith(pref):
                    return avail, "v1beta"

        # Si ninguno de los preferidos coincide, tomar el primer modelo disponible
        return available[0], "v1beta"

    # Fallback si ListModels no respondió: intentar con candidatos estándar
    return clean_req or "gemini-3.5-flash", "v1beta"


def call_gemini_api(prompt: str, api_key: str, model: str = "gemini-3.5-flash") -> str:
    """Envía un prompt a la API de Google Gemini con auto-recuperación de modelo."""
    if not api_key:
        raise ValueError("No se ha configurado la clave GEMINI_API_KEY.")

    # Resolver el modelo soportado por la API key
    selected_model, api_version = select_best_gemini_model(api_key, model)

    candidate_models = [
        selected_model,
        "gemini-3.5-flash",
        "gemini-flash-lite-latest",
        "gemini-3-flash-preview",
        "gemini-flash-latest",
        "gemini-2.5-flash",
        "gemini-1.5-flash",
    ]
    # Eliminar duplicados manteniendo el orden
    candidate_models = list(dict.fromkeys(candidate_models))

    last_error = None
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "Eres un profesor universitario de Cálculo Multivariado de élite, apasionado, claro y didáctico.\n"
                            "REGLAS OBLIGATORIAS:\n"
                            "1. Responde SIEMPRE en español fluido y didáctico.\n"
                            "2. Proporciona una explicación COMPLETA, bien estructurada, con introducción, análisis y conclusiones.\n"
                            "3. Utiliza formato Markdown con títulos claros, listas y fórmulas matemáticas en LaTeX ($...$ para inline y $$...$$ para bloques).\n"
                            "4. No dejes oraciones a medias ni explicaciones truncadas.\n\n"
                            + prompt
                        )
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 3500,
            "thinkingConfig": {
                "thinkingBudget": 0
            }
        },
    }

    for candidate in candidate_models:
        for ver in ["v1beta", "v1"]:
            url = f"https://generativelanguage.googleapis.com/{ver}/models/{candidate}:generateContent?key={api_key}"
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=25)
                if response.status_code == 200:
                    res_json = response.json()
                    candidates = res_json.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        clean_texts = [p.get("text", "") for p in parts if not p.get("thought", False)]
                        result = "".join(clean_texts).strip()
                        if result:
                            return result
                else:
                    error_msg = response.text
                    try:
                        err_json = response.json()
                        if "error" in err_json:
                            error_msg = err_json["error"].get("message", error_msg)
                    except Exception:
                        pass
                    last_error = f"URL={url} | Status={response.status_code} | Msg={error_msg}"
            except Exception as e:
                last_error = str(e)

    # Si todos los candidatos fallaron con 404 u otro error:
    available_list = list_supported_gemini_models(api_key)
    avail_str = f" Modelos disponibles en tu proyecto: {', '.join(available_list)}" if available_list else ""
    raise RuntimeError(
        f"No se pudo conectar con ningún modelo de Gemini compatible.{avail_str}. Detalle: {last_error}"
    )


def call_deepseek_api(prompt: str, api_key: str, model: str = "deepseek-chat") -> str:
    """Envía un prompt a la API de DeepSeek."""
    if not api_key:
        raise ValueError("No se ha configurado la clave DEEPSEEK_API_KEY.")

    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un profesor universitario experto en Cálculo Multivariado. "
                    "Explica con claridad, rigor y enfoque geométrico cómo el Polinomio de Taylor aproxima a la función en R^2."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "max_tokens": 800,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=25)
    if response.status_code != 200:
        error_msg = response.text
        try:
            err_json = response.json()
            if "error" in err_json:
                error_msg = err_json["error"].get("message", error_msg)
        except Exception:
            pass
        raise RuntimeError(f"Error en DeepSeek API ({response.status_code}): {error_msg}")

    res_json = response.json()
    choices = res_json.get("choices", [])
    if not choices:
        return "No se obtuvo respuesta del modelo DeepSeek."

    return choices[0].get("message", {}).get("content", "")


def generate_heuristic_explanation(
    func_str: str,
    x0: float,
    y0: float,
    f0: float,
    degree: int,
    poly_str: str,
    fx: str,
    fy: str,
    det_h: str | None,
    valid_pct: float,
    valid_radius: float,
    tolerance: float,
) -> str:
    """
    Genera un diagnóstico pedagógico analítico estructurado cuando no hay una API Key configurada.
    """
    text = []
    text.append(f"### :material/school: Diagnóstico Didáctico Automatizado para $f(x, y) = {func_str}$")
    text.append(f"- **Punto de anclaje:** En $(x_0, y_0) = ({x0}, {y0})$, la superficie alcanza la altura $z = {f0:.4f}$.")

    if degree == 0:
        text.append(
            f"- **Plano Horizontal ($n=0$):** El polinomio es la constante $P_0 = {f0:.4f}$. "
            f"No incorpora pendiente ni curvatura, así que el error crece linealmente en cuanto "
            f"la superficie deja de ser plana alrededor del centro."
        )
    elif degree == 1:
        text.append(
            f"- **Plano Tangente ($n=1$):** La pendiente en dirección $x$ es $f_x = {fx}$ y en $y$ es $f_y = {fy}$. "
            f"El plano tangente es una aproximación puramente plana (lineal). No captura la curvatura, por lo que el error "
            f"crece cuadráticamente conforme te alejas del centro."
        )
    elif degree == 2:
        text.append(
            f"- **Aproximación Cuadrática ($n=2$):** Incorpora la curvatura mediante la matriz Hessiana. "
            f"El determinante Hessiano es $\\det(H) = {det_h if det_h is not None else 'N/A'}$. "
        )
        if det_h is not None:
            text.append(
                "La aproximación cuadrática adopta la forma de una cuádrica local (paraboloide o cono/ensilladura), "
                "adaptándose mucho mejor a la concavidad o convexidad de la superficie original."
            )
    else:
        text.append(
            f"- **Orden Superior ($n={degree}$):** La aproximación incorpora derivadas de orden {degree}, "
            f"lo que extiende significativamente la zona donde el polinomio se 'pega' a la superficie original."
        )

    text.append(
        f"- **Comportamiento del Residuo:** Con una tolerancia $\\epsilon = {tolerance}$, la aproximación cubre el "
        f"**{valid_pct:.1f}%** de la ventana visualizada (radio de confianza $\\approx {valid_radius:.2f}$ unidades). "
        f"Fuera de la región de validez (curva blanca E = ε del mapa de calor), la tasa de divergencia del polinomio supera el umbral permitido."
    )

    return "\n\n".join(text)


def explain_current_state_with_ai(
    func_str: str,
    x0: float,
    y0: float,
    f0: float,
    degree: int,
    poly_str: str,
    gradient_str: tuple[str, str],
    hessian_det_str: str | None,
    valid_percentage: float,
    rmse: float,
    max_error: float,
    valid_radius: float,
    tolerance: float,
    custom_question: str | None = None,
    override_api_key: str | None = None,
    override_provider: str | None = None,
) -> str:
    """
    Coordina la generación de la explicación del modelo actual, usando Gemini, DeepSeek o el generador analítico.
    """
    creds = get_ai_credentials()
    provider = override_provider or creds["provider"]

    if override_api_key is not None:
        active_key = override_api_key.strip()
    else:
        active_key = creds["gemini_key"] if provider == "gemini" else creds["deepseek_key"]

    fx, fy = gradient_str

    # Si no hay llave disponible, retornar el diagnóstico analítico local
    if not active_key:
        fallback = generate_heuristic_explanation(
            func_str=func_str,
            x0=x0,
            y0=y0,
            f0=f0,
            degree=degree,
            poly_str=poly_str,
            fx=fx,
            fy=fy,
            det_h=hessian_det_str,
            valid_pct=valid_percentage,
            valid_radius=valid_radius,
            tolerance=tolerance,
        )
        return (
            fallback
            + "\n\n> :material/lightbulb: *Para habilitar explicaciones generadas en tiempo real por IA (Gemini / DeepSeek), "
            "configura tu API Key en el archivo `.env` o ingrésala en la barra lateral.*"
        )

    # Preparar el prompt estructurado con los datos del estado actual
    prompt = f"""
Analiza la siguiente aproximación del Polinomio de Taylor en dos variables para un estudiante de Cálculo Multivariado:

1. Función original: f(x, y) = {func_str}
2. Punto de expansión: (x0, y0) = ({x0}, {y0}), donde f({x0}, {y0}) = {f0:.4f}
3. Grado del polinomio: n = {degree}
4. Polinomio de Taylor calculado: P_{degree}(x, y) = {poly_str}
5. Gradiente en el punto: fx = {fx}, fy = {fy}
6. Determinante Hessiano: {hessian_det_str if hessian_det_str else 'No aplica (grado < 2)'}
7. Estadísticas del Residuo:
   - Umbral de tolerancia de error (epsilon): {tolerance}
   - Porcentaje del dominio dentro de la tolerancia: {valid_percentage:.1f}%
   - Radio estimado de validez: {valid_radius:.2f}
   - Error máximo en la ventana: {max_error:.4f}
   - RMSE: {rmse:.4f}

Pregunta o enfoque solicitado:
{custom_question if custom_question else 'Explica conceptualmente: ¿Qué significa geométricamente este polinomio comparado con la función real? ¿Por qué la zona de validez tiene esa forma y qué ocurre al aumentar el grado o cambiar de punto? Responde en español, usando formato Markdown y LaTeX claro.'}
"""

    if provider == "gemini":
        return call_gemini_api(prompt, active_key, creds["gemini_model"])
    elif provider == "deepseek":
        return call_deepseek_api(prompt, active_key, creds["deepseek_model"])
    else:
        raise ValueError(f"Proveedor de IA desconocido: '{provider}'. Usa 'gemini' o 'deepseek'.")
