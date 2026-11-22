"""Textos en lenguaje sencillo para que cualquier persona entienda la app.

Contiene:
- El glosario de términos técnicos explicados sin jerga.
- Los experimentos guiados (configuraciones listas para cargar con un clic).
- `describe_current_state`, que narra en palabras lo que se ve en pantalla.
"""

from dataclasses import dataclass
import math

# -----------------------------------------------------------------------------
# Glosario: término técnico -> explicación en palabras simples
# -----------------------------------------------------------------------------
GLOSSARY: dict[str, str] = {
    "Función f(x, y)": (
        "Una regla que a cada punto del suelo (x, y) le asigna una altura z. "
        "Si dibujas todas esas alturas obtienes una **superficie**, como un terreno con colinas y valles."
    ),
    "Punto de expansión (x₀, y₀)": (
        "El lugar donde 'apoyamos' la aproximación. Ahí la copia es perfecta; "
        "cuanto más te alejas de este punto, más se equivoca."
    ),
    "Grado n": (
        "Qué tanta información usa la copia. Grado 0 solo sabe la altura; grado 1 también la inclinación; "
        "grado 2 también la curvatura; grados mayores, detalles cada vez más finos."
    ),
    "Polinomio de Taylor Pₙ(x, y)": (
        "La 'copia' de la función hecha solo con sumas, restas y potencias de x e y. "
        "Es mucho más fácil de calcular que la función original y se parece a ella cerca del punto de apoyo."
    ),
    "Plano tangente": (
        "La aproximación de grado 1: una hoja plana que toca la superficie en el punto de apoyo "
        "con la misma inclinación, como una tabla apoyada sobre una colina."
    ),
    "Gradiente ∇f": (
        "Dos números que dicen qué tan empinada es la superficie al caminar en dirección x y en dirección y. "
        "Apunta hacia donde la superficie sube más rápido."
    ),
    "Matriz Hessiana H": (
        "Una tabla de 4 números que describe cómo se curva la superficie: si es un tazón (mínimo), "
        "una cúpula (máximo) o una silla de montar (sube en una dirección y baja en otra)."
    ),
    "Error (residuo) E": (
        "La distancia vertical entre la superficie real y la copia: E = |f − Pₙ|. "
        "Si E es pequeño, la copia es buena en ese lugar."
    ),
    "Tolerancia ε": (
        "Cuánto error estás dispuesto a aceptar. Tú la eliges: si el error es menor que ε, "
        "decimos que la copia 'sirve' en ese punto."
    ),
    "Región de validez": (
        "La zona donde el error es menor que la tolerancia ε, o sea, donde la copia sirve. "
        "En el mapa de error es la zona verde, rodeada por la línea negra."
    ),
    "RMSE / MAE": (
        "Promedios del error en toda la ventana. Sirven para comparar configuraciones con un solo número: "
        "más pequeño = mejor copia en general."
    ),
}


# -----------------------------------------------------------------------------
# Experimentos guiados
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Experiment:
    title: str
    preset: str          # Clave en PRESETS (ui/components.py)
    degree: int
    x0: float
    y0: float
    radius: float
    what_to_look: str    # Qué debería observar la persona


EXPERIMENTS: list[Experiment] = [
    Experiment(
        title="1 · La copia más simple (grado 0)",
        preset="Colina suave (campana)",
        degree=0, x0=0.5, y0=0.5, radius=2.0,
        what_to_look=(
            "La copia naranja es un **piso plano** a la altura del punto amarillo. "
            "Solo acierta justo ahí; en cuanto te mueves, se equivoca."
        ),
    ),
    Experiment(
        title="2 · Una tabla apoyada (grado 1)",
        preset="Colina suave (campana)",
        degree=1, x0=0.5, y0=0.5, radius=2.0,
        what_to_look=(
            "Ahora la copia es un **plano inclinado** que toca la colina en el punto amarillo "
            "con la misma pendiente. Acierta en una zona más amplia que el grado 0."
        ),
    ),
    Experiment(
        title="3 · Copiar la curvatura (grado 2)",
        preset="Colina suave (campana)",
        degree=2, x0=0.5, y0=0.5, radius=2.0,
        what_to_look=(
            "La copia ya se **dobla** como la colina. Mira cómo crece la zona verde en la pestaña "
            "\"¿Dónde funciona?\"."
        ),
    ),
    Experiment(
        title="4 · ¡Copia perfecta!",
        preset="Tazón (paraboloide)",
        degree=2, x0=1.0, y0=1.0, radius=3.0,
        what_to_look=(
            "El tazón ya es un polinomio de grado 2, así que la copia de grado 2 coincide **exactamente**: "
            "el error es cero en todas partes."
        ),
    ),
    Experiment(
        title="5 · Más grado = mejor copia",
        preset="Olas (seno × coseno)",
        degree=6, x0=0.0, y0=0.0, radius=3.0,
        what_to_look=(
            "Con grado 6 la copia sigue a las olas bastante lejos del centro. "
            "Prueba a bajar el grado a 1 y a 2 y compara."
        ),
    ),
    Experiment(
        title="6 · Lejos del centro siempre falla",
        preset="Olas (seno × coseno)",
        degree=2, x0=0.0, y0=0.0, radius=6.0,
        what_to_look=(
            "Con una ventana grande se ve la idea clave: la copia es buena **cerca** del punto amarillo "
            "y se despega de la función real cuando te alejas."
        ),
    ),
]


# -----------------------------------------------------------------------------
# Narrador del estado actual
# -----------------------------------------------------------------------------
_DEGREE_MEANING = {
    0: (
        "un **piso plano** a la altura de la superficie en el punto amarillo. "
        "Solo conoce esa altura: no sabe si el terreno sube o baja"
    ),
    1: (
        "el **plano tangente**: una hoja plana que toca la superficie en el punto amarillo "
        "con la misma inclinación, pero que no sabe doblarse"
    ),
    2: (
        "una superficie que copia la **altura, la inclinación y la curvatura** "
        "(si el terreno tiene forma de tazón, de cúpula o de silla de montar)"
    ),
}


def _quality_label(valid_pct: float) -> tuple[str, str]:
    """Etiqueta y emoji según el porcentaje de la ventana donde la copia sirve."""
    if valid_pct >= 90:
        return "excelente", "🟢"
    if valid_pct >= 60:
        return "buena", "🟡"
    if valid_pct >= 25:
        return "regular", "🟠"
    return "pobre", "🔴"


def _fmt(v: float, nd: int = 3) -> str:
    if not math.isfinite(v):
        return "no definido"
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        return f"{v:.2e}"
    return f"{v:.{nd}f}"


def describe_current_state(
    degree: int,
    x0: float,
    y0: float,
    f0: float,
    radius: float,
    tolerance: float,
    valid_pct: float,
    max_error: float,
    point_x1: float | None = None,
    point_y1: float | None = None,
    point_f: float | None = None,
    point_p: float | None = None,
    point_err: float | None = None,
) -> str:
    """Devuelve un párrafo en Markdown que explica, sin jerga, lo que se ve en pantalla."""
    meaning = _DEGREE_MEANING.get(
        degree,
        f"una superficie que copia la altura, la inclinación, la curvatura y **{degree - 2} nivel(es) "
        "más de detalle**. Cuanto mayor es el grado, más lejos del punto amarillo sigue pareciéndose",
    )
    lines = [
        f"**Lo que ves:** la superficie **azul** es la función real. La **naranja** es su copia de "
        f"grado {degree}, que es {meaning}.",
        f"**El punto de apoyo** (amarillo) está en ({x0:g}, {y0:g}), donde la altura es "
        f"{_fmt(f0)}. Ahí la copia es exacta.",
    ]

    if math.isfinite(max_error) and max_error < 1e-9:
        lines.append(
            "✨ **¡La copia es perfecta en toda la ventana!** Esto ocurre cuando la función ya es un "
            f"polinomio de grado {degree} o menor: no hay nada más que copiar."
        )
    else:
        label, icon = _quality_label(valid_pct)
        lines.append(
            f"{icon} **¿Qué tan buena es?** Aceptando un error de hasta {tolerance:g}, la copia sirve en el "
            f"**{valid_pct:.0f} %** del área mostrada (un cuadrado de ±{radius:g} alrededor del punto). "
            f"Calificación: **{label}**."
        )

    if (
        point_x1 is not None and point_err is not None
        and point_f is not None and point_p is not None
        and math.isfinite(point_err)
    ):
        verdict = "✅ dentro de lo aceptable" if point_err <= tolerance else "❌ más de lo aceptable"
        lines.append(
            f"**En el punto de prueba** ({point_x1:g}, {point_y1:g}) (línea roja en el 3D): la función real vale "
            f"{_fmt(point_f)} y la copia dice {_fmt(point_p)}. Se equivoca por {_fmt(point_err)} → {verdict}."
        )

    if degree < 6 and not (math.isfinite(max_error) and max_error < 1e-9):
        lines.append(
            f"💡 **Prueba:** sube el grado a {degree + 1} y mira cómo la naranja se pega más a la azul."
        )
    return "\n\n".join(lines)
