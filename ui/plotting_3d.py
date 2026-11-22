"""Gráficos tridimensionales interactivos con Plotly: superficies f y P_n, y superficie del residuo."""

import numpy as np
import plotly.graph_objects as go
from core.taylor_math import NumericEvaluationResult, PointComparisonResult


def build_taylor_3d_figure(
    eval_res: NumericEvaluationResult,
    x0: float,
    y0: float,
    z0: float,
    func_str: str,
    degree: int,
    taylor_opacity: float = 0.70,
    point_cmp: PointComparisonResult | None = None,
) -> go.Figure:
    """
    Construye la figura tridimensional que superpone f(x, y) (Viridis) y P_n(x, y)
    (paleta cálida con transparencia ajustable), el punto central y, si se pasa
    `point_cmp`, el comparador vertical de error puntual en (x1, y1).
    """
    fig = go.Figure()

    # 1. Superficie de la función original f(x, y)
    fig.add_trace(
        go.Surface(
            x=eval_res.X,
            y=eval_res.Y,
            z=eval_res.Z_real,
            name=f"Función real f(x, y) = {func_str}",
            colorscale=[[0, "#1E3A8A"], [1, "#60A5FA"]],
            opacity=0.9,
            showscale=False,
            showlegend=True,
            hovertemplate="<b>Función real f(x, y)</b><br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>",
        )
    )

    # 2. Superficie del Polinomio de Taylor P_n(x, y)
    fig.add_trace(
        go.Surface(
            x=eval_res.X,
            y=eval_res.Y,
            z=eval_res.Z_taylor,
            name=f"Copia de Taylor P{degree}(x, y) (grado {degree})",
            colorscale=[[0, "#C2410C"], [1, "#FDBA74"]],
            opacity=taylor_opacity,
            showscale=False,
            showlegend=True,
            hovertemplate=f"<b>Copia P{degree}(x, y)</b><br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<br>z: %{{z:.3f}}<extra></extra>",
        )
    )

    # 3. Trazador del punto central (x0, y0, f(x0, y0))
    if np.isfinite(z0):
        fig.add_trace(
            go.Scatter3d(
                x=[x0],
                y=[y0],
                z=[z0],
                mode="markers+text",
                name=f"Punto de apoyo ({x0:g}, {y0:g})",
                text=[f"  ({x0:.2f}, {y0:.2f}, {z0:.2f})"],
                textposition="top right",
                textfont=dict(color="#FFD700", size=13),
                marker=dict(
                    size=8,
                    color="#FFD700",
                    symbol="diamond",
                    line=dict(color="#000000", width=2),
                ),
                hovertemplate=f"<b>Punto de Expansión</b><br>x0: {x0:.3f}<br>y0: {y0:.3f}<br>z0: {z0:.3f}<extra></extra>",
            )
        )

    # 4. Comparador vertical de error puntual: segmento entre f(x1, y1) y P_n(x1, y1)
    if point_cmp is not None and np.isfinite(point_cmp.f_exact) and np.isfinite(point_cmp.p_approx):
        x1, y1 = point_cmp.x1, point_cmp.y1
        fig.add_trace(
            go.Scatter3d(
                x=[x1, x1],
                y=[y1, y1],
                z=[point_cmp.f_exact, point_cmp.p_approx],
                mode="lines+markers",
                name=f"Error en el punto de prueba ({x1:.2f}, {y1:.2f})",
                line=dict(color="#E11D48", width=7),
                marker=dict(size=5, color=["#1D4ED8", "#EA580C"], symbol="circle"),
                text=["Función real f(x₁, y₁)", "Copia Pₙ(x₁, y₁)"],
                hovertemplate=(
                    "<b>%{text}</b><br>x₁: %{x:.3f}<br>y₁: %{y:.3f}<br>z: %{z:.5f}"
                    f"<br>|E| = {point_cmp.abs_error:.3e}<extra></extra>"
                ),
            )
        )

    # Configuración de diseño y cámara espacial
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Eje X", gridcolor="#E2E8F0", zerolinecolor="#94A3B8"),
            yaxis=dict(title="Eje Y", gridcolor="#E2E8F0", zerolinecolor="#94A3B8"),
            zaxis=dict(title="Altura z", gridcolor="#E2E8F0", zerolinecolor="#94A3B8"),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="cube",
        ),
        margin=dict(l=10, r=10, b=10, t=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        height=620,
    )

    return fig


def build_error_surface_figure(
    eval_res: NumericEvaluationResult,
    x0: float,
    y0: float,
    tolerance: float,
) -> go.Figure:
    """
    Construye la gráfica 3D específica del residuo |f(x, y) - P_n(x, y)|
    junto con el plano de tolerancia z = epsilon y su curva de contorno |E| = epsilon.
    """
    fig = go.Figure()

    # Superficie del error absoluto (recortada por percentiles para no perder escala)
    fig.add_trace(
        go.Surface(
            x=eval_res.X,
            y=eval_res.Y,
            z=eval_res.Z_error_display,
            name="Error |f − Pₙ| (distancia entre real y copia)",
            colorscale=[[0, "#16A34A"], [0.5, "#FACC15"], [1, "#DC2626"]],
            opacity=0.9,
            showscale=True,
            showlegend=True,
            colorbar=dict(title=dict(text="Error", font=dict(size=12)), len=0.7),
            # Curva de nivel |E| = ε dibujada sobre la superficie y proyectada al plano XY
            contours=dict(
                z=dict(
                    show=True,
                    start=tolerance,
                    end=tolerance,
                    size=1.0,
                    color="#000000",
                    width=6,
                    project=dict(z=True),
                )
            ),
            hovertemplate="<b>Error Absoluto</b><br>x: %{x:.3f}<br>y: %{y:.3f}<br>Error: %{z:.4f}<extra></extra>",
        )
    )

    # Plano de tolerancia horizontal z = epsilon
    Z_tol = np.full_like(eval_res.X, tolerance)
    fig.add_trace(
        go.Surface(
            x=eval_res.X,
            y=eval_res.Y,
            z=Z_tol,
            name=f"Techo de error aceptable (ε = {tolerance})",
            colorscale=[[0, "#64748B"], [1, "#64748B"]],
            opacity=0.35,
            showscale=False,
            showlegend=True,
            hovertemplate=f"<b>Plano de Tolerancia</b><br>ε: {tolerance:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Eje X"),
            yaxis=dict(title="Eje Y"),
            zaxis=dict(title="Error |f − Pₙ|"),
            aspectmode="cube",
        ),
        margin=dict(l=10, r=10, b=10, t=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        height=450,
    )

    return fig
