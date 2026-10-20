"""Módulo de construcción de gráficos tridimensionales interactivos con Plotly."""

import numpy as np
import plotly.graph_objects as go
from core.numeric_eval import NumericEvaluationResult


def build_taylor_3d_figure(
    eval_res: NumericEvaluationResult,
    x0: float,
    y0: float,
    z0: float,
    func_str: str,
    degree: int,
    show_real: bool = True,
    show_taylor: bool = True,
    show_anchor: bool = True,
    show_tolerance_mask: bool = True,
    real_opacity: float = 0.85,
    taylor_opacity: float = 0.70,
) -> go.Figure:
    """
    Construye la figura tridimensional interactiva comparando f(x, y) y P_n(x, y).
    """
    fig = go.Figure()

    # 1. Superficie de la función original f(x, y)
    if show_real:
        fig.add_trace(
            go.Surface(
                x=eval_res.X,
                y=eval_res.Y,
                z=eval_res.Z_real,
                name=f"f(x, y) = {func_str}",
                colorscale="Viridis",
                opacity=real_opacity,
                showscale=True,
                colorbar=dict(
                    title=dict(text="f(x, y)", font=dict(size=12)),
                    x=-0.12,
                    len=0.7,
                ),
                hovertemplate="<b>f(x, y)</b><br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>",
            )
        )

    # 2. Superficie del Polinomio de Taylor P_n(x, y)
    if show_taylor:
        fig.add_trace(
            go.Surface(
                x=eval_res.X,
                y=eval_res.Y,
                z=eval_res.Z_taylor,
                name=f"P_{degree}(x, y) (Taylor)",
                colorscale="Reds",
                opacity=taylor_opacity,
                showscale=True,
                colorbar=dict(
                    title=dict(text=f"P_{degree}(x,y)", font=dict(size=12)),
                    x=1.02,
                    len=0.7,
                ),
                hovertemplate=f"<b>P_{degree}(x, y)</b><br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<br>z: %{{z:.3f}}<extra></extra>",
            )
        )

    # 3. Región de validez / convergencia (puntos donde el error <= tolerancia)
    if show_tolerance_mask:
        # Puntos válidos sobre la superficie de Taylor
        valid_X = eval_res.X[eval_res.valid_mask]
        valid_Y = eval_res.Y[eval_res.valid_mask]
        valid_Z = eval_res.Z_taylor[eval_res.valid_mask]

        if len(valid_X) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=valid_X,
                    y=valid_Y,
                    z=valid_Z,
                    mode="markers",
                    name=f"Zona Válida (|Error| ≤ {eval_res.tolerance})",
                    marker=dict(
                        size=3.5,
                        color="#00FFAA",
                        opacity=0.75,
                        symbol="circle",
                    ),
                    hovertemplate="<b>Zona de Tolerancia</b><br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>",
                )
            )

    # 4. Punto de anclaje / expansión (x0, y0, f(x0, y0))
    if show_anchor and np.isfinite(z0):
        fig.add_trace(
            go.Scatter3d(
                x=[x0],
                y=[y0],
                z=[z0],
                mode="markers+text",
                name=f"Punto ({x0}, {y0})",
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

    # Configuración de diseño y cámara espacial
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Eje X", gridcolor="#E2E8F0", zerolinecolor="#94A3B8"),
            yaxis=dict(title="Eje Y", gridcolor="#E2E8F0", zerolinecolor="#94A3B8"),
            zaxis=dict(title="Eje Z", gridcolor="#E2E8F0", zerolinecolor="#94A3B8"),
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
    junto con el plano de tolerancia z = epsilon.
    """
    fig = go.Figure()

    # Superficie del error absoluto
    fig.add_trace(
        go.Surface(
            x=eval_res.X,
            y=eval_res.Y,
            z=eval_res.Z_error,
            name="|f(x, y) - P_n(x, y)|",
            colorscale="Plasma",
            opacity=0.9,
            showscale=True,
            colorbar=dict(title=dict(text="Error", font=dict(size=12)), len=0.7),
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
            name=f"Tolerancia (ε = {tolerance})",
            colorscale=[[0, "#00E676"], [1, "#00E676"]],
            opacity=0.4,
            showscale=False,
            hovertemplate=f"<b>Plano de Tolerancia</b><br>ε: {tolerance:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Eje X"),
            yaxis=dict(title="Eje Y"),
            zaxis=dict(title="Error Absoluto |f - P_n|"),
            aspectmode="cube",
        ),
        margin=dict(l=10, r=10, b=10, t=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        height=450,
    )

    return fig
