"""Mapa de calor 2D del residuo |f - P_n| con curvas de nivel de la región de validez."""

import numpy as np
import plotly.graph_objects as go
from core.taylor_math import NumericEvaluationResult, PointComparisonResult

# Piso numérico para log10(E) cuando el error es exactamente 0 (coincidencia perfecta)
LOG_ERROR_FLOOR = 1e-16


def build_error_heatmap_figure(
    eval_res: NumericEvaluationResult,
    x0: float,
    y0: float,
    tolerance: float,
    point_cmp: PointComparisonResult | None = None,
) -> go.Figure:
    """
    Construye el mapa de calor logarítmico log10(E) (escala Turbo) del error absoluto
    E(x, y), en escala verde (la copia sirve, E < ε) / rojo (falla, E > ε), con curvas de nivel:
      - una curva gruesa en E = ε que delimita la región de validez,
      - curvas finas en cada década de error.
    """
    x_vals = eval_res.X[0, :]
    y_vals = eval_res.Y[:, 0]

    with np.errstate(all="ignore"):
        Z = np.log10(np.maximum(eval_res.Z_error, LOG_ERROR_FLOOR))
    level = float(np.log10(tolerance))
    z_label = "log₁₀(E)"

    fig = go.Figure()

    # Escala centrada en la tolerancia: verde = la copia sirve (E < ε), rojo = falla (E > ε).
    finite_all = Z[np.isfinite(Z)]
    span = 3.0
    if finite_all.size > 0:
        span = max(1.0, float(np.max(np.abs(finite_all - level))))
    zmin, zmax = level - span, level + span
    tick_exps = np.arange(np.floor(zmin), np.ceil(zmax) + 1)
    tick_exps = tick_exps[(tick_exps >= zmin) & (tick_exps <= zmax)]
    if tick_exps.size > 7:
        tick_exps = tick_exps[:: int(np.ceil(tick_exps.size / 7))]
    tickvals = list(tick_exps) + [level]
    ticktext = [f"{10.0 ** e:.0e}".replace("e+00", "").replace("e-0", "e-").replace("e+0", "e") for e in tick_exps]
    ticktext.append(f"<b>ε = {tolerance:g}</b>")

    fig.add_trace(
        go.Heatmap(
            x=x_vals,
            y=y_vals,
            z=Z,
            zmin=zmin,
            zmax=zmax,
            colorscale=[[0, "#15803D"], [0.45, "#BBF7D0"], [0.5, "#FEF9C3"], [0.55, "#FECACA"], [1, "#B91C1C"]],
            colorbar=dict(
                title=dict(text="Error", font=dict(size=12)),
                tickvals=tickvals,
                ticktext=ticktext,
                len=0.85,
            ),
            customdata=eval_res.Z_error,
            hovertemplate="x: %{x:.3f}<br>y: %{y:.3f}<br>Error: %{customdata:.3g}<extra></extra>",
            name=z_label,
        )
    )

    # Curvas de nivel finas en cada década de error
    finite = Z[np.isfinite(Z)]
    if finite.size > 0:
        z_min, z_max = np.floor(finite.min()), np.ceil(finite.max())
        if z_max > z_min:
            fig.add_trace(
                go.Contour(
                    x=x_vals,
                    y=y_vals,
                    z=Z,
                    contours=dict(
                        coloring="none",
                        start=z_min,
                        end=z_max,
                        size=1.0,
                        showlabels=False,
                    ),
                    line=dict(color="rgba(51,65,85,0.35)", width=1, dash="dot"),
                    showscale=False,
                    hoverinfo="skip",
                    name="Curvas de nivel del error (log₁₀)",
                )
            )

    # Curva de nivel principal: frontera de la región de validez E = ε
    fig.add_trace(
        go.Contour(
            x=x_vals,
            y=y_vals,
            z=Z,
            contours=dict(coloring="none", start=level, end=level, size=1.0),
            line=dict(color="#000000", width=3.5),
            showscale=False,
            showlegend=True,
            hoverinfo="skip",
            name=f"Frontera: error = {tolerance:g}",
        )
    )

    # Punto de expansión
    fig.add_trace(
        go.Scatter(
            x=[x0],
            y=[y0],
            mode="markers",
            marker=dict(size=13, color="#FFD700", symbol="diamond", line=dict(color="#000000", width=2)),
            name="Punto de apoyo",
            hovertemplate=f"<b>Punto de expansión</b><br>({x0:.3f}, {y0:.3f})<extra></extra>",
        )
    )

    # Punto del comparador puntual
    if point_cmp is not None:
        fig.add_trace(
            go.Scatter(
                x=[point_cmp.x1],
                y=[point_cmp.y1],
                mode="markers",
                marker=dict(size=12, color="#FFFFFF", symbol="x", line=dict(color="#E11D48", width=2)),
                name="Punto de prueba",
                hovertemplate=(
                    f"<b>Comparador puntual</b><br>({point_cmp.x1:.3f}, {point_cmp.y1:.3f})"
                    f"<br>E = {point_cmp.abs_error:.3e}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        xaxis=dict(title="Eje X", range=[x_vals[0], x_vals[-1]]),
        yaxis=dict(title="Eje Y", range=[y_vals[0], y_vals[-1]], scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, b=10, t=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5, font=dict(size=11)),
        height=520,
    )

    return fig
