"""
xai.py — Explainability utilities for EV-BMS Dashboard (White Neumorphism).

9 SHAP / XAI visualisations with light-themed matplotlib plots:
    1. Global Feature Importance
    2. SHAP Distribution Plot
    3. SHAP Heatmap
    4. Temperature Dependence Plot
    5. Cycle Influence Plot
    6. Current Influence Plot
    7. RL Action Influence Plot
    8. Feature Ranking Plot
    9. Combined GRU + RL XAI Figure

Handles SHAP shapes: (samples, features, actions) → slice by chosen action.
"""

import numpy as np
import matplotlib
import streamlit as st
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# White neumorphism palette for plots
# ---------------------------------------------------------------------------
BG      = "#ecf0f3"
CARD    = "#f7f9fb"
TXT     = "#333333"
MUTED   = "#64748b"
PRIMARY = "#3b82f6"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"
DANGER  = "#ef4444"
GRID    = "#d1d5db"

RL_FEATURES  = ["SoH", "Temp", "Cycle", "Current"]
GRU_FEATURES = ["IR", "QC", "QD", "Tavg", "Tmax", "ChargeTime"]
ACTIONS      = ["Decrease", "Maintain", "Increase"]
ACT_COLORS   = [DANGER, WARNING, SUCCESS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _light_fig(figsize=(7, 4)):
    """Create a matplotlib figure with the white neumorphism theme."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=TXT, labelsize=9)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    ax.title.set_color(TXT)
    return fig, ax

def _fin(fig):
    fig.tight_layout()
    return fig


def _safe_shap_for_action(shap_values, chosen_action):
    """Extract SHAP values for chosen action, handling various shapes."""
    if isinstance(shap_values, list):
        sv = np.array(shap_values[chosen_action])
        return sv[0] if sv.ndim >= 2 else sv
    sv = np.array(shap_values)
    if sv.ndim == 3:
        return sv[0, :, chosen_action]
    elif sv.ndim == 2:
        return sv[0]
    return sv


def _all_actions_shap(shap_values):
    """Return (3, num_features) matrix of SHAP values across actions."""
    if isinstance(shap_values, list):
        return np.array([sv[0] for sv in shap_values])
    sv = np.array(shap_values)
    if sv.ndim == 3:
        return sv[0].T
    return np.stack([sv[0]] * 3)


# ---------------------------------------------------------------------------
# Core SHAP Computation
# ---------------------------------------------------------------------------
def compute_shap_values(_dqn_model, state, feature_names=None):
    import shap
    if feature_names is None:
        feature_names = RL_FEATURES

    def predict_fn(x):
        return _dqn_model(x, training=False).numpy()

    background = np.array([[1.0, 25.0, 100.0, 50.0]])
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(state, silent=True)

    q_values = predict_fn(state)[0]
    chosen_action = int(np.argmax(q_values))

    if not isinstance(shap_values, list):
        sv = np.array(shap_values)
        if sv.ndim == 3:
            shap_values = [sv[:, :, i] for i in range(sv.shape[2])]
        else:
            shap_values = [shap_values]

    return shap_values, chosen_action, q_values


# ---------------------------------------------------------------------------
# 1. Global Feature Importance
# ---------------------------------------------------------------------------
def generate_feature_importance_plot(shap_values, feature_names=None, chosen_action=0):
    if feature_names is None:
        feature_names = RL_FEATURES
    vals = np.abs(_safe_shap_for_action(shap_values, chosen_action))
    order = np.argsort(vals)

    fig, ax = _light_fig((7, 3.5))
    colors = [PRIMARY, "#6366f1", "#8b5cf6", "#a78bfa"]
    ax.barh(np.array(feature_names)[order], vals[order],
            color=[colors[i % 4] for i in order], edgecolor="#c5cbd3", lw=0.5)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
    ax.set_title("Global Feature Importance", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.2, color=GRID)
    return _fin(fig)


# ---------------------------------------------------------------------------
# 2. SHAP Distribution Plot
# ---------------------------------------------------------------------------
def generate_shap_distribution_plot(shap_values, state, feature_names=None, chosen_action=0):
    if feature_names is None:
        feature_names = RL_FEATURES
    sv = _safe_shap_for_action(shap_values, chosen_action)
    order = np.argsort(np.abs(sv))[::-1]

    fig, ax = _light_fig((7, 3.5))
    colors = [SUCCESS if v >= 0 else DANGER for v in sv[order]]
    y = np.arange(len(feature_names))
    ax.barh(y, sv[order], color=colors, edgecolor="#c5cbd3", lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{feature_names[i]} = {state[0][i]:.2f}" for i in order],
                       fontsize=9, color=TXT)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP Value", fontsize=10)
    ax.set_title("SHAP Distribution Plot", fontsize=13, fontweight="bold")
    ax.axvline(0, color=GRID, lw=0.8, ls="--")
    ax.grid(axis="x", alpha=0.15, color=GRID)
    return _fin(fig)


# ---------------------------------------------------------------------------
# 3. SHAP Heatmap
# ---------------------------------------------------------------------------
def generate_shap_heatmap(shap_values, feature_names=None):
    if feature_names is None:
        feature_names = RL_FEATURES
    matrix = _all_actions_shap(shap_values)

    fig, ax = _light_fig((7, 3.5))
    cmap = LinearSegmentedColormap.from_list("c", [DANGER, "#fef3c7", SUCCESS])
    vmax = max(abs(matrix.min()), abs(matrix.max())) or 1.0
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, fontsize=9, color=TXT)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(ACTIONS, fontsize=9, color=TXT)

    for i in range(3):
        for j in range(len(feature_names)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    color=TXT, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=TXT)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TXT, fontsize=8)
    ax.set_title("SHAP Heatmap (All Actions)", fontsize=13, fontweight="bold")
    return _fin(fig)


# ---------------------------------------------------------------------------
# 4–6. Dependence Plots
# ---------------------------------------------------------------------------
def _dependence(_dqn_model, state, idx, name, rng, n=30):
    xs = np.linspace(rng[0], rng[1], n)
    curves = {a: [] for a in range(3)}
    for v in xs:
        s = state.copy(); s[0, idx] = v
        q = _dqn_model(s, training=False).numpy()[0]
        for a in range(3):
            curves[a].append(q[a])

    fig, ax = _light_fig((7, 3.5))
    for a in range(3):
        ax.plot(xs, curves[a], color=ACT_COLORS[a], lw=2.5, label=ACTIONS[a], alpha=0.85)
    ax.axvline(state[0, idx], color=PRIMARY, ls="--", lw=1.5,
               label=f"Current ({state[0, idx]:.1f})")
    ax.set_xlabel(name, fontsize=10); ax.set_ylabel("Q-Value", fontsize=10)
    ax.set_title(f"{name} Dependence", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, facecolor=CARD, edgecolor=GRID, labelcolor=TXT, loc="best")
    ax.grid(True, alpha=0.2, color=GRID)
    return _fin(fig)

def generate_temperature_dependence(dqn_model, state):
    return _dependence(dqn_model, state, 1, "Temperature (°C)", (5, 55))

def generate_cycle_dependence(dqn_model, state):
    return _dependence(dqn_model, state, 2, "Cycle Count", (1, 2000))

def generate_current_dependence(dqn_model, state):
    return _dependence(dqn_model, state, 3, "Charging Current (A)", (0, 150))


# ---------------------------------------------------------------------------
# 7. RL Action Influence Plot
# ---------------------------------------------------------------------------
def generate_action_influence_plot(shap_values, feature_names=None):
    if feature_names is None:
        feature_names = RL_FEATURES
    matrix = _all_actions_shap(shap_values)
    x = np.arange(len(feature_names))
    w = 0.25

    fig, ax = _light_fig((7, 3.5))
    for a in range(3):
        ax.bar(x + a * w, matrix[a], w, color=ACT_COLORS[a],
               label=ACTIONS[a], edgecolor="#c5cbd3", lw=0.5)
    ax.set_xticks(x + w)
    ax.set_xticklabels(feature_names, fontsize=9, color=TXT)
    ax.set_ylabel("SHAP Value", fontsize=10)
    ax.set_title("RL Action Influence per Feature", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, facecolor=CARD, edgecolor=GRID, labelcolor=TXT)
    ax.axhline(0, color=GRID, lw=0.6, ls="--")
    ax.grid(axis="y", alpha=0.15, color=GRID)
    return _fin(fig)


# ---------------------------------------------------------------------------
# 8. Feature Ranking Plot
# ---------------------------------------------------------------------------
def generate_feature_ranking_plot(shap_values, feature_names=None):
    if feature_names is None:
        feature_names = RL_FEATURES
    matrix = np.abs(_all_actions_shap(shap_values))
    agg = matrix.mean(axis=0)
    order = np.argsort(agg)[::-1]

    fig, ax = _light_fig((7, 3.5))
    ranks = np.arange(len(feature_names))
    colors = [PRIMARY, "#6366f1", "#8b5cf6", "#a78bfa"]
    ax.barh(ranks, agg[order],
            color=[colors[i % 4] for i in range(len(feature_names))],
            edgecolor="#c5cbd3", lw=0.5)
    ax.set_yticks(ranks)
    ax.set_yticklabels(np.array(feature_names)[order], fontsize=10, color=TXT)
    ax.invert_yaxis()
    ax.set_xlabel("Aggregate |SHAP| (all actions)", fontsize=10)
    ax.set_title("Feature Ranking", fontsize=13, fontweight="bold")
    for i, v in enumerate(agg[order]):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9, color=MUTED)
    ax.grid(axis="x", alpha=0.15, color=GRID)
    return _fin(fig)


# ---------------------------------------------------------------------------
# 9. Combined GRU + RL XAI Figure
# ---------------------------------------------------------------------------
def generate_combined_xai_plot(shap_values, state, gru_inputs, chosen_action=0):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(CARD)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=TXT, labelsize=9)
        ax.xaxis.label.set_color(TXT)
        ax.yaxis.label.set_color(TXT)
        ax.title.set_color(TXT)

    # Left: GRU input magnitudes
    gru_vals = np.array(gru_inputs).flatten()
    gru_norm = gru_vals / gru_vals.max() if gru_vals.max() > 0 else gru_vals
    y1 = np.arange(len(GRU_FEATURES))
    ax1.barh(y1, gru_norm, color=PRIMARY, edgecolor="#c5cbd3", lw=0.5, alpha=0.8)
    ax1.set_yticks(y1)
    ax1.set_yticklabels(GRU_FEATURES, fontsize=9, color=TXT)
    ax1.invert_yaxis()
    ax1.set_xlabel("Normalised Magnitude", fontsize=10)
    ax1.set_title("GRU Input Features", fontsize=12, fontweight="bold")
    for i, (nv, rv) in enumerate(zip(gru_norm, gru_vals)):
        ax1.text(nv + 0.02, i, f"{rv:.2f}", va="center", fontsize=8, color=MUTED)
    ax1.grid(axis="x", alpha=0.15, color=GRID)

    # Right: RL SHAP
    sv = _safe_shap_for_action(shap_values, chosen_action)
    y2 = np.arange(len(RL_FEATURES))
    cols = [SUCCESS if v >= 0 else DANGER for v in sv]
    ax2.barh(y2, sv, color=cols, edgecolor="#c5cbd3", lw=0.5)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(RL_FEATURES, fontsize=9, color=TXT)
    ax2.invert_yaxis()
    ax2.set_xlabel("SHAP Value", fontsize=10)
    ax2.set_title("RL Decision SHAP", fontsize=12, fontweight="bold")
    ax2.axvline(0, color=GRID, lw=0.8, ls="--")
    ax2.grid(axis="x", alpha=0.15, color=GRID)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Textual Reasoning
# ---------------------------------------------------------------------------
def generate_reasoning_text(action, soh, temp, cycle, current):
    action_map = {0: "Decrease Charging", 1: "Maintain Charging", 2: "Increase Charging"}
    action_str = action_map.get(action, "Unknown")
    reasons = []

    if action == 0:
        if temp > 35:
            reasons.append(f"Battery temperature is elevated ({temp:.1f}°C), requiring "
                           "reduced current to prevent thermal runaway.")
        if soh < 0.70:
            reasons.append(f"Calibrated SoH is critically low ({soh:.2f}), indicating "
                           "severe degradation vulnerability.")
        if cycle > 500:
            reasons.append(f"High cycle count ({cycle}) indicates significant aging and "
                           "increased degradation risk.")
        if current > 80:
            reasons.append(f"Charging current ({current:.1f}A) is high; reducing it "
                           "extends remaining useful life.")
        if not reasons:
            reasons.append("The controller decreases charging current because battery "
                           "cycle count is high and temperature conditions indicate "
                           "potential degradation risk.")
    elif action == 1:
        if 20 <= temp <= 35:
            reasons.append(f"Battery temperature ({temp:.1f}°C) is within the optimal "
                           "operating window.")
        if soh >= 0.70:
            reasons.append(f"Calibrated SoH ({soh:.2f}) supports current charging load.")
        if not reasons:
            reasons.append("The current charging regime is optimal for minimising "
                           "degradation while maintaining performance.")
    elif action == 2:
        if temp < 30:
            reasons.append(f"Battery temperature ({temp:.1f}°C) is cool enough to "
                           "safely accept a higher charge rate.")
        if soh > 0.80:
            reasons.append(f"Battery health is strong ({soh:.2f}), enabling faster charging.")
        if current < 40:
            reasons.append(f"Charging current ({current:.1f}A) is low and can be "
                           "safely increased.")
        if not reasons:
            reasons.append("Battery parameters are under safe thresholds, allowing "
                           "accelerated charging speed.")

    text = f"<strong>Decision: {action_str}</strong><br><br>\n"
    text += f"The AI controller recommended to <strong>{action_str.lower()}</strong> based on:<ul>\n"
    for r in reasons:
        text += f"<li>{r}</li>\n"
    text += "</ul>"
    return text
