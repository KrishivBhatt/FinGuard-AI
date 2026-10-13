"""
FinGuard AI - Visualizations Module
Creates interactive, highly styled Plotly charts for fintech risk analytics,
model performance evaluations, and spending breakdowns.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import config

# Cohesive dark fintech layout template
PLOTLY_TEMPLATE = "plotly_dark"
CHART_BG = "#0F172A"
CARD_BG = "#1E293B"
TEXT_COLOR = "#F8FAFC"
GRID_COLOR = "#334155"


def create_confusion_matrix_chart(cm_data: Dict[str, Any], model_name: str) -> go.Figure:
    """
    Renders an interactive annotated heatmap for a model's confusion matrix.
    """
    matrix = cm_data["matrix"]
    tn, fp = matrix[0]
    fn, tp = matrix[1]

    z = [[tn, fp], [fn, tp]]
    x = ["Predicted Legitimate (0)", "Predicted Fraud (1)"]
    y = ["Actual Legitimate (0)", "Actual Fraud (1)"]

    text = [
        [f"True Negatives (TN)<br><b>{tn:,}</b>", f"False Positives (FP)<br><b>{fp:,}</b>"],
        [f"False Negatives (FN)<br><b>{fn:,}</b>", f"True Positives (TP)<br><b>{tp:,}</b>"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white", "family": "Inter, sans-serif"},
        colorscale=[
            [0.0, "#1E293B"],
            [0.5, "#1E3A8A"],
            [1.0, "#3B82F6"]
        ],
        showscale=False
    ))

    fig.update_layout(
        title=f"<b>Confusion Matrix: {model_name}</b>",
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(side="bottom", gridcolor=GRID_COLOR),
        yaxis=dict(autorange="reversed", gridcolor=GRID_COLOR),
        height=380
    )
    return fig


def create_model_comparison_bar_chart(models_metrics: Dict[str, Any]) -> go.Figure:
    """
    Renders grouped bar chart comparing Accuracy, Precision, Recall, F1, and PR-AUC.
    """
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "pr_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "PR-AUC"]

    fig = go.Figure()

    colors = {
        "logistic_regression": "#60A5FA",  # Light blue
        "decision_tree": "#F59E0B",        # Amber
        "random_forest": "#10B981"         # Emerald
    }

    for key, data in models_metrics.items():
        name = data.get("model_name", key)
        values = [data.get(m, 0.0) for m in metrics_to_plot]
        color = colors.get(key, "#3B82F6")

        fig.add_trace(go.Bar(
            name=name,
            x=metric_labels,
            y=values,
            marker_color=color,
            text=[f"{v:.3f}" for v in values],
            textposition="auto",
            textfont=dict(size=11, family="Inter, sans-serif")
        ))

    fig.update_layout(
        title="<b>Multi-Model Performance Comparison</b>",
        barmode="group",
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        yaxis=dict(range=[0, 1.08], gridcolor=GRID_COLOR, title="Score (0 to 1.0)"),
        xaxis=dict(gridcolor=GRID_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=420
    )
    return fig


def create_pr_curve_chart(curves: Dict[str, Any], model_name: str, pr_auc: float) -> go.Figure:
    """
    Renders Precision-Recall Curve.
    """
    pr_data = curves.get("pr", {})
    precision = pr_data.get("precision", [1, 0])
    recall = pr_data.get("recall", [0, 1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode="lines",
        name=f"{model_name} (PR-AUC = {pr_auc:.3f})",
        line=dict(color="#3B82F6", width=3),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.15)"
    ))

    fig.update_layout(
        title=f"<b>Precision-Recall Curve: {model_name}</b>",
        xaxis=dict(title="Recall (Detection Rate)", range=[0, 1], gridcolor=GRID_COLOR),
        yaxis=dict(title="Precision (Accuracy of Fraud Flags)", range=[0, 1.05], gridcolor=GRID_COLOR),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380
    )
    return fig


def create_roc_curve_chart(curves: Dict[str, Any], model_name: str, roc_auc: float) -> go.Figure:
    """
    Renders ROC Curve with random guess baseline.
    """
    roc_data = curves.get("roc", {})
    fpr = roc_data.get("fpr", [0, 1])
    tpr = roc_data.get("tpr", [0, 1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"{model_name} (ROC-AUC = {roc_auc:.3f})",
        line=dict(color="#10B981", width=3),
        fill="tozeroy",
        fillcolor="rgba(16, 185, 129, 0.15)"
    ))

    # Baseline diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier Baseline",
        line=dict(color="#64748B", dash="dash", width=1.5)
    ))

    fig.update_layout(
        title=f"<b>Receiver Operating Characteristic (ROC): {model_name}</b>",
        xaxis=dict(title="False Positive Rate", range=[0, 1], gridcolor=GRID_COLOR),
        yaxis=dict(title="True Positive Rate (Recall)", range=[0, 1.05], gridcolor=GRID_COLOR),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380
    )
    return fig


def create_feature_importance_chart(model: Any, feature_names: List[str], top_n: int = 15) -> Optional[go.Figure]:
    """
    Renders horizontal bar chart of top N feature importances for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    top_features = [feature_names[i] for i in indices][::-1]
    top_scores = [importances[i] for i in indices][::-1]

    fig = go.Figure(go.Bar(
        x=top_scores,
        y=top_features,
        orientation="h",
        marker=dict(
            color=top_scores,
            colorscale=[[0, "#1E3A8A"], [0.5, "#3B82F6"], [1, "#10B981"]]
        ),
        text=[f"{v:.4f}" for v in top_scores],
        textposition="outside",
        textfont=dict(color="white", size=10)
    ))

    fig.update_layout(
        title=f"<b>Top {top_n} Most Influential Decision Features (Gini Importance)</b>",
        xaxis=dict(title="Normalized Gini Importance", gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        margin=dict(l=60, r=40, t=50, b=40),
        height=450
    )
    return fig


def create_spending_category_donut(df_categories: pd.DataFrame, currency_symbol: str = "₹") -> go.Figure:
    """
    Renders interactive donut chart for personal spending categories.
    """
    fig = go.Figure(data=[go.Pie(
        labels=df_categories["Category"],
        values=df_categories["Total_Amount"],
        hole=0.55,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Total: " + currency_symbol + "%{value:,.2f}<br>Share: %{percent}<extra></extra>",
        marker=dict(
            colors=px.colors.qualitative.Plotly
        )
    )])

    fig.update_layout(
        title="<b>Spending by Category</b>",
        paper_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
        margin=dict(l=20, r=20, t=50, b=20),
        height=380
    )
    return fig


def create_monthly_trend_line_chart(df_monthly: pd.DataFrame, currency_symbol: str = "₹") -> go.Figure:
    """
    Renders line/area chart for monthly spending trajectory.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_monthly["Month_Year"],
        y=df_monthly["Total_Amount"],
        mode="lines+markers",
        line=dict(color="#3B82F6", width=3),
        marker=dict(size=8, color="#60A5FA"),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.15)",
        hovertemplate="Month: %{x}<br>Spending: " + currency_symbol + "%{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Monthly Spending Trajectory</b>",
        xaxis=dict(title="Month", gridcolor=GRID_COLOR),
        yaxis=dict(title=f"Spending ({currency_symbol})", gridcolor=GRID_COLOR),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380
    )
    return fig


def create_amount_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Renders comparative distribution of transaction amounts (Legit vs Fraud).
    """
    if "Amount" not in df.columns or "Class" not in df.columns:
        return go.Figure()

    legit = df[df["Class"] == 0]["Amount"]
    fraud = df[df["Class"] == 1]["Amount"]

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=legit,
        name="Legitimate (0)",
        marker_color="#10B981",
        boxpoints="outliers"
    ))
    fig.add_trace(go.Box(
        y=fraud,
        name="Fraudulent (1)",
        marker_color="#EF4444",
        boxpoints="outliers"
    ))

    fig.update_layout(
        title="<b>Transaction Amount Distribution by Class</b>",
        yaxis=dict(title="Transaction Amount", gridcolor=GRID_COLOR, type="log"),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380
    )
    return fig
