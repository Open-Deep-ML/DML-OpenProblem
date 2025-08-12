# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.2.4",
#     "plotly==5.24.1",
#     "scikit-learn==1.6.1",
#     "pandas==2.2.3",
# ]
# ///

import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Understanding F-Score in Binary Classification

        The F-Score is a crucial metric in machine learning that provides a balanced measure of a model's predictive performance by combining precision and recall. It's particularly valuable when dealing with imbalanced datasets.
        """
    ).center()
    return


@app.cell(hide_code=True)
def _(mo):
    # math definition accordion
    definition = mo.md(r"""
    The F-β Score is defined mathematically as:

    \[
    F_{\beta} = \frac{(1 + \beta^2) \cdot \text{precision} \cdot \text{recall}}{\beta^2 \cdot \text{precision} + \text{recall}}
    \]

    Key Components:

    - **Precision**: $\frac{\text{True Positives}}{\text{True Positives + False Positives}}$

    - **Recall**: $\frac{\text{True Positives}}{\text{True Positives + False Negatives}}$

    - **β**: Controls the balance between precision and recall

          - β = 1: F1-Score (balanced)

          - β > 1: More weight on recall

          - β < 1: More weight on precision
    """)

    mo.accordion({"### Mathematical Formulation": definition})
    return (definition,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    1. **Choose a scenario** from the dropdown or enter your own labels
    2. **Set the β value** to balance precision vs. recall
    3. **Calculate** → **Visualize** → **Compare** the results!
    """).callout(kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### How to Use This Notebook:""")
    return


@app.cell(hide_code=True)
def _(journey_diagram, mo):
    mo.mermaid(journey_diagram)
    return


@app.cell(hide_code=True)
def _(input_section):
    input_section
    return


@app.cell(hide_code=True)
def _(calculate_button):
    calculate_button.center()
    return


@app.cell(hide_code=True)
def _(beta_input, calculate_button, f_score, mo, pd, pred_values, true_values):
    # Display relevant results
    result_display = None
    error_message = None

    # Initialize result values as None (For display at the end)
    f_beta_score = None
    precision_val = None
    recall_val = None
    conf_matrix = None
    tn_val = fp_val = fn_val = tp_val = None
    specificity_val = accuracy_val = None

    if calculate_button.value:
        # Use the values from dropdown
        result = f_score(true_values, pred_values, beta_input.value)

        if len(result) == 5:  # Error occurred
            _, _, _, _, error_str = result
            error_message = mo.md(f"⚠️ **Error:** {error_str}").callout(kind="danger")
        else:
            f_beta_score, precision_val, recall_val, conf_matrix = result

            # Create metrics display appropriate formatting
            metrics_display = mo.hstack([
                mo.stat(f"F{beta_input.value:.1f} Score", f"{f_beta_score:.3f}"),
                mo.stat("Precision", f"{precision_val:.3f}"),
                mo.stat("Recall", f"{recall_val:.3f}"),
            ])

            # Extract values from CF
            tn_val, fp_val, fn_val, tp_val = conf_matrix.ravel()

            values_df = pd.DataFrame([
                [tn_val, fp_val],
                [fn_val, tp_val]
            ], index=["Actual: 0", "Actual: 1"], columns=["Predicted: 0", "Predicted: 1"])

            cm_table = mo.ui.table(values_df, selection=None)

            # additional derived metrics
            specificity_val = tn_val / (tn_val + fp_val) if (tn_val + fp_val) > 0 else 0
            accuracy_val = (tp_val + tn_val) / (tp_val + tn_val + fp_val + fn_val) if (tp_val + tn_val + fp_val + fn_val) > 0 else 0

            additional_metrics = mo.hstack([
                mo.stat("Accuracy", f"{accuracy_val:.3f}"),
                mo.stat("Specificity", f"{specificity_val:.3f}"),
            ])

            # interpretation based on values
            if precision_val > 0.7 and recall_val < 0.3:
                interpretation = "This model has good precision but poor recall."
            elif recall_val > 0.7 and precision_val < 0.3:
                interpretation = "This model has good recall but poor precision."
            elif abs(precision_val - recall_val) < 0.2:
                interpretation = "This model is balanced between precision and recall."
            elif precision_val < 0.5 and recall_val < 0.5:
                interpretation = "This model needs improvement in both precision and recall."
            else:
                interpretation = "This model performs reasonably well."

            # beta explanation
            if beta_input.value == 1:
                beta_explanation = "balanced"
            elif beta_input.value > 1:
                beta_explanation = "more focus on recall"
            else:
                beta_explanation = "more focus on precision"

            # all components with improved formatting
            result_display = mo.vstack([
                mo.md("## Results").center(),
                metrics_display,
                mo.md("### Confusion Matrix").center(),
                cm_table,
                additional_metrics,
                mo.md(f"""
                **Interpretation:**

                - A model with F{beta_input.value:.1f} Score of {f_beta_score:.3f} has {beta_explanation}.
                - {interpretation}
                """).callout(kind="info")
            ])

    # Return error message if there was an error, otherwise return result display
    output = error_message if error_message else result_display
    output
    return (
        accuracy_val,
        additional_metrics,
        beta_explanation,
        cm_table,
        conf_matrix,
        error_message,
        error_str,
        f_beta_score,
        fn_val,
        fp_val,
        interpretation,
        metrics_display,
        output,
        precision_val,
        recall_val,
        result,
        result_display,
        specificity_val,
        tn_val,
        tp_val,
        values_df,
    )


@app.cell(hide_code=True)
def _(confusion_matrix, go, mo, np, pd, pred_values, px, true_values):
    # classification results
    def create_visualizations():
        try:
            # string inputs to numpy array
            y_true_vis = np.array([int(x.strip()) for x in true_values.split(',')])
            y_pred_vis = np.array([int(x.strip()) for x in pred_values.split(',')])

            # arrays are of equal length
            if len(y_true_vis) != len(y_pred_vis):
                return mo.md("Error: True and predicted labels must have the same length").callout(kind="danger")

            # confusion matrix calc
            cm_vis = confusion_matrix(y_true_vis, y_pred_vis)
            if cm_vis.shape != (2, 2):
                # Handle case with missing classes
                if cm_vis.shape == (1, 1):
                    class_val = 0 if y_true_vis[0] == 0 else 1
                    expanded = np.zeros((2, 2))
                    expanded[class_val, class_val] = cm_vis[0, 0]
                    cm_vis = expanded
                else:
                    return mo.md("Error: Unexpected confusion matrix shape").callout(kind="danger")

            # Extract components
            tn_vis, fp_vis, fn_vis, tp_vis = cm_vis.ravel()

            # confusion matrix heatmap with improved styling
            z = np.array([[tn_vis, fp_vis], [fn_vis, tp_vis]])

            # text annotations with categories and values
            text_annotations = [
                [f'TN: {tn_vis}', f'FP: {fp_vis}'], 
                [f'FN: {fn_vis}', f'TP: {tp_vis}']
            ]

            heatmap = go.Figure(data=go.Heatmap(
                z=z,
                x=['Predicted: 0', 'Predicted: 1'],
                y=['Actual: 0', 'Actual: 1'],
                text=text_annotations,
                texttemplate="%{text}",
                colorscale='Blues'
            ))

            heatmap.update_layout(
                title='Confusion Matrix Heatmap',
                height=400,
                margin=dict(t=50, b=50, l=50, r=50),
                font=dict(size=12)
            )

            # Calculate precision and recall
            precision_vis = tp_vis / (tp_vis + fp_vis) if (tp_vis + fp_vis) > 0 else 0
            recall_vis = tp_vis / (tp_vis + fn_vis) if (tp_vis + fn_vis) > 0 else 0

            # precision-recall bar chart with decent styling
            metrics_bar = go.Figure(data=[go.Bar(
                x=['Precision', 'Recall'],
                y=[precision_vis, recall_vis],
                text=[f'{precision_vis:.3f}', f'{recall_vis:.3f}'],
                textposition='auto',
                marker_color=['rgba(255,99,132,0.7)', 'rgba(54,162,235,0.7)']
            )])

            metrics_bar.update_layout(
                title='Precision and Recall',
                yaxis_range=[0, 1],
                height=400,
                margin=dict(t=50, b=50, l=50, r=50)
            )

            # visual representation of predictions
            classifications = []
            for i in range(len(y_true_vis)):
                if y_true_vis[i] == 1 and y_pred_vis[i] == 1:
                    classifications.append('TP')
                elif y_true_vis[i] == 0 and y_pred_vis[i] == 0:
                    classifications.append('TN')
                elif y_true_vis[i] == 1 and y_pred_vis[i] == 0:
                    classifications.append('FN')
                else:  # y_true_vis[i] == 0 and y_pred_vis[i] == 1
                    classifications.append('FP')

            # DataFrame for visualization
            df_vis = pd.DataFrame({
                'Index': range(len(y_true_vis)),
                'True Label': y_true_vis,
                'Predicted Label': y_pred_vis,
                'Classification': classifications
            })

            # color map for classifications with improved colors
            colors = {
                'TP': 'rgba(99, 255, 132, 0.8)',  # Green
                'TN': 'rgba(99, 132, 255, 0.8)',  # Blue
                'FP': 'rgba(255, 99, 132, 0.8)',  # Red
                'FN': 'rgba(255, 206, 86, 0.8)'   # Yellow
            }

            # summary of classifications
            tp_count = classifications.count('TP')
            tn_count = classifications.count('TN')
            fp_count = classifications.count('FP')
            fn_count = classifications.count('FN')

            # improved scatter plot
            predictions_fig = px.scatter(
                df_vis, x='Index', y='True Label',
                color='Classification',
                symbol='Predicted Label',
                color_discrete_map=colors,
                title='Prediction Results',
                labels={'True Label': 'True Class', 'Predicted Label': 'Predicted Class'},
                height=400
            )

            predictions_fig.update_layout(
                margin=dict(t=50, b=50, l=50, r=50),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )

            # summary of counts
            classification_summary = mo.md(f"""
            **Classification Summary:**

            - **True Positives (TP):** {tp_count} (correctly predicted positives)
            - **True Negatives (TN):** {tn_count} (correctly predicted negatives)
            - **False Positives (FP):** {fp_count} (incorrectly predicted as positive)
            - **False Negatives (FN):** {fn_count} (incorrectly predicted as negative)

            **Overall Accuracy:** {(tp_count + tn_count) / len(classifications):.3f}
            """).callout(kind="info")

            # return the visualizations arranged in tabs with the summary
            return mo.vstack([
                mo.ui.tabs({
                    "Confusion Matrix": heatmap,
                    "Precision & Recall": metrics_bar,
                    "Predictions": predictions_fig
                }),
                classification_summary
            ])

        except Exception as e:
            return mo.md(f"Error generating visualizations: {str(e)}").callout(kind="danger")

    # visualization button
    visualize_button = mo.ui.run_button(label="Visualize Results")
    return create_visualizations, visualize_button


@app.cell(hide_code=True)
def _(visualize_button):
    visualize_button.center()
    return


@app.cell(hide_code=True)
def _(create_visualizations, visualize_button):
    # render visualizations
    visualizations = None
    if visualize_button.value:
        visualizations = create_visualizations()
    visualizations
    return (visualizations,)


@app.cell(hide_code=True)
def _(mo):
    # add interactive F-score comparison for different beta values
    mo.md("## F-Score for Different β Values").center()
    return


@app.cell(hide_code=True)
def _(compare_button):
    compare_button.center()
    return


@app.cell(hide_code=True)
def _(mo):
    @mo.cache
    def calculate_multiple_fscores(precision, recall, beta_values):
        """Calculate F-scores for multiple beta values"""
        f_scores = []
        for beta in beta_values:
            if precision + recall == 0:
                f_scores.append(0)
            else:
                f_beta = ((1 + beta**2) * precision * recall) / ((beta**2 * precision) + recall)
                f_scores.append(f_beta)
        return f_scores

    compare_button = mo.ui.run_button(label="Compare F-Scores")
    return calculate_multiple_fscores, compare_button


@app.cell(hide_code=True)
def _(
    calculate_multiple_fscores,
    compare_button,
    confusion_matrix,
    go,
    mo,
    np,
    pd,
    precision_val,
    pred_values,
    recall_val,
    true_values,
):
    comparison_result = None

    if compare_button.value:
        try:
            # if we already have precision and recall from previous calculations, use them
            if precision_val is not None and recall_val is not None:
                precision_comp = precision_val
                recall_comp = recall_val
            else:
                # otherwise, calculate them from the labels
                # string inputs to numpy array
                y_true_comp = np.array([int(x.strip()) for x in true_values.split(',')])
                y_pred_comp = np.array([int(x.strip()) for x in pred_values.split(',')])

                # confusion matrix calc
                cm_comp = confusion_matrix(y_true_comp, y_pred_comp)

                # handle case where confusion matrix is smaller than 2x2
                if cm_comp.shape != (2, 2):
                    if cm_comp.shape == (1, 1):
                        # only one class appears - need to expand matrix
                        if y_true_comp[0] == 0:  # only negatives
                            cm_expanded_comp = np.zeros((2, 2), dtype=int)
                            cm_expanded_comp[0, 0] = cm_comp[0, 0]  # TN
                            cm_comp = cm_expanded_comp
                        else:  # only positives
                            cm_expanded_comp = np.zeros((2, 2), dtype=int)
                            cm_expanded_comp[1, 1] = cm_comp[0, 0]  # TP
                            cm_comp = cm_expanded_comp
                    else:
                        raise ValueError("Unexpected confusion matrix shape")

                # extract components
                tn_comp, fp_comp, fn_comp, tp_comp = cm_comp.ravel()

                # precision and recall calc
                precision_comp = tp_comp / (tp_comp + fp_comp) if (tp_comp + fp_comp) > 0 else 0
                recall_comp = tp_comp / (tp_comp + fn_comp) if (tp_comp + fn_comp) > 0 else 0

                # F-scores calc for different beta values
                beta_values_comp = [0.5, 1.0, 2.0]
                beta_labels_comp = ['F0.5', 'F1', 'F2']

                f_scores_comp = calculate_multiple_fscores(precision_comp, recall_comp, beta_values_comp)

            # comparison chart with improved styling
            df_comp = pd.DataFrame({
                'Beta': beta_labels_comp,
                'F-Score': f_scores_comp,
                'Emphasis': ['Precision', 'Balanced', 'Recall']
            })

            fig_comp = go.Figure()

            # bars for F-scores with improved colors
            fig_comp.add_trace(go.Bar(
                x=df_comp['Beta'],
                y=df_comp['F-Score'],
                text=[f'{score:.3f}' for score in f_scores_comp],
                textposition='auto',
                marker_color=['rgba(255,165,0,0.7)', 'rgba(0,128,0,0.7)', 'rgba(0,0,255,0.7)']
            ))

            # reference lines for precision and recall
            fig_comp.add_trace(go.Scatter(
                x=df_comp['Beta'],
                y=[precision_comp] * len(beta_values_comp),
                mode='lines',
                name='Precision',
                line=dict(color='red', width=2, dash='dash')
            ))

            fig_comp.add_trace(go.Scatter(
                x=df_comp['Beta'],
                y=[recall_comp] * len(beta_values_comp),
                mode='lines',
                name='Recall',
                line=dict(color='blue', width=2, dash='dash')
            ))

            fig_comp.update_layout(
                title=f'F-Score Comparison (Precision: {precision_comp:.3f}, Recall: {recall_comp:.3f})',
                xaxis_title='F-Score Type',
                yaxis_title='Value',
                yaxis_range=[0, 1],
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=400,
                margin=dict(t=50, b=50, l=50, r=50)
            )

            # comparison text explaining differences with improved formatting
            if precision_comp > recall_comp:
                comparison_text = f"When precision ({precision_comp:.3f}) > recall ({recall_comp:.3f}), F0.5 > F2"
                recommendation = "higher precision"
            elif precision_comp < recall_comp:
                comparison_text = f"When precision ({precision_comp:.3f}) < recall ({recall_comp:.3f}), F0.5 < F2"
                recommendation = "higher recall"
            else:
                comparison_text = f"When precision ({precision_comp:.3f}) = recall ({recall_comp:.3f}), F0.5 = F2"
                recommendation = "balanced precision and recall"

            # explanation with improved formatting
            explanation_comp = mo.md(f"""
            **F-Score Comparison Explained:**

            - **F0.5 ({f_scores_comp[0]:.3f})**: Emphasizes precision over recall (good for spam detection)
            - **F1 ({f_scores_comp[1]:.3f})**: Balances precision and recall equally
            - **F2 ({f_scores_comp[2]:.3f})**: Emphasizes recall over precision (good for medical screening)

            **For your current data:**

            - {comparison_text}
            - This indicates that this model is better suited for tasks that require {recommendation}
            """).callout(kind="info")

            comparison_result = mo.vstack([fig_comp, explanation_comp])

        except Exception as e:
            comparison_result = mo.md(f"Error in F-score comparison: {str(e)}").callout(kind="danger")

    comparison_result
    return (
        beta_labels_comp,
        beta_values_comp,
        cm_comp,
        cm_expanded_comp,
        comparison_result,
        comparison_text,
        df_comp,
        explanation_comp,
        f_scores_comp,
        fig_comp,
        fn_comp,
        fp_comp,
        precision_comp,
        recall_comp,
        recommendation,
        tn_comp,
        tp_comp,
        y_pred_comp,
        y_true_comp,
    )


@app.cell(hide_code=True)
def _(mo):
    # conclusion and insights
    conclusion = mo.vstack([
        mo.md("## Key Takeaways").center(),
        mo.callout(
            mo.md("""
                **Key Insights about F-Score:**

                - Balances precision and recall according to your specific needs
                - F1-Score gives equal weight to precision and recall
                - F2-Score emphasizes finding all positive cases (high recall)
                - F0.5-Score emphasizes avoiding false positives (high precision)
            """),
            kind="success"
        ),
        mo.accordion({
            "🔍 Practical Applications": mo.md("""
                - **Medical Diagnosis**: Use F2-Score when missing a disease detection is costlier than a false alarm
                - **Spam Detection**: Use F0.5-Score when marking legitimate emails as spam is worse than missing some spam
                - **Fraud Detection**: Use F1-Score when both false alarms and missed frauds have similar costs
                - **Information Retrieval**: Use different β values based on user needs for precision vs. comprehensiveness
            """),
            "🚀 Advanced Exploration": mo.md("""
                1. Experiment with different β values to understand their impact
                2. Try various label combinations representing real-world scenarios
                3. Understand how class imbalance affects F-Score calculation
                4. Explore how different thresholds affect the precision-recall trade-off
                5. Compare F-Score with other metrics like accuracy, ROC-AUC, and PR-AUC
            """)
        })
    ])
    return (conclusion,)


@app.cell(hide_code=True)
def _(conclusion):
    conclusion
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Appendix code (containing helper functions and code)""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.metrics import confusion_matrix
    return confusion_matrix, go, np, pd, px


@app.cell(hide_code=True)
def _(mo):
    # Concept explanation accordion
    concept_explanation = mo.accordion({
        "🎯 When to Use F-Score": mo.md("""
        F-Score is especially valuable when:

        1. **Class Imbalance**: When one class significantly outnumbers another
        2. **Cost-sensitive Scenarios**: When false positives and false negatives have different costs
        3. **Domain-specific Requirements**: When either precision or recall is more important

        Common β values:
        - **F1 (β=1)**: Balanced weight between precision and recall
        - **F2 (β=2)**: Puts more emphasis on recall (finding all positives)
        - **F0.5 (β=0.5)**: Puts more emphasis on precision (avoiding false positives)
        """),

        "💡 Real-world Examples": mo.md("""
        - **Medical Diagnosis (β>1)**: When missing a disease detection (false negative) is more harmful than a false alarm
        - **Spam Detection (β<1)**: When marking legitimate emails as spam (false positive) is worse than missing some spam
        - **Fraud Detection (balanced)**: When both false alarms and missed frauds are equally problematic
        """)
    })
    return (concept_explanation,)


@app.cell(hide_code=True)
def _():
    # sampel scenario presets with detailed descriptions
    scenarios = {
        "Custom (Current)": {
            "true": "1, 0, 1, 1, 0, 1",
            "pred": "1, 0, 1, 0, 0, 1",
            "description": "Your custom input data"
        },
        "Perfect Prediction": {
            "true": "1, 0, 1, 0, 1, 0, 1, 0",
            "pred": "1, 0, 1, 0, 1, 0, 1, 0",
            "description": "Model perfectly predicts all labels (F-Score = 1.0)"
        },
        "All Positive Predictions": {
            "true": "1, 0, 1, 0, 1, 0, 1, 0",
            "pred": "1, 1, 1, 1, 1, 1, 1, 1",
            "description": "Model predicts everything as positive (high recall, low precision)"
        },
        "All Negative Predictions": {
            "true": "1, 0, 1, 0, 1, 0, 1, 0",
            "pred": "0, 0, 0, 0, 0, 0, 0, 0",
            "description": "Model predicts everything as negative (zero recall)"
        },
        "Imbalanced Dataset": {
            "true": "1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0",
            "pred": "1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0",
            "description": "Dataset with few positive examples (13% positive), showing how F-Score handles imbalance"
        }
    }
    return (scenarios,)


@app.cell(hide_code=True)
def _(mo, scenarios):
    # Selected scenario
    example_scenarios = mo.ui.dropdown(
        label="Load Example Scenario",
        options=list(scenarios.keys()),
        value="Custom (Current)"
    )
    return (example_scenarios,)


@app.cell(hide_code=True)
def _(example_scenarios, mo, scenarios):
    # set init values based on the default scenario
    selected = scenarios[example_scenarios.value]

    true_labels = mo.ui.text_area(
        label="True Labels (comma-separated 0s and 1s)",
        value=selected["true"],
        placeholder="Enter binary labels"
    )

    pred_labels = mo.ui.text_area(
        label="Predicted Labels (comma-separated 0s and 1s)",
        value=selected["pred"],
        placeholder="Enter predicted labels"
    )

    beta_input = mo.ui.number(
        label="β (Beta) Value",
        value=1.0,
        step=0.1,
        start=0.1,
        stop=10
    )
    return beta_input, pred_labels, selected, true_labels


@app.cell(hide_code=True)
def _(confusion_matrix, mo, np):
    # F-Score calc function
    def f_score(y_true, y_pred, beta):
        """
        Calculate F-Beta Score for binary classification

        Args:
            y_true (str): True binary labels as comma-separated string
            y_pred (str): Predicted binary labels as comma-separated string
            beta (float): Beta value for F-Score

        Returns:
            tuple: (F-Beta Score, precision, recall, confusion matrix) or (None, None, None, None, error_msg) on error
        """
        try:
            # string inputs to numpy array, handling potential whitespace
            y_true_arr = np.array([int(x.strip()) for x in y_true.split(',')])
            y_pred_arr = np.array([int(x.strip()) for x in y_pred.split(',')])

            # ensure arrays are of equal length
            if len(y_true_arr) != len(y_pred_arr):
                raise ValueError("True and predicted label arrays must be of equal length")

            # check that labels are binary (0s and 1s)
            for arr, name in [(y_true_arr, "True labels"), (y_pred_arr, "Predicted labels")]:
                invalid_vals = [x for x in arr if x not in [0, 1]]
                if invalid_vals:
                    raise ValueError(f"{name} must contain only 0s and 1s. Found: {invalid_vals}")

            # compute confusion matrix
            cm = confusion_matrix(y_true_arr, y_pred_arr)

            # handle case where confusion matrix is smaller than 2x2
            if cm.shape != (2, 2):
                # this happens when one class is missing in both true and predicted
                if cm.shape == (1, 1):
                    # only one class appears - need to expand matrix
                    if y_true_arr[0] == 0:  # only negatives
                        cm_expanded = np.zeros((2, 2), dtype=int)
                        cm_expanded[0, 0] = cm[0, 0]  # TN
                        cm = cm_expanded
                    else:  # only positives
                        cm_expanded = np.zeros((2, 2), dtype=int)
                        cm_expanded[1, 1] = cm[0, 0]  # TP
                        cm = cm_expanded
                else:
                    raise ValueError("Unexpected confusion matrix shape. Ensure both classes are represented.")

            # extract values from confusion matrix
            tn, fp, fn, tp = cm.ravel()

            # precision and recall calc
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            # F-Beta Score calc
            if precision + recall == 0:
                return 0.0, precision, recall, cm

            f_beta = ((1 + beta**2) * precision * recall) / ((beta**2 * precision) + recall)
            return round(f_beta, 3), round(precision, 3), round(recall, 3), cm

        except Exception as e:
            return None, None, None, None, str(e)

    # run button to calculate F-Score
    calculate_button = mo.ui.run_button(label="Calculate F-Score")
    return calculate_button, f_score


@app.cell(hide_code=True)
def _(beta_input, example_scenarios, mo, pred_labels, scenarios, true_labels):
    # when dropdown changes, show what data is being used
    selected_scenario = scenarios[example_scenarios.value]

    # display message about the selected scenario
    if example_scenarios.value == "Custom (Current)":
        # for custom data, show the actual values being used
        true_values = true_labels.value
        pred_values = pred_labels.value

        # count positives and negatives to give useful feedback
        try:
            true_arr = [int(x.strip()) for x in true_values.split(',')]
            pred_arr = [int(x.strip()) for x in pred_values.split(',')]

            true_pos_count = sum(true_arr)
            true_neg_count = len(true_arr) - true_pos_count
            pred_pos_count = sum(pred_arr)
            pred_neg_count = len(pred_arr) - pred_pos_count

            scenario_msg = mo.md(f"""
            **Current Data Summary:**

            - **True Labels:** {true_pos_count} positive, {true_neg_count} negative ({true_pos_count/len(true_arr)*100:.1f}% positive)
            - **Predicted Labels:** {pred_pos_count} positive, {pred_neg_count} negative ({pred_pos_count/len(pred_arr)*100:.1f}% positive)
            - **Total samples:** {len(true_arr)}
            """).callout(kind="info")
        except:
            # Fallback if there's an error parsing the input
            scenario_msg = mo.md("Using custom data from the input fields").callout(kind="info")
    else:
        # get values from the selected scenario
        true_values = selected_scenario["true"]
        pred_values = selected_scenario["pred"]

        # informative message about the scenario
        scenario_msg = mo.md(f"""
        **{example_scenarios.value}**

        {selected_scenario["description"]}

        _Labels have been loaded into the input fields above._
        """).callout(kind="info")

    # show interactive elements
    input_section = mo.vstack([
        mo.md("### Data Input and Parameters"),
        mo.hstack([true_labels, pred_labels], gap=2),
        mo.hstack([beta_input, example_scenarios], gap=2),
        scenario_msg
    ])
    return (
        input_section,
        pred_arr,
        pred_neg_count,
        pred_pos_count,
        pred_values,
        scenario_msg,
        selected_scenario,
        true_arr,
        true_neg_count,
        true_pos_count,
        true_values,
    )


@app.cell(hide_code=True)
def _():
    # Add user journey diagram showing how to use the notebook
    journey_diagram = """
    journey
      title 🎯 F-Score Adventure Map 🧭
      section 🧠 Learn
        Read intro: 5: Me
        Check math: 3: Me, Brain
      section 🔬 Play
        Pick example: 4: Me, 🐭
        Set β value: 3: Me, 🧪
        Calculate: 5: Me, 🤖
      section 📊 Explore
        Visualize: 4: Me, 👁️
        Compare F-scores: 5: Me, 🚀
    """
    return (journey_diagram,)


if __name__ == "__main__":
    app.run()
