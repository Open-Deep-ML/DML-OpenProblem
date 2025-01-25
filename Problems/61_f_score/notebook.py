# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.2.1",
#     "plotly==5.24.1",
#     "scikit-learn==0.24.2"
# ]
# ///

import marimo

__generated_with = "0.10.17"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Understanding F-Score in Binary Classification

        The F-Score is a crucial metric in machine learning that provides a balanced measure of a model's predictive performance by combining precision and recall.
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


@app.cell
def _(beta_input, mo, pred_labels, true_labels):
    mo.hstack([true_labels, pred_labels, beta_input])
    return


@app.cell(hide_code=True)
def _(mo):
    # Interactive input UI elements for classification data
    true_labels = mo.ui.text_area(
        label="True Labels (comma-separated 0s and 1s)",
        value="1, 0, 1, 1, 0, 1",
        placeholder="Enter binary labels"
    )

    pred_labels = mo.ui.text_area(
        label="Predicted Labels (comma-separated 0s and 1s)",
        value="1, 0, 1, 0, 0, 1",
        placeholder="Enter predicted labels"
    )

    beta_input = mo.ui.number(
        label="β (Beta) Value",
        value=1.0,
        step=0.1,
        start=0.1,
        stop=10
    )
    return beta_input, pred_labels, true_labels


@app.cell
def _(calculate_button):
    calculate_button
    return


@app.cell
def _(confusion_matrix, mo, np):
    # F-Score calculation function
    def f_score(y_true, y_pred, beta):
        """
        Calculate F-Beta Score for binary classification

        Args:
            y_true (str): True binary labels as comma-separated string
            y_pred (str): Predicted binary labels as comma-separated string
            beta (float): Beta value for F-Score

        Returns:
            float: F-Beta Score rounded to 3 decimal places
        """
        try:
            # Convert string inputs to numpy array, handling potential whitespace
            y_true = np.array([int(x.strip()) for x in y_true.split(',')])
            y_pred = np.array([int(x.strip()) for x in y_pred.split(',')])

            # Ensure arrays are of equal length
            if len(y_true) != len(y_pred):
                raise ValueError("True and predicted label arrays must be of equal length")

            # Compute cf
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

            # Calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            # Calculate F-Beta Score
            if precision + recall == 0:
                return 0.0

            f_beta = ((1 + beta**2) * precision * recall) / ((beta**2 * precision) + recall)
            return round(f_beta, 3)
        
        except Exception as e:
            mo.md(f"**Error:** {str(e)}")
            return 0.0

    # Run button to calculate F-Score
    calculate_button = mo.ui.run_button(label="Calculate F-Score")
    return calculate_button, f_score


@app.cell
def _(beta_input, calculate_button, f_score, mo, pred_labels, true_labels):
    # Display F-Score result
    f_score_result = 0.0
    if calculate_button.value:
        f_score_result = f_score(true_labels.value, pred_labels.value, beta_input.value)

    result_display = mo.callout(
        mo.md(f"**F-Score Result:** {f_score_result}"),
        kind="info"
    )
    result_display
    return f_score_result, result_display


@app.cell
def _(visualize_button):
    visualize_button
    return


@app.cell
def _(confusion_matrix, go, mo, np, pred_labels, true_labels):
    # Visualization of classification results
    def create_confusion_matrix_plot():
        # Convert string inputs to numpy array
        y_true = np.array([int(x.strip()) for x in true_labels.value.split(',')])
        y_pred = np.array([int(x.strip()) for x in pred_labels.value.split(',')])

        # Compute cf
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Create cf visualization
        labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
        values = [tn, fp, fn, tp]

        fig = go.Figure(data=[go.Bar(
            x=labels, 
            y=values, 
            text=values, 
            textposition='auto'
        )])

        fig.update_layout(
            title='Confusion Matrix Breakdown',
            xaxis_title='Confusion Matrix Components',
            yaxis_title='Count'
        )

        return fig

    # Visualization button
    visualize_button = mo.ui.run_button(label="Visualize Confusion Matrix")
    return create_confusion_matrix_plot, visualize_button


@app.cell
def _(create_confusion_matrix_plot, visualize_button):
    # Render cf plot
    confusion_plot = None
    if visualize_button.value:
        confusion_plot = create_confusion_matrix_plot()
    confusion_plot
    return (confusion_plot,)


app._unparsable_cell(
    r"""
        conclusion
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    # Conclusion and insights
    conclusion = mo.vstack([
        mo.callout(
            mo.md("""
                **Key Insights about F-Score:**

                - Balances precision and recall
                - Useful for imbalanced datasets
                - β controls precision-recall trade-off
            """),
            kind="success"
        ),
        mo.accordion({
            "🔍 Practical Applications": mo.md("""
                - Medical Diagnosis
                - Spam Detection
                - Fraud Detection
                - Information Retrieval
            """),
            "🚀 Advanced Exploration": mo.md("""
                1. Experiment with different β values
                2. Try various label combinations
                3. Understand precision-recall trade-offs
            """)
        })
    ])
    return (conclusion,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import plotly.express as px
    import plotly.graph_objs as go
    from sklearn.metrics import confusion_matrix
    return confusion_matrix, go, np, px


if __name__ == "__main__":
    app.run()
