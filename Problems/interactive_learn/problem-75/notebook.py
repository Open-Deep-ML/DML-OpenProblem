# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.2.2",
#     "plotly==6.0.0",
# ]
# ///

import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Understanding Confusion Matrices in Binary Classification

        The [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix) is a fundamental tool for evaluating classification models. It provides a detailed breakdown of correct and incorrect predictions, helping us understand where our model succeeds and fails. Let's explore it interactively!
        """
    ).center()
    return


@app.cell(hide_code=True)
def _(mo):
    definition = mo.md(r"""
    For binary classification, the confusion matrix is a $2 \times 2$ matrix:

    \[
    M = \begin{pmatrix}
    \text{TP} & \text{FN} \\
    \text{FP} & \text{TN}
    \end{pmatrix}
    \]

    where:

    - TP (True Positives): Correctly predicted positive cases

    - FN (False Negatives): Incorrectly predicted negative cases

    - FP (False Positives): Incorrectly predicted positive cases

    - TN (True Negatives): Correctly predicted negative cases
    """)

    mo.accordion({"### Mathematical Definition": definition})
    return (definition,)


@app.cell
def _(flow, mo):
    mo.accordion({"Process Flow": flow.center()})
    return


@app.cell(hide_code=True)
def _(mo):
    # flowchart showing confusion matrix computation
    flow = mo.mermaid("""
    graph TD
        A[Input Data<br>y_true, y_pred] --> B[Count Predictions]
        B --> C[Organize in 2x2 Matrix]
        C --> D[Calculate Metrics]
        D --> E[Precision]
        D --> F[Recall]
        D --> G[Accuracy]
        D --> H[F1-Score]
    """)
    return (flow,)


@app.cell
def _(mo):
    mo.md(
        """
        ### Input Data
        Enter the actual and predicted classifications for each individual (0 for negative, 1 for positive).
        """
    )
    return


@app.cell
def _(data_controls):
    data_controls
    return


@app.cell
def _(mo):
    # Create number inputs for 12 individuals
    n_samples = 12
    actual_inputs = mo.ui.array([
        mo.ui.number(value=0, start=0, stop=1, label=f"Actual {i+1}")
        for i in range(n_samples)
    ])
    predicted_inputs = mo.ui.array([
        mo.ui.number(value=0, start=0, stop=1, label=f"Predicted {i+1}")
        for i in range(n_samples)
    ])

    # Create data table using markdown with LaTeX
    data_table = mo.md(r"""
    $$
    \begin{array}{|c|c|c|c|c|c|c|c|c|c|c|c|c|}
    \hline
    \text{Individual} & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 \\
    \hline
    \text{Actual} & a_1 & a_2 & a_3 & a_4 & a_5 & a_6 & a_7 & a_8 & a_9 & a_{10} & a_{11} & a_{12} \\
    \hline
    \text{Predicted} & p_1 & p_2 & p_3 & p_4 & p_5 & p_6 & p_7 & p_8 & p_9 & p_{10} & p_{11} & p_{12} \\
    \hline
    \end{array}
    $$
    """)

    # Stack inputs horizontally and data table below
    data_controls = mo.vstack([
        mo.hstack([
            mo.vstack([
                mo.md("**Actual Classifications:**"),
                actual_inputs
            ]),
            mo.vstack([
                mo.md("**Predicted Classifications:**"),
                predicted_inputs
            ])
        ], justify="start", align="start"),
        mo.md("### Data Table:"),
        data_table
    ], gap=2)  # Added gap for better spacing
    return (
        actual_inputs,
        data_controls,
        data_table,
        n_samples,
        predicted_inputs,
    )


@app.cell
def _(compute_button):
    compute_button.center()
    return


@app.cell
def _(mo):
    compute_button = mo.ui.run_button(label="Compute Confusion Matrix")
    return (compute_button,)


@app.cell
def _(
    actual_inputs,
    compute_button,
    explanation,
    mo,
    np,
    predicted_inputs,
    px,
):
    results = None
    if compute_button.value:
        # get data from inputs
        actual_values = np.array([inp.value for inp in actual_inputs])
        predicted_values = np.array([inp.value for inp in predicted_inputs])

        #  results for each individual
        results_array = []
        for actual, pred in zip(actual_values, predicted_values):
            if actual == 1 and pred == 1:
                result = "TP"
            elif actual == 1 and pred == 0:
                result = "FN"
            elif actual == 0 and pred == 1:
                result = "FP"
            else:
                result = "TN"
            results_array.append(result)

        # confusion matrix calc
        tp = sum(1 for r in results_array if r == "TP")
        fn = sum(1 for r in results_array if r == "FN")
        fp = sum(1 for r in results_array if r == "FP")
        tn = sum(1 for r in results_array if r == "TN")

        conf_matrix = np.array([[tp, fn], [fp, tn]])
        total = tp + fn + fp + tn

        # performance metrics calc
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # results table using markdown with LaTeX
        results_table = mo.md(r"""
        $$
        \begin{array}{|c|c|c|c|c|c|c|c|c|c|c|c|c|}
        \hline
        \text{Individual} & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 \\
        \hline
        \text{Actual} & """ + " & ".join(str(v) for v in actual_values) + r""" \\
        \hline
        \text{Predicted} & """ + " & ".join(str(v) for v in predicted_values) + r""" \\
        \hline
        \text{Result} & """ + " & ".join(results_array) + r""" \\
        \hline
        \end{array}
        $$
        """)

        # Create confusion matrix visualization
        fig = px.imshow(
            conf_matrix,
            labels=dict(x="Predicted", y="Actual"),
            x=['Positive', 'Negative'],
            y=['Positive', 'Negative'],
            aspect="auto",
            title="Confusion Matrix Heatmap",
            color_continuous_scale="RdBu",
            width=500,
            height=500,
            text_auto=True
        )

        fig.update_traces(
            texttemplate="%{z}",
            textfont={"size": 20},
            hoverongaps=False,
            hovertemplate="<br>".join([
                "Actual: %{y}",
                "Predicted: %{x}",
                "Count: %{z}",
                "<extra></extra>"
            ])
        )

        # matrix interpretation
        matrix_interpretation = mo.md(f"""
        ### Matrix Interpretation

        - True Positives (TP): {tp} (Actual: Positive, Predicted: Positive)

        - False Negatives (FN): {fn} (Actual: Positive, Predicted: Negative)

        - False Positives (FP): {fp} (Actual: Negative, Predicted: Positive)

        - True Negatives (TN): {tn} (Actual: Negative, Predicted: Negative)

        **Metrics:**

        - Accuracy: {accuracy:.2f}

        - Precision: {precision:.2f}

        - Recall: {recall:.2f}

        - F1 Score: {f1:.2f}
        """)

        results = mo.vstack([
            mo.md("### Results"),
            results_table,
            # confusion matrix and interpretation side-by-side
            mo.hstack([
                fig,
                matrix_interpretation
            ], justify="start", align="start"),
            explanation,
            # final callout
            mo.callout(
                mo.md("""
                🎉 Congratulations! You've successfully:

                - Understood how confusion matrices work in binary classification

                - Learned to interpret TP, FN, FP, and TN

                - Explored key metrics like accuracy, precision, recall, and F1 score

                - Gained hands-on experience with interactive confusion matrix analysis
                """),
                kind="success"
                )
        ])
    results
    return (
        accuracy,
        actual,
        actual_values,
        conf_matrix,
        f1,
        fig,
        fn,
        fp,
        matrix_interpretation,
        precision,
        pred,
        predicted_values,
        recall,
        result,
        results,
        results_array,
        results_table,
        tn,
        total,
        tp,
    )


@app.cell(hide_code=True)
def _(mo):
    explanation = mo.accordion({
        "🎯 Understanding the Results": mo.md("""
        **Interpreting the Confusion Matrix:**

        1. **Top-left (TP)**: Correctly identified positive cases
        2. **Top-right (FN)**: Missed positive cases
        3. **Bottom-left (FP)**: False alarms
        4. **Bottom-right (TN)**: Correctly identified negative cases
        """),

        "📊 Derived Metrics": mo.md("""
        - **Accuracy**: Overall correctness (TP + TN) / Total
        - **Precision**: Positive predictive value TP / (TP + FP)
        - **Recall**: True positive rate TP / (TP + FN)
        - **F1 Score**: Harmonic mean of precision and recall
        """),

        "💡 Best Practices": mo.md("""
        1. Consider class imbalance
        2. Look at all metrics, not just accuracy
        3. Choose metrics based on your problem context
        4. Use confusion matrix for model debugging
        """)
    })
    return (explanation,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import plotly.express as px
    return np, px


if __name__ == "__main__":
    app.run()
