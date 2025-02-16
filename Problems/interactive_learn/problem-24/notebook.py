# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy==2.2.2",
#     "pandas==2.2.3",
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
        # Understanding Single Neuron with Sigmoid Activation

        The single neuron is the fundamental building block of neural networks. It's a simple yet powerful computational unit that can perform binary classification by processing multidimensional input features through a sigmoid activation function. Let's explore this interactively!
        """
    ).center()
    return


@app.cell(hide_code=True)
def _(mo):
    # math definition accordion
    definition = mo.md(r"""
    A single neuron processes input features through these key steps:

    1. **Weighted Sum Calculation:**

    \[
    z = \sum_{i=1}^{n} (w_i \times feature_i) + bias
    \]

    2. **Sigmoid Activation Function:**

    \[
    \sigma(z) = \frac{1}{1 + e^{-z}}
    \]

    3. **Mean Squared Error:**

    \[
    MSE = \frac{1}{n}\sum_{i=1}^{n} (predicted_i - true_i)^2
    \]

    where $w_i$ are weights, $feature_i$ are input features, and $n$ is the number of features.
    """)

    mo.accordion({"### Mathematical Formulation": definition})
    return (definition,)


@app.cell
def _(insights):
    insights
    return


@app.cell(hide_code=True)
def _(mo):
    # insights accordion
    insights = mo.accordion({
        "🧠 Understanding the Components": mo.md("""
        **Key Concepts:**

        1. **Input Features**: Multiple numerical values representing different aspects of the data
        2. **Weights**: Learned parameters that determine feature importance
        3. **Bias**: Offset term that helps the model fit the data better
        4. **Sigmoid Function**: Squashes input to probability between 0 and 1
        """),

        "📊 Model Behavior": mo.md("""
        The neuron's behavior is influenced by:

        1. **Feature Values**: Input data characteristics
        2. **Weight Magnitudes**: Control feature influence
        3. **Bias Term**: Shifts the decision boundary
        4. **Sigmoid Activation**: Creates non-linear output
        """)
    })
    return (insights,)


@app.cell
def _(feature_inputs):
    feature_inputs
    return


@app.cell(hide_code=True)
def _(mo):
    # input controls: features
    features_1 = mo.ui.array([
        mo.ui.number(value=0.5, label="Feature 1", step=0.1),
        mo.ui.number(value=1.0, label="Feature 2", step=0.1)
    ])

    features_2 = mo.ui.array([
        mo.ui.number(value=-1.5, label="Feature 1", step=0.1),
        mo.ui.number(value=-2.0, label="Feature 2", step=0.1)
    ])

    features_3 = mo.ui.array([
        mo.ui.number(value=2.0, label="Feature 1", step=0.1),
        mo.ui.number(value=1.5, label="Feature 2", step=0.1)
    ])

    feature_inputs = mo.vstack([
        mo.md("### Input Features"),
        mo.hstack([
            mo.vstack([
                mo.md("**Example 1**"),
                features_1
            ]),
            mo.vstack([
                mo.md("**Example 2**"),
                features_2
            ]),
            mo.vstack([
                mo.md("**Example 3**"),
                features_3
            ])
        ])
    ])
    return feature_inputs, features_1, features_2, features_3


@app.cell
def _(weight_controls):
    weight_controls
    return


@app.cell(hide_code=True)
def _(mo):
    # Weight and bias (wandb) controls
    w1 = mo.ui.number(
        value=0.7,
        label="Weight 1",
        step=0.1
    )

    w2 = mo.ui.number(
        value=-0.4,
        label="Weight 2",
        step=0.1
    )

    bias = mo.ui.number(
        value=-0.1,
        label="Bias",
        step=0.1
    )

    weight_controls = mo.vstack([
        mo.md("### Model Parameters"),
        mo.hstack([w1, w2, bias]),
        mo.callout(
            mo.md("""
            Adjust weights and bias to see how they affect the neuron's output:

            - Positive weights increase output for positive features

            - Negative weights increase output for negative features

            - Bias shifts the decision boundary
            """),
            kind="info"
        )
    ])
    return bias, w1, w2, weight_controls


@app.cell
def _(calculate_button):
    calculate_button
    return


@app.cell
def _(mo):
    calculate_button = mo.ui.run_button(label="Calculate Neuron Output")
    return (calculate_button,)


@app.cell
def _(results_display):
    results_display
    return


@app.cell(hide_code=True)
def _(
    bias,
    calculate_button,
    features_1,
    features_2,
    features_3,
    mo,
    np,
    w1,
    w2,
):
    def sigmoid(z):
            return 1 / (1 + np.exp(-z))

    def compute_neuron_output(features, weights, bias):
        z = np.dot(features, weights) + bias
        return sigmoid(z)

    results = []
    mse = 0

    if calculate_button.value:
        # Convert inputs to numpy arrays
        weights = np.array([w1.value, w2.value])
        true_labels = np.array([0, 1, 0])

        # Process each feature vector
        feature_vectors = [
            [f.value for f in features_1.elements],
            [f.value for f in features_2.elements],
            [f.value for f in features_3.elements]
        ]

        predictions = []
        for features in feature_vectors:
            pred = compute_neuron_output(features, weights, bias.value)
            predictions.append(round(float(pred), 4))

        # Calculate MSE
        mse = round(np.mean((np.array(predictions) - true_labels) ** 2), 4)
        results = predictions

    results_display = mo.vstack([
        mo.md(f"### Results"),
        mo.md(f"**Predictions:** {results}"),
        mo.md(f"**Mean Squared Error:** {mse}"),
        mo.accordion({
            "🔍 Understanding the Results": mo.md(f"""
            - Predictions close to 0 indicate class 0
            - Predictions close to 1 indicate class 1
            - Lower MSE means better model performance
            """)
        })
    ])
    return (
        compute_neuron_output,
        feature_vectors,
        features,
        mse,
        pred,
        predictions,
        results,
        results_display,
        sigmoid,
        true_labels,
        weights,
    )


@app.cell
def _(visualize_button):
    visualize_button
    return


@app.cell(hide_code=True)
def _(mo):
    visualize_button = mo.ui.run_button(label="Visualize Decision Boundary")
    return (visualize_button,)


@app.cell(hide_code=True)
def _(bias, features_1, features_2, features_3, np, pd, px, w1, w2):
    # plotting function
    def plot_decision_boundary():
        # meshgrid for visualization
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)

        # Z values calculated using the neuron model
        Z = 1 / (1 + np.exp(-(w1.value * X + w2.value * Y + bias.value)))

        df = pd.DataFrame({
            'x': X.flatten(),
            'y': Y.flatten(),
            'z': Z.flatten()
        })

        # Plot decision boundary using density contour
        fig = px.density_contour(
            df, x='x', y='y', z='z',
            title='Decision Boundary Visualization',
            labels={'x': 'Feature 1', 'y': 'Feature 2', 'z': 'Probability'}
        )

        # Add scatter points for input examples
        feature_points = pd.DataFrame({
            'Feature 1': [
                features_1.elements[0].value,
                features_2.elements[0].value,
                features_3.elements[0].value
            ],
            'Feature 2': [
                features_1.elements[1].value,
                features_2.elements[1].value,
                features_3.elements[1].value
            ],
            'Label': ['Example 1', 'Example 2', 'Example 3']
        })

        scatter = px.scatter(
            feature_points,
            x='Feature 1',
            y='Feature 2',
            text='Label'
        )

        for trace in scatter.data:
            fig.add_trace(trace)

        fig.update_layout(
            width=800,
            height=600
        )
        return fig
    return (plot_decision_boundary,)


@app.cell(hide_code=True)
def _(plot_decision_boundary, visualize_button):
    decision_boundary_plot = None
    if visualize_button.value:
        decision_boundary_plot = plot_decision_boundary()
    decision_boundary_plot
    return (decision_boundary_plot,)


@app.cell
def _(conclusion):
    conclusion
    return


@app.cell(hide_code=True)
def _(mo):
    # Conclusion and next steps
    conclusion = mo.vstack([
        mo.callout(
            mo.md("""
                **Congratulations!** 
                You've explored the single neuron model with sigmoid activation. You've learned:

                - How weights and bias affect the decision boundary
                - How sigmoid activation converts linear combinations to probabilities
                - How to evaluate performance using MSE
            """),
            kind="success"
        ),
        mo.accordion({
            "🎯 Applications": mo.md("""
                - Binary classification tasks
                - Feature importance analysis
                - Basic pattern recognition
                - Building blocks for larger neural networks
            """),
            "🚀 Next Steps": mo.md("""
                1. Implement the neuron in a real project
                2. Explore other activation functions
                3. Add more features to handle complex patterns
                4. Learn about gradient descent for training
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
    import pandas as pd
    import plotly.express as px
    return np, pd, px


if __name__ == "__main__":
    app.run()
