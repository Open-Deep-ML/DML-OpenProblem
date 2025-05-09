# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.2.1",
#     "matplotlib==3.10.0",
#     "plotly==5.24.1",
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
        # Understanding Ridge Regression Loss Function

        [Ridge Regression](https://en.wikipedia.org/wiki/Ridge_regression) is a powerful regularized version of linear regression that helps prevent overfitting. Let's explore its loss function interactively!
        """
    ).center()
    return


@app.cell(hide_code=True)
def _(mo):
    # Mathematical definition accordion
    definition = mo.md(r"""
    The Ridge Regression loss function combines Mean Squared Error (MSE) with L2 regularization:

    \[
    L(\beta) = \underbrace{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}_{\text{MSE}} + \underbrace{\lambda \sum_{j=1}^p \beta_j^2}_{\text{L2 Regularization}}
    \]

    where:

    - $n$ is the number of samples

    - $y_i$ is the true value

    - $\hat{y}_i$ is the predicted value

    - $\lambda$ (alpha) is the regularization parameter

    - $\beta_j$ are the model coefficients
    """)

    mo.accordion({"### Mathematical Formulation": definition})
    return (definition,)


@app.cell(hide_code=True)
def _(mo):
    insights = mo.accordion({
        "🔍 Key Components": mo.md("""
        **1. Mean Squared Error (MSE)**
        - Measures prediction accuracy
        - Penalizes larger errors more heavily
        - Always non-negative

        **2. L2 Regularization Term**
        - Controls model complexity
        - Prevents coefficient values from becoming too large
        - Helps prevent overfitting
        """),

        "⚙️ Role of Alpha (λ)": mo.md("""
        The regularization parameter α controls:

        1. α = 0: Equivalent to standard linear regression
        2. Small α: Slight regularization effect
        3. Large α: Strong regularization, coefficients approach zero
        """),

        "🧮 Coefficient Shrinkage": mo.md("""
        Ridge regression shrinks coefficients by adding a penalty proportional to their squared magnitude:

        - Larger coefficients incur higher penalties
        - The penalty applies to all coefficients equally
        - Unlike Lasso, Ridge typically keeps all features but with reduced magnitudes
        - Mathematically, Ridge finds the minimum of: ||y - Xβ||² + λ||β||²
        """)
    })
    return (insights,)


@app.cell(hide_code=True)
def _(mo):
    # controls for sample data
    sample_size = mo.ui.slider(
        start=4,
        stop=20,
        value=10,
        step=1,
        label="Sample Size"
    )

    alpha = mo.ui.number(
        value=0.1,
        start=0,
        stop=10,
        step=0.1,
        label="Regularization Parameter (α)"
    )

    controls = mo.hstack([
        mo.vstack([
            mo.md("### Data Parameters"),
            sample_size,
            alpha
        ])
    ])
    return alpha, controls, sample_size


@app.cell(hide_code=True)
def _(controls):
    controls
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Model Coefficients""")
    return


@app.cell
def _(mo):
    coefficient_inputs = mo.ui.array([
        mo.ui.number(value=0.2, label="Coefficient 1", step=0.1),
        mo.ui.number(value=2.0, label="Coefficient 2", step=0.1)
    ], label="Model Coefficients")

    coefficient_section = mo.hstack([
        coefficient_inputs,
        mo.callout(
            mo.md("Adjust coefficients to see how they affect the loss value."),
            kind="warn"
        )
    ])
    return coefficient_inputs, coefficient_section


@app.cell(hide_code=True)
def _(coefficient_section):
    coefficient_section
    return


@app.cell
def _(np):
    def ridge_loss(X, w, y_true, alpha):
        """Calculate Ridge Regression loss.

        Args:
            X (np.ndarray): Feature matrix (n_samples, n_features)
            w (np.ndarray): Coefficient vector
            y_true (np.ndarray): True target values
            alpha (float): Regularization parameter

        Returns:
            float: Ridge loss value
        """
        n_samples = X.shape[0]

        # Check dimension compatibility
        if X.shape[1] != len(w):
            raise ValueError(f"Coefficient count ({len(w)}) must match feature count ({X.shape[1]})")

        y_pred = X @ w
        mse = np.mean((y_true - y_pred) ** 2)
        regularization = alpha * np.sum(w ** 2)
        return mse + regularization
    return (ridge_loss,)


@app.cell(hide_code=True)
def _(mo):
    visualize_button = mo.ui.run_button(label="Visualize Predictions")
    return (visualize_button,)


@app.cell
def _(alpha, coefficient_inputs, mo, np, ridge_loss, sample_size):
    # Generate sample data
    X = np.column_stack([
        np.arange(1, sample_size.value + 1),
        np.ones(sample_size.value)
    ])
    y_true = np.arange(2, sample_size.value + 2)
    w = np.array(coefficient_inputs.value)

    try:
        # Calculate loss
        current_loss = ridge_loss(X, w, y_true, alpha.value)

        # Calculate components for display
        mse_component = np.mean((y_true - X @ w) ** 2)
        reg_component = alpha.value * np.sum(w ** 2)

        # Display results
        result_display = mo.vstack([
            mo.md("### Current Loss Value"),
            mo.callout(
                mo.md(f"Ridge Loss: **{current_loss:.4f}**\n\n"
                    f"- MSE Component: {mse_component:.4f}\n"
                    f"- Regularization Component: {reg_component:.4f}"),
                kind="info"
            )
        ])
    except Exception as e:
        result_display = mo.vstack([
            mo.md("### Error"),
            mo.callout(
                mo.md(f"Error: {str(e)}"),
                kind="danger"
            )
        ])
        current_loss = None
    return (
        X,
        current_loss,
        mse_component,
        reg_component,
        result_display,
        w,
        y_true,
    )


@app.cell(hide_code=True)
def _(result_display):
    result_display
    return


@app.cell
def _(pd, px):
    def plot_predictions(X, w, y_true):
        y_pred = X @ w

        # Use pandas DataFrame for better compatibility with plotly
        df = pd.DataFrame({
            'x': X[:, 0],
            'True Values': y_true,
            'Predictions': y_pred
        })

        # Prepare data for plotting
        plot_df = pd.melt(df, id_vars=['x'], value_vars=['True Values', 'Predictions'],
                         var_name='Type', value_name='Value')

        fig = px.scatter(plot_df, x='x', y='Value', color='Type',
                        title='True Values vs Predictions',
                        labels={'Value': 'Value', 'x': 'Sample Index'})

        # Add lines connecting points
        for series_name in ['True Values', 'Predictions']:
            series_data = df[['x', series_name]].sort_values('x')
            fig.add_scatter(x=series_data['x'], y=series_data[series_name], 
                           mode='lines', name=f'{series_name} (line)',
                           line=dict(dash='dash'))

        return fig
    return (plot_predictions,)


@app.cell(hide_code=True)
def _(visualize_button):
    visualize_button
    return


@app.cell
def _(X, mo, plot_predictions, visualize_button, w, y_true):
    plot_results = None
    if visualize_button.value:
        try:
            plot_results = plot_predictions(X, w, y_true)
        except Exception as e:
            plot_results = mo.md(f"Error generating plot: {str(e)}").callout(kind="danger")
    plot_results
    return (plot_results,)


@app.cell(hide_code=True)
def _(mo):
    conclusion = mo.vstack([
        mo.callout(
            mo.md("""
                **Congratulations!** 
                You've explored the Ridge Regression loss function interactively. Key takeaways:

                - Understanding the balance between MSE and regularization
                - Impact of the regularization parameter (α)
                - How coefficients affect predictions and loss
                - How Ridge regression shrinks coefficients toward zero
            """),
            kind="success"
        ),
        mo.accordion({
            "🎯 Applications": mo.md("""
                - High-dimensional data analysis
                - Feature selection
                - Preventing overfitting in linear models
                - Multicollinearity handling
            """),
            "🚀 Next Steps": mo.md("""
                1. Implement gradient descent optimization
                2. Compare with Lasso regression
                3. Explore cross-validation for α selection
                4. Apply to real-world datasets
            """)
        })
    ])
    return (conclusion,)


@app.cell(hide_code=True)
def _(conclusion):
    conclusion
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import plotly.express as px
    import pandas as pd
    return np, pd, px


if __name__ == "__main__":
    app.run()
