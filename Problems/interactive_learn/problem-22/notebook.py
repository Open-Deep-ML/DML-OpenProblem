# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.0",
#     "numpy==2.2.1",
# ]
# ///

import marimo

__generated_with = "0.10.10"
app = marimo.App(css_file="/Users/adityakhalkar/Library/Application Support/mtheme/themes/deepml.css")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Understanding the Sigmoid Activation Function

        The [Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function), also known as the logistic function, is one of the most important activation functions in neural networks. It transforms any input into a value between 0 and 1, making it particularly useful for binary classification and probability estimation. Let's explore it interactively!
        """
    ).center()
    return


@app.cell
def _(mo):
    value = mo.md(r"""
    The Sigmoid function is defined as:

    \[
    \sigma(z) = \frac{1}{1 + e^{-z}}
    \]

    where $e$ is Euler's number (approximately ~ 2.71828). This function has several key properties:

    - Output is always between 0 and 1

    - S-shaped curve (sigmoid shape)

    - Smooth and continuous

    - Differentiable everywhere
    """)
    mo.accordion({"### Mathematical Definition" : value})
    return (value,)


@app.cell
def _(mo):
    steepness = mo.ui.slider(
        start=0.1,
        stop=5.0,
        value=1.0,
        step=0.1,
        label="Steepness Factor (β)"
    )

    _callout = mo.callout(
        mo.md("""
            Adjust steepness to see how it affects the sigmoid curve.

            **Observe how:**

            - Higher values make the transition steeper

            - Lower values make the transition more gradual

            - This affects how quickly the function transitions from 0 to 1
        """),
        kind="info"
    )

    x_range = mo.ui.range_slider(
        start=-10,
        stop=10,
        step=0.5,
        value=[-5, 5],
        label="X-axis Range"
    )

    controls = mo.vstack([
        mo.md("### Adjust Parameters"),
        mo.hstack([
            mo.vstack([
                steepness,
                mo.accordion({
                    "About Steepness": _callout
                })
            ]),
            mo.vstack([
                x_range,
                mo.accordion({
                    "About Range": "Adjust to see different regions of the S-shaped curve."
                })
            ])
        ])
    ])
    return controls, steepness, x_range


@app.cell
def _(mo):
    test_input = mo.ui.number(
        value=0.0,
        start=-10,
        stop=10,
        step=0.1,
        label="Test Input Value"
    )

    input_controls = mo.vstack([
        mo.md("### Test Specific Values"),
        test_input,
        mo.accordion({
            "About Testing": "Enter specific values to see their Sigmoid outputs (always between 0 and 1)."
        })
    ])
    return input_controls, test_input


@app.cell
def _(mo, steepness):
    formula_display = mo.vstack([
        mo.md(
            f"""
            ### Current Sigmoid Configuration

            With steepness parameter $\\beta = {steepness.value:.2f}$, the current Sigmoid function is:

            \\[
            \\sigma(z) = \\frac{{1}}{{1 + e^{{-{steepness.value:.2f}z}}}}
            \\]

            Key points at current steepness:

            - Center point at z = 0, where σ(0) = 0.5

            - As z → ∞, σ(z) → 1

            - As z → -∞, σ(z) → 0
            """
        ),
    ])
    return (formula_display,)


@app.cell
def _(formula_display):
    formula_display
    return


@app.cell(hide_code=True)
def _(mo, np, plt, steepness, test_input, x_range):
    @mo.cache(pin_modules=True)
    def plot_sigmoid():
        if x_range.value[0] >= x_range.value[1]:
            raise ValueError("Invalid x_range: start value must be less than stop value.")

        x = np.linspace(x_range.value[0], x_range.value[1], 1000)
        y = 1 / (1 + np.exp(-steepness.value * x))

        plt.figure(figsize=(12, 7))

        # Plot main sigmoid curve
        plt.plot(x, y, 
                label=f'Sigmoid (β = {steepness.value:.2f})', 
                color='blue', 
                linewidth=2)

        # Plot horizontal asymptotes
        plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Upper Asymptote (y=1)')
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Lower Asymptote (y=0)')

        # Plot midpoint
        plt.plot([0], [0.5], 'ro', label='Midpoint (0, 0.5)')

        # Plot test point if within range
        if x_range.value[0] <= test_input.value <= x_range.value[1]:
            test_output = 1 / (1 + np.exp(-steepness.value * test_input.value))
            plt.scatter([test_input.value], [test_output], 
                       color='green', s=100, 
                       label=f'Test point: σ({test_input.value:.2f}) = {test_output:.4f}')

        plt.grid(True, alpha=0.3)
        plt.title(f'Sigmoid Function (β = {steepness.value:.2f})')
        plt.xlabel('Input (z)')
        plt.ylabel('Output (σ(z))')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add vertical zero line
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

        # Set y-axis limits with some padding
        plt.ylim(-0.1, 1.1)

        plot_display = mo.vstack([
            mo.as_html(plt.gca()),
        ])

        return plot_display
    return (plot_sigmoid,)


@app.cell
def _(controls):
    controls
    return


@app.cell
def _(input_controls):
    input_controls
    return


@app.cell
def _(plot_sigmoid):
    plot_sigmoid()
    return


@app.cell
def _(mo):
    conclusion = mo.vstack([
        mo.callout(
            mo.md("""
                **Congratulations!** 
                You've explored the Sigmoid function interactively. You've learned:

                - How Sigmoid squashes any input to a value between 0 and 1
                - The effect of the steepness parameter
                - Why it's useful for binary classification
                - Its symmetric properties around the midpoint
            """),
            kind="success"
        ),
        mo.accordion({
            "Next Steps": mo.md("""
                1. **Implementation:** Try implementing Sigmoid from scratch
                2. **Compare:** Explore how it differs from tanh and softmax
                3. **Practice:** Use it in a binary classification task
                4. **Advanced:** Learn about vanishing gradient problems
            """),
            "🎯 Common Applications": mo.md("""
                - Binary classification output layers
                - Logistic regression
                - Neural network hidden layers (historically)
                - Probability estimation
                - Feature scaling between 0 and 1
            """),
        })
    ])
    return (conclusion,)


@app.cell
def _(mo):
    mo.md(f"""
    This interactive learning experience was designed to help you understand the Sigmoid activation function. Hope this helps in your deep learning journey!
    """)
    return


@app.cell
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
    import matplotlib.pyplot as plt
    return np, plt


if __name__ == "__main__":
    app.run()
