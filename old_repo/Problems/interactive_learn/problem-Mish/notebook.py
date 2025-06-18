# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.0",
#     "numpy==2.2.1",
# ]
# ///

import marimo

__generated_with = "0.10.12"
app = marimo.App(css_file="/Users/adityakhalkar/Library/Application Support/mtheme/themes/deepml.css")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Understanding the Mish Activation Function

        The [Mish](https://arxiv.org/abs/1908.08681) activation function is a smooth, self-regularizing, non-monotonic activation function that aims to improve neural network performance. Let's explore its unique properties interactively!
        """
    ).center()
    return


@app.cell
def _(mo):
    value = mo.md(r"""
    For an input $x$, the Mish function is defined as:

    \[
    \text{Mish}(x) = x \cdot \tanh(\ln(1 + e^x))
    \]

    It combines the characteristics of the softplus function ($\ln(1 + e^x)$) with hyperbolic tangent, creating a smooth and non-monotonic activation function.
    """)
    mo.accordion({"### Mathematical Definition" : value})
    return (value,)


@app.cell
def _(mo):
    x_range = mo.ui.range_slider(
        start=-5,
        stop=5,
        step=0.5,
        value=[-3, 3],
        label="X-axis Range"
    )

    _callout = mo.callout(
        mo.md("""
            Observe the unique characteristics of Mish:

            **Key Properties:**

            - Smooth and non-monotonic behavior
            - Self-regularizing properties
            - Unbounded above, bounded below
            - Preserves small negative gradients
            - Smoother than ReLU and its variants
        """),
        kind="info"
    )

    smoothing = mo.ui.slider(
        start=0.1,
        stop=2.0,
        value=1.0,
        step=0.1,
        label="Smoothing Factor"
    )

    controls = mo.vstack([
        mo.md("### Adjust Parameters"),
        mo.hstack([
            mo.vstack([
                smoothing,
                mo.accordion({
                    "About Smoothing": _callout
                })
            ]),
            mo.vstack([
                x_range,
                mo.accordion({
                    "About Range": "Adjust to see different regions of the function."
                })
            ])
        ])
    ])
    return controls, smoothing, x_range


@app.cell
def _(mo):
    test_input = mo.ui.number(
        value=0.0,
        start=-5,
        stop=5,
        step=0.1,
        label="Test Input Value"
    )

    input_controls = mo.vstack([
        mo.md("### Test Specific Values"),
        test_input,
        mo.accordion({
            "About Testing": "Enter specific values to see their Mish outputs."
        })
    ])
    return input_controls, test_input


@app.cell
def _(mo, smoothing):
    formula_display = mo.vstack([
        mo.md(
            f"""
            ### Current Mish Configuration

            With smoothing factor $\\beta = {smoothing.value:.3f}$, the current Mish function is:

            $$
            f(x) = x \\cdot \\tanh(\\beta \\cdot \\ln(1 + e^x))
            $$

            - Output is unbounded above (like ReLU)
            - Lower bounded by ≈ -0.31
            - Non-monotonic near x = 0
            - Preserves small negative gradients
            """
        ),
    ])
    return (formula_display,)


@app.cell
def _(formula_display):
    formula_display
    return


@app.cell(hide_code=True)
def _(mo, np, plt, smoothing, test_input, x_range):
    @mo.cache(pin_modules=True)
    def plot_mish():
        if x_range.value[0] >= x_range.value[1]:
            raise ValueError("Invalid x_range: start value must be less than stop value.")

        x = np.linspace(x_range.value[0], x_range.value[1], 1000)
        softplus = np.log1p(np.exp(x))
        y = x * np.tanh(smoothing.value * softplus)

        plt.figure(figsize=(12, 7))

        # Main function
        plt.plot(x, y, 
                label=f'Mish (β={smoothing.value:.2f})', 
                color='purple', 
                linewidth=2)

        # Plot test point if within range (vary slider accordingly)
        if x_range.value[0] <= test_input.value <= x_range.value[1]:
            test_softplus = np.log1p(np.exp(test_input.value))
            test_output = test_input.value * np.tanh(smoothing.value * test_softplus)
            plt.scatter([test_input.value], [test_output], 
                       color='orange', s=100, 
                       label=f'Test point: f({test_input.value:.2f}) = {test_output:.2f}')

        plt.grid(True, alpha=0.3)
        plt.title(f'Mish Function (β = {smoothing.value:.2f})')
        plt.xlabel('Input (x)')
        plt.ylabel('Output (Mish(x))')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add zero lines
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

        plot_display = mo.vstack([
            mo.as_html(plt.gca()),
        ])

        return plot_display
    return (plot_mish,)


@app.cell
def _(controls):
    controls
    return


@app.cell
def _(input_controls):
    input_controls
    return


@app.cell
def _(plot_mish):
    plot_mish()
    return


@app.cell
def _(mo):
    conclusion = mo.vstack([
        mo.callout(
            mo.md("""
                **Congratulations!** 
                You've explored the Mish activation function interactively. You've learned:

                - How Mish combines softplus and tanh
                - Its smooth, non-monotonic behavior
                - Why it's effective in deep networks
            """),
            kind="success"
        ),
        mo.accordion({
            "Next Steps": mo.md("""
                1. **Implementation:** Try implementing Mish in a neural network
                2. **Compare:** Test against ReLU and other modern activations
                3. **Experiment:** Observe training stability improvements
                4. **Advanced:** Study the gradients and their behavior
            """),
            "🎯 Common Applications": mo.md("""
                - Deep Neural Networks
                - Computer Vision Tasks
                - Classification Problems
                - When smooth gradients are crucial
                - Problems requiring self-regularization
            """),
        })
    ])
    return (conclusion,)


@app.cell
def _(mo):
    mo.md(f"""
    This interactive learning experience was designed to help you understand the Mish activation function. Hope this helps in your deep learning journey!
    """)
    return


@app.cell
def _(conclusion):
    conclusion
    return


@app.cell(hide_code=True)
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
