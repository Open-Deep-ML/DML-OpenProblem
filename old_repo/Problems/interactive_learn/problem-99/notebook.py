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
        # Understanding the Softplus Activation Function

        The [Softplus function](https://en.wikipedia.org/wiki/Softplus) is a smooth approximation of the ReLU activation function. It provides a more gradual transition around zero, which can be beneficial in certain neural network architectures. Let's explore it interactively!
        """
    ).center()
    return


@app.cell
def _(mo):
    value = mo.md(r"""
    For an input $x$, the Softplus function is defined as:

    \[
    \text{Softplus}(x) = \log(1 + e^x)
    \]

    where $\log$ is the natural logarithm. This function provides a smooth, differentiable alternative to ReLU.
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
            Observe how Softplus behaves:

            - For large negative values: Output approaches 0
            - Around 0: Smooth transition
            - For large positive values: Approaches linear (similar to ReLU)
        """),
        kind="info"
    )

    controls = mo.vstack([
        mo.md("### Adjust Range"),
        mo.hstack([
            mo.vstack([
                x_range,
                mo.accordion({
                    "About Function": _callout
                })
            ])
        ])
    ])
    return controls, x_range


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
            "About Testing": "Enter specific values to see their Softplus outputs and compare with ReLU."
        })
    ])
    return input_controls, test_input


@app.cell
def _(mo):
    formula_display = mo.vstack([
        mo.md(
            r"""
            ### Softplus Function Properties

            The Softplus function $f(x) = \log(1 + e^x)$ has these key properties:

            - Always positive output (like ReLU)
            - Smooth and differentiable everywhere
            - Derivative is the logistic sigmoid function: $f'(x) = \frac{1}{1 + e^{-x}}$
            """
        ),
    ])
    return (formula_display,)


@app.cell
def _(formula_display):
    formula_display
    return


@app.cell(hide_code=True)
def _(mo, np, plt, test_input, x_range):
    @mo.cache(pin_modules=True)
    def plot_softplus():
        if x_range.value[0] >= x_range.value[1]:
            raise ValueError("Invalid x_range: start value must be less than stop value.")

        x = np.linspace(x_range.value[0], x_range.value[1], 1000)
        y_softplus = np.log1p(np.exp(x))  # numerically stable version
        y_relu = np.maximum(0, x)  # ReLU for comparison

        plt.figure(figsize=(12, 7))

        # Plot Softplus
        plt.plot(x, y_softplus, 
                label='Softplus', 
                color='blue', 
                linewidth=2)

        # Plot ReLU for comparison (to see the smooth approximation)
        plt.plot(x, y_relu, 
                label='ReLU (for comparison)', 
                color='red', 
                linestyle='--', 
                alpha=0.5,
                linewidth=2)

        # Plot test point if within range (adjust slider value accordingly)
        if x_range.value[0] <= test_input.value <= x_range.value[1]:
            test_output = np.log1p(np.exp(test_input.value))
            plt.scatter([test_input.value], [test_output], 
                       color='green', s=100, 
                       label=f'Test point: f({test_input.value:.2f}) = {test_output:.2f}')

        plt.grid(True, alpha=0.3)
        plt.title('Softplus Function vs ReLU')
        plt.xlabel('Input (x)')
        plt.ylabel('Output (Softplus(x))')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add zero lines
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

        plot_display = mo.vstack([
            mo.as_html(plt.gca()),
        ])

        return plot_display
    return (plot_softplus,)


@app.cell
def _(controls):
    controls
    return


@app.cell
def _(input_controls):
    input_controls
    return


@app.cell
def _(plot_softplus):
    plot_softplus()
    return


@app.cell
def _(mo):
    conclusion = mo.vstack([
        mo.callout(
            mo.md("""
                **Congratulations!** 
                You've explored the Softplus activation function interactively. You've learned:

                - How Softplus smoothly approximates ReLU
                - The mathematical formula and its properties
                - How it compares to ReLU visually
                - Its behavior in different input regions
            """),
            kind="success"
        ),
        mo.accordion({
            "Key Takeaways": mo.md("""
                - Softplus is always positive
                - Provides smooth transitions (differentiable everywhere)
                - Approaches ReLU for large positive values
                - More computationally expensive than ReLU
            """),
        })
    ])
    return (conclusion,)


@app.cell
def _(mo):
    mo.md(f"""
    This interactive learning experience was designed to help you understand the Softplus activation function. Hope this helps in your deep learning journey!
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
