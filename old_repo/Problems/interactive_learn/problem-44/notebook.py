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
        # Understanding the Leaky ReLU Activation Function

        The [Leaky ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) (Leaky Rectified Linear Unit) is a variant of the ReLU activation function that helps prevent the "dying ReLU" problem in neural networks. Let's explore it interactively!
        """
    ).center()
    return


@app.cell
def _(mo):
    value = mo.md(r"""
    For an input $z$, the Leaky ReLU function with parameter $\alpha$ is defined as:

    \[
    \text{LeakyReLU}(z) = \begin{cases} 
    z & \text{if } z > 0 \\
    \alpha z & \text{if } z \leq 0
    \end{cases}
    \]

    where $\alpha$ is typically a small positive value (default is 0.01) that determines the slope for negative inputs.
    """)
    mo.accordion({"### Mathematical Definition" : value})
    return (value,)


@app.cell
def _(mo):
    alpha = mo.ui.slider(
        start=0.01,
        stop=0.5,
        value=0.01,
        step=0.01,
        label="Alpha (α)"
    )

    _callout = mo.callout(
        mo.md("""
            Adjust alpha to see how it affects the negative part of the function.

            **Observe how:**

            - Higher alpha values make negative inputs more significant

            - Lower alpha values make the function more similar to regular ReLU
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
                alpha,
                mo.accordion({
                    "About Alpha": _callout
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
    return alpha, controls, x_range


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
            "About Testing": "Enter specific values to see their Leaky ReLU outputs."
        })
    ])
    return input_controls, test_input


@app.cell
def _(alpha, mo):
    formula_display = mo.vstack([
        mo.md(
            f"""
            ### Current Leaky ReLU Configuration

            With alpha parameter $\\alpha = {alpha.value:.3f}$, the current Leaky ReLU function is:

            \\[
            f(z) = \\begin{{cases}} 
            z & \\text{{if }} z > 0 \\\\
            {alpha.value:.3f}z & \\text{{if }} z \\leq 0
            \\end{{cases}}
            \\]

            - For positive inputs: output = input
            - For negative inputs: output = {alpha.value:.3f} × input

            """
        ),
    ])
    return (formula_display,)


@app.cell
def _(formula_display):
    formula_display
    return


@app.cell(hide_code=True)
def _(alpha, mo, np, plt, test_input, x_range):
    @mo.cache(pin_modules=True)
    def plot_leaky_relu():
        if x_range.value[0] >= x_range.value[1]:
            raise ValueError("Invalid x_range: start value must be less than stop value.")

        x = np.linspace(x_range.value[0], x_range.value[1], 1000)
        y = np.where(x > 0, x, alpha.value * x)

        plt.figure(figsize=(12, 7))

        # Plot negative region
        mask_neg = x <= 0
        plt.plot(x[mask_neg], y[mask_neg], 
                label=f'Negative region (slope = {alpha.value:.3f})', 
                color='red', 
                linewidth=2)

        # Plot positive region
        mask_pos = x > 0
        plt.plot(x[mask_pos], y[mask_pos], 
                label='Positive region (slope = 1.0)', 
                color='blue', 
                linewidth=2)

        # Plot test point if within range
        if x_range.value[0] <= test_input.value <= x_range.value[1]:
            test_output = test_input.value if test_input.value > 0 else alpha.value * test_input.value
            plt.scatter([test_input.value], [test_output], 
                       color='green', s=100, 
                       label=f'Test point: f({test_input.value:.2f}) = {test_output:.2f}')

        plt.grid(True, alpha=0.3)
        plt.title(f'Leaky ReLU Function (α = {alpha.value:.3f})')
        plt.xlabel('Input (z)')
        plt.ylabel('Output (LeakyReLU(z))')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add zero lines
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

        plot_display = mo.vstack([
            mo.as_html(plt.gca()),
        ])

        return plot_display
    return (plot_leaky_relu,)


@app.cell
def _(controls):
    controls
    return


@app.cell
def _(input_controls):
    input_controls
    return


@app.cell
def _(plot_leaky_relu):
    plot_leaky_relu()
    return


@app.cell
def _(mo):
    conclusion = mo.vstack([
        mo.callout(
            mo.md("""
                **Congratulations!** 
                You've explored the Leaky ReLU function interactively. You've learned:

                - How Leaky ReLU modifies ReLU to handle negative inputs
                - The effect of the alpha parameter
                - Why this helps prevent the "dying ReLU" problem
            """),
            kind="success"
        ),
        mo.accordion({
            "Next Steps": mo.md("""
                1. **Implementation:** Try implementing Leaky ReLU from scratch
                2. **Compare:** Explore how it differs from other ReLU variants
                3. **Experiment:** Test different alpha values in a neural network
                4. **Advanced:** Learn about other solutions to the dying ReLU problem
            """),
            "🎯 Common Applications": mo.md("""
                - Deep neural networks
                - Convolutional neural networks (CNNs)
                - Feature extraction layers
                - When regular ReLU causes dead neurons
                - Training very deep networks
            """),
        })
    ])
    return (conclusion,)


@app.cell
def _(mo):
    mo.md(f"""
    This interactive learning experience was designed to help you understand the Leaky ReLU activation function. Hope this helps in your deep learning journey!
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
