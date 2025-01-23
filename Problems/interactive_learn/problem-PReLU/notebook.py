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
        # Understanding the PReLU Activation Function

        The [PReLU](https://arxiv.org/abs/1502.01852) (Parametric Rectified Linear Unit) is an advanced variant of the ReLU activation function that introduces learnable parameters for handling negative values in neural networks. Let's explore it interactively!
        """
    ).center()
    return


@app.cell
def _(mo):
    value = mo.md(r"""
    For an input $z$, the PReLU function with learnable parameter $\alpha$ is defined as:

    \[
    \text{PReLU}(z) = \begin{cases} 
    z & \text{if } z > 0 \\
    \alpha z & \text{if } z \leq 0
    \end{cases}
    \]

    where $\alpha$ is a learnable parameter that can be optimized during training, unlike Leaky ReLU where it's fixed.
    """)
    mo.accordion({"### Mathematical Definition" : value})
    return (value,)


@app.cell
def _(mo):
    alpha = mo.ui.slider(
        start=0.01,
        stop=1.0,
        value=0.25,
        step=0.01,
        label="Alpha (α)"
    )

    _callout = mo.callout(
        mo.md("""
            Adjust alpha to see how it affects the negative part of the function.

            **Key Differences from Leaky ReLU:**

            - α is learned during training for each neuron
            - Can adapt to the data distribution
            - May have different values for different channels
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
            "About Testing": "Enter specific values to see their PReLU outputs."
        })
    ])
    return input_controls, test_input


@app.cell
def _(alpha, mo):
    formula_display = mo.vstack([
        mo.md(
            f"""
            ### Current PReLU Configuration

            With learnable parameter $\\alpha = {alpha.value:.3f}$, the current PReLU function is:

            \\[
            f(z) = \\begin{{cases}} 
            z & \\text{{if }} z > 0 \\\\
            {alpha.value:.3f}z & \\text{{if }} z \\leq 0
            \\end{{cases}}
            \\]

            - For positive inputs: output = input
            - For negative inputs: output = {alpha.value:.3f} × input (learnable)

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
    def plot_prelu():
        if x_range.value[0] >= x_range.value[1]:
            raise ValueError("Invalid x_range: start value must be less than stop value.")

        x = np.linspace(x_range.value[0], x_range.value[1], 1000)
        y = np.where(x > 0, x, alpha.value * x)

        plt.figure(figsize=(12, 7))

        # Plot -ve region
        mask_neg = x <= 0
        plt.plot(x[mask_neg], y[mask_neg], 
                label=f'Negative region (learned slope = {alpha.value:.3f})', 
                color='purple', 
                linewidth=2)

        # Plot +ve region
        mask_pos = x > 0
        plt.plot(x[mask_pos], y[mask_pos], 
                label='Positive region (slope = 1.0)', 
                color='green', 
                linewidth=2)

        # Plot test point if within range (modify x range slider accordingly)
        if x_range.value[0] <= test_input.value <= x_range.value[1]:
            test_output = test_input.value if test_input.value > 0 else alpha.value * test_input.value
            plt.scatter([test_input.value], [test_output], 
                       color='orange', s=100, 
                       label=f'Test point: f({test_input.value:.2f}) = {test_output:.2f}')

        plt.grid(True, alpha=0.3)
        plt.title(f'PReLU Function (learned α = {alpha.value:.3f})')
        plt.xlabel('Input (z)')
        plt.ylabel('Output (PReLU(z))')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add zero lines
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

        plot_display = mo.vstack([
            mo.as_html(plt.gca()),
        ])

        return plot_display
    return (plot_prelu,)


@app.cell
def _(controls):
    controls
    return


@app.cell
def _(input_controls):
    input_controls
    return


@app.cell
def _(plot_prelu):
    plot_prelu()
    return


@app.cell
def _(mo):
    conclusion = mo.vstack([
        mo.callout(
            mo.md("""
                **Congratulations!** 
                You've explored the PReLU activation function interactively. You've learned:

                - How PReLU differs from Leaky ReLU through learnable parameters
                - The effect of adaptive alpha parameters
                - Why this helps improve model performance
            """),
            kind="success"
        ),
        mo.accordion({
            "Next Steps": mo.md("""
                1. **Implementation:** Try implementing PReLU in a neural network
                2. **Compare:** Test against ReLU and Leaky ReLU
                3. **Experiment:** Observe learned alpha values after training
                4. **Advanced:** Implement channel-wise PReLU parameters
            """),
            "🎯 Common Applications": mo.md("""
                - Deep Convolutional Neural Networks
                - Image Classification Tasks
                - Feature Extraction Networks
                - When adaptive negative slopes are beneficial
                - Performance-critical deep learning models
            """),
        })
    ])
    return (conclusion,)


@app.cell
def _(mo):
    mo.md(f"""
    This interactive learning experience was designed to help you understand the PReLU activation function. Hope this helps in your deep learning journey!
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
