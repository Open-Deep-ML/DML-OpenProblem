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
        # Understanding the SELU Activation Function

        The [SELU](https://arxiv.org/abs/1706.02515) (Scaled Exponential Linear Unit) is a self-normalizing activation function that automatically ensures normalized outputs. Let's explore its unique properties interactively!
        """
    ).center()
    return


@app.cell
def _(mo):
    value = mo.md(r"""
    For an input $z$, the SELU function with parameter $\alpha$ is defined as:

    \[
    \text{SELU}(z) = \lambda \begin{cases} 
    z & \text{if } z > 0 \\
    \alpha(e^z - 1) & \text{if } z \leq 0
    \end{cases}
    \]

    where $\lambda \approx 1.0507$ is a scaling parameter and $\alpha \approx 1.6733$ is the default shape parameter for self-normalization.
    """)
    mo.accordion({"### Mathematical Definition" : value})
    return (value,)


@app.cell
def _(mo):
    alpha = mo.ui.slider(
        start=0.1,
        stop=3.0,
        value=1.6732632423543772,  # Default SELU alpha
        step=0.01,
        label="Alpha (α)"
    )

    _callout = mo.callout(
        mo.md("""
            Adjust alpha to see how it affects the negative part of the function.

            **Key Properties:**

            - Default α ≈ 1.6733 ensures self-normalization
            - Fixed λ ≈ 1.0507 provides proper scaling
            - Experiment with different α values to understand impact
            - Standard SELU uses fixed parameters for optimal performance
        """),
        kind="info"
    )

    x_range = mo.ui.range_slider(
        start=-5,
        stop=5,
        step=0.5,
        value=[-3, 3],
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
        start=-5,
        stop=5,
        step=0.1,
        label="Test Input Value"
    )

    input_controls = mo.vstack([
        mo.md("### Test Specific Values"),
        test_input,
        mo.accordion({
            "About Testing": "Enter specific values to see their SELU outputs."
        })
    ])
    return input_controls, test_input


@app.cell
def _(alpha, mo):
    formula_display = mo.vstack([
        mo.md(
            f"""
            ### Current SELU Configuration

            With parameter $\\alpha = {alpha.value:.3f}$ and fixed $\\lambda = 1.0507$, the current SELU function is:

            $$
            f(z) = 1.0507 \\times \\begin{{cases}} 
            z & \\text{{if }} z > 0 \\\\
            {alpha.value:.3f}(e^z - 1) & \\text{{if }} z \\leq 0
            \\end{{cases}}
            $$

            - For positive inputs: output = 1.0507 × input
            - For negative inputs: output = 1.0507 × {alpha.value:.3f} × (e^input - 1)
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
    def plot_selu():
        if x_range.value[0] >= x_range.value[1]:
            raise ValueError("Invalid x_range: start value must be less than stop value.")

        scale = 1.0507009873554804

        x = np.linspace(x_range.value[0], x_range.value[1], 1000)
        y = scale * np.where(x > 0, x, alpha.value * (np.exp(x) - 1))

        plt.figure(figsize=(12, 7))

        # Plot -ve region
        mask_neg = x <= 0
        plt.plot(x[mask_neg], y[mask_neg], 
                label=f'Negative region (α={alpha.value:.3f})', 
                color='darkred', 
                linewidth=2)

        # Plot +ve region
        mask_pos = x > 0
        plt.plot(x[mask_pos], y[mask_pos], 
                label='Positive region (λx)', 
                color='darkgreen', 
                linewidth=2)

        # Plot test point if within range (extend range to show point if needed)
        if x_range.value[0] <= test_input.value <= x_range.value[1]:
            if test_input.value > 0:
                test_output = scale * test_input.value
            else:
                test_output = scale * alpha.value * (np.exp(test_input.value) - 1)
            plt.scatter([test_input.value], [test_output], 
                       color='orange', s=100, 
                       label=f'Test point: f({test_input.value:.2f}) = {test_output:.2f}')

        plt.grid(True, alpha=0.3)
        plt.title(f'SELU Function (α = {alpha.value:.3f})')
        plt.xlabel('Input (z)')
        plt.ylabel('Output (SELU(z))')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add zero lines
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

        plot_display = mo.vstack([
            mo.as_html(plt.gca()),
        ])

        return plot_display
    return (plot_selu,)


@app.cell
def _(controls):
    controls
    return


@app.cell
def _(input_controls):
    input_controls
    return


@app.cell
def _(plot_selu):
    plot_selu()
    return


@app.cell
def _(mo):
    conclusion = mo.vstack([
        mo.callout(
            mo.md("""
                **Congratulations!** 
                You've explored the SELU activation function interactively. You've learned:

                - How SELU achieves self-normalization
                - The role of λ and α parameters
                - Why specific parameter values are chosen
            """),
            kind="success"
        ),
        mo.accordion({
            "Next Steps": mo.md("""
                1. **Implementation:** Try implementing SELU in a neural network
                2. **Compare:** Test against other activation functions
                3. **Experiment:** Observe self-normalizing properties
                4. **Advanced:** Learn about proper initialization with SELU
            """),
            "🎯 Common Applications": mo.md("""
                - Deep Neural Networks
                - Self-Normalizing Neural Networks (SNNs)
                - Networks requiring stable training
                - Deep architectures sensitive to normalization
                - Cases where batch normalization is impractical
            """),
        })
    ])
    return (conclusion,)


@app.cell
def _(mo):
    mo.md(f"""
    This interactive learning experience was designed to help you understand the SELU activation function. Hope this helps in your deep learning journey!
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
