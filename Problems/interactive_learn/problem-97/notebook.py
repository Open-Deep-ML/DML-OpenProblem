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
        # Understanding the ELU Activation Function

        The [ELU](https://pytorch.org/docs/stable/generated/torch.nn.ELU.html) (Exponential Linear Unit) is a smooth variant of ReLU that allows negative values, helping to push mean unit activations closer to zero. Let's explore it interactively!
        """
    ).center()
    return


@app.cell
def _(mo):
    value = mo.md(r"""
    For an input $z$, the ELU function with parameter $\alpha$ is defined as:

    \[
    \text{ELU}(z) = \begin{cases} 
    z & \text{if } z > 0 \\
    \alpha(e^z - 1) & \text{if } z \leq 0
    \end{cases}
    \]

    where $\alpha$ is typically a positive value (default is 1.0) that determines the magnitude of the negative saturation.
    """)
    mo.accordion({"### Mathematical Definition": value})
    return (value,)


@app.cell
def _(mo):
    alpha = mo.ui.slider(
        start=0.1,
        stop=3.0,
        value=1.0,
        step=0.1,
        label="Alpha (α)"
    )

    _callout = mo.callout(
        mo.md("""
            Adjust alpha to see how it affects the negative part of the function.

            **Observe how:**

            - Higher alpha values increase the magnitude of negative outputs
            - The function smoothly transitions at z = 0
            - Negative values are bounded by -α
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
            "About Testing": "Enter specific values to see their ELU outputs."
        })
    ])
    return input_controls, test_input


@app.cell
def _(alpha, mo):
    formula_display = mo.vstack([
        mo.md(
            f"""
            ### Current ELU Configuration

            With alpha parameter $\\alpha = {alpha.value:.3f}$, the current ELU function is:

            \\[
            f(z) = \\begin{{cases}} 
            z & \\text{{if }} z > 0 \\\\
            {alpha.value:.3f}(e^z - 1) & \\text{{if }} z \\leq 0
            \\end{{cases}}
            \\]

            - For positive inputs: output = input
            - For negative inputs: output = {alpha.value:.3f}(e^z - 1)

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
    def plot_elu():
        if x_range.value[0] >= x_range.value[1]:
            raise ValueError("Invalid x_range: start value must be less than stop value.")

        x = np.linspace(x_range.value[0], x_range.value[1], 1000)
        y = np.where(x > 0, x, alpha.value * (np.exp(x) - 1))

        plt.figure(figsize=(12, 7))

        # Plot -ve region
        mask_neg = x <= 0
        plt.plot(x[mask_neg], y[mask_neg], 
                label=f'Negative region (exponential decay)', 
                color='red', 
                linewidth=2)

        # Plot +ve region
        mask_pos = x > 0
        plt.plot(x[mask_pos], y[mask_pos], 
                label='Positive region (linear)', 
                color='blue', 
                linewidth=2)

        # Plot test point if within range (expand x slider range to show test point)

        if x_range.value[0] <= test_input.value <= x_range.value[1]:
            test_output = test_input.value if test_input.value > 0 else alpha.value * (np.exp(test_input.value) - 1)
            plt.scatter([test_input.value], [test_output], 
                       color='green', s=100, 
                       label=f'Test point: ELU({test_input.value:.2f}) = {test_output:.2f}')

        plt.grid(True, alpha=0.3)
        plt.title(f'ELU Function (α = {alpha.value:.3f})')
        plt.xlabel('Input (z)')
        plt.ylabel('Output (ELU(z))')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add zero lines
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

        plot_display = mo.vstack([
            mo.as_html(plt.gca()),
        ])

        return plot_display
    return (plot_elu,)


@app.cell
def _(controls):
    controls
    return


@app.cell
def _(input_controls):
    input_controls
    return


@app.cell
def _(plot_elu):
    plot_elu()
    return


@app.cell
def _(mo):
    conclusion = mo.vstack([
        mo.callout(
            mo.md("""
                **Congratulations!** 
                You've explored the ELU function interactively. You've learned:

                - How ELU smoothly handles negative inputs
                - The effect of the alpha parameter
                - Why it can help with mean unit activations
                - How it combines the benefits of ReLU with negative values
            """),
            kind="success"
        ),
        mo.accordion({
            "Next Steps": mo.md("""
                1. **Implementation:** Try implementing ELU from scratch
                2. **Compare:** Explore how it differs from ReLU and Leaky ReLU
                3. **Experiment:** Test different alpha values in a neural network
                4. **Advanced:** Learn about variants like SELU
            """),
            "🎯 Common Applications": mo.md("""
                - Deep neural networks
                - When faster learning is desired
                - When negative values are important
                - When dealing with vanishing gradients
                - When mean activations closer to zero are beneficial
            """),
        })
    ])
    return (conclusion,)


@app.cell
def _(mo):
    mo.md(f"""
    This interactive learning experience was designed to help you understand the ELU activation function. Hope this helps in your deep learning journey!
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
