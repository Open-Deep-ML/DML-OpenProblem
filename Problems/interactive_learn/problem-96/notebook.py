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
        # Understanding the Hard Sigmoid Activation Function

        The [Hard Sigmoid](https://arxiv.org/abs/1511.00363v3) is a piece-wise linear approximation of the sigmoid activation function. It's computationally more efficient while maintaining similar characteristics. Let's explore it interactively!
        """
    ).center()
    return


@app.cell
def _(mo):
    value = mo.md(r"""
    For an input $x$, the Hard Sigmoid function is defined as:

    \[
    \text{HardSigmoid}(x) = \begin{cases} 
    0 & \text{if } x \leq -2.5 \\
    0.2x + 0.5 & \text{if } -2.5 < x < 2.5 \\
    1 & \text{if } x \geq 2.5
    \end{cases}
    \]

    This piece-wise linear function approximates the smooth sigmoid curve with straight lines.
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
            Observe the three distinct regions of the Hard Sigmoid function:

            - Below -2.5: Output is 0
            - Between -2.5 and 2.5: Linear interpolation (0.2x + 0.5)
            - Above 2.5: Output is 1
        """),
        kind="info"
    )

    controls = mo.vstack([
        mo.md("### Adjust Parameters"),
        mo.hstack([
            mo.vstack([
                x_range,
                mo.accordion({
                    "About Function Regions": _callout
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
            "About Testing": "Enter specific values to see their Hard Sigmoid outputs."
        })
    ])
    return input_controls, test_input


@app.cell
def _(mo):
    formula_display = mo.vstack([
        mo.md(
            r"""
            ### Current Hard Sigmoid Configuration

            The Hard Sigmoid function is defined piece-wise:

            \[
            f(x) = \begin{cases} 
            0 & \text{if } x \leq -2.5 \\
            0.2x + 0.5 & \text{if } -2.5 < x < 2.5 \\
            1 & \text{if } x \geq 2.5
            \end{cases}
            \]

            - For x ≤ -2.5: output = 0
            - For -2.5 < x < 2.5: output = 0.2x + 0.5
            - For x ≥ 2.5: output = 1
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
    def plot_hard_sigmoid():
        if x_range.value[0] >= x_range.value[1]:
            raise ValueError("Invalid x_range: start value must be less than stop value.")

        x = np.linspace(x_range.value[0], x_range.value[1], 1000)
        y = np.clip(0.2 * x + 0.5, 0, 1)

        plt.figure(figsize=(12, 7))

        # Plot the defined three regions with different colors
        mask_left = x <= -2.5
        mask_middle = (x > -2.5) & (x < 2.5)
        mask_right = x >= 2.5

        plt.plot(x[mask_left], y[mask_left], 
                label='Lower saturation (y = 0)', 
                color='red', 
                linewidth=2)
        plt.plot(x[mask_middle], y[mask_middle], 
                label='Linear region (y = 0.2x + 0.5)', 
                color='blue', 
                linewidth=2)
        plt.plot(x[mask_right], y[mask_right], 
                label='Upper saturation (y = 1)', 
                color='green', 
                linewidth=2)

        # Plot test point if within range (extend slider range accordingly)
        if x_range.value[0] <= test_input.value <= x_range.value[1]:
            test_output = np.clip(0.2 * test_input.value + 0.5, 0, 1)
            plt.scatter([test_input.value], [test_output], 
                       color='purple', s=100, 
                       label=f'Test point: f({test_input.value:.2f}) = {test_output:.2f}')

        plt.grid(True, alpha=0.3)
        plt.title('Hard Sigmoid Function')
        plt.xlabel('Input (x)')
        plt.ylabel('Output (HardSigmoid(x))')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add zero lines
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

        plot_display = mo.vstack([
            mo.as_html(plt.gca()),
        ])

        return plot_display
    return (plot_hard_sigmoid,)


@app.cell
def _(controls):
    controls
    return


@app.cell
def _(input_controls):
    input_controls
    return


@app.cell
def _(plot_hard_sigmoid):
    plot_hard_sigmoid()
    return


@app.cell
def _(mo):
    conclusion = mo.vstack([
        mo.callout(
            mo.md("""
                **Congratulations!** 
                You've explored the Hard Sigmoid function interactively. You've learned:

                - How Hard Sigmoid approximates the standard sigmoid function
                - The three distinct regions of the function
                - The simplicity of the function -> only requires basic arithmetic operations -> making it computationally cheaper
            """),
            kind="success"
        ),
        mo.accordion({
            "Next Steps": mo.md("""
                1. **Implementation:** Try implementing Hard Sigmoid from scratch
                2. **Compare:** Explore how it differs from regular sigmoid
                3. **Experiment:** Test performance benefits in a neural network
                4. **Advanced:** Learn about other sigmoid approximations
            """),
            "🎯 Common Applications": mo.md("""
                - Neural networks requiring fast computation
                - Embedded systems with limited resources
                - Mobile applications
                - Real-time inference systems
                - Hardware implementations of neural networks
            """),
        })
    ])
    return (conclusion,)


@app.cell
def _(mo):
    mo.md(f"""
    This interactive learning experience was designed to help you understand the Hard Sigmoid activation function. Hope this helps in your deep learning journey!
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
