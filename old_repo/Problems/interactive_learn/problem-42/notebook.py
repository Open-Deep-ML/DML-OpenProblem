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
        # Understanding the ReLU Activation Function

        The [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) (Rectified Linear Unit) is one of the most widely used activation functions in neural networks due to its computational efficiency and effectiveness in deep learning. Let's explore it interactively!
        """
    ).center()
    return


@app.cell
def _(mo):
    value = mo.md(r"""
    For an input $z$, the ReLU function is defined as:

    \[
    \text{ReLU}(z) = \begin{cases} 
    z & \text{if } z > 0 \\
    0 & \text{if } z \leq 0
    \end{cases}
    \]

    This can also be written simply as: $\text{ReLU}(z) = \max(0, z)$
    """)
    mo.accordion({"### Mathematical Definition": value})
    return (value,)


@app.cell
def _(mo):
    x_range = mo.ui.range_slider(
        start=-10,
        stop=10,
        step=0.5,
        value=[-5, 5],
        label="X-axis Range"
    )

    _callout = mo.callout(
        mo.md("""
            Adjust the range to explore different regions of the function.

            **Observe how:**

            - All negative inputs are mapped to zero
            - Positive inputs remain unchanged
            - This creates non-linearity in neural networks
        """),
        kind="info"
    )

    controls = mo.vstack([
        mo.md("### Adjust Parameters"),
        mo.hstack([
            mo.vstack([
                x_range,
                mo.accordion({
                    "About Range": _callout
                })
            ])
        ])
    ])
    return controls, x_range


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
            "About Testing": "Enter specific values to see their ReLU outputs."
        })
    ])
    return input_controls, test_input


@app.cell
def _(mo):
    formula_display = mo.vstack([
        mo.md(
            f"""
            ### Current ReLU Configuration

            The ReLU function is defined as:

            \[
            f(z) = \max(0, z) = \\begin{{cases}} 
            z & \\text{{if }} z > 0 \\\\
            0 & \\text{{if }} z \\leq 0
            \\end{{cases}}
            \]

            - For positive inputs: output = input
            - For negative inputs: output = 0

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
    def plot_relu():
        if x_range.value[0] >= x_range.value[1]:
            raise ValueError("Invalid x_range: start value must be less than stop value.")

        x = np.linspace(x_range.value[0], x_range.value[1], 1000)
        y = np.maximum(0, x)

        plt.figure(figsize=(12, 7))

        # Plot -ve region
        mask_neg = x <= 0
        plt.plot(x[mask_neg], y[mask_neg], 
                label='Zero region (ReLU = 0)', 
                color='red', 
                linewidth=2)

        # Plot +ve region
        mask_pos = x > 0
        plt.plot(x[mask_pos], y[mask_pos], 
                label='Linear region (ReLU = x)', 
                color='blue', 
                linewidth=2)

        # Plot test point if within range (expand slider if it doesn't appear)
        if x_range.value[0] <= test_input.value <= x_range.value[1]:
            test_output = max(0, test_input.value)
            plt.scatter([test_input.value], [test_output], 
                       color='green', s=100, 
                       label=f'Test point: ReLU({test_input.value:.2f}) = {test_output:.2f}')

        plt.grid(True, alpha=0.3)
        plt.title('ReLU Function')
        plt.xlabel('Input (z)')
        plt.ylabel('Output (ReLU(z))')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add zero lines
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

        plot_display = mo.vstack([
            mo.as_html(plt.gca()),
        ])

        return plot_display
    return (plot_relu,)


@app.cell
def _(controls):
    controls
    return


@app.cell
def _(input_controls):
    input_controls
    return


@app.cell
def _(plot_relu):
    plot_relu()
    return


@app.cell
def _(mo):
    conclusion = mo.vstack([
        mo.callout(
            mo.md("""
                **Congratulations!** 
                You've explored the ReLU function interactively. You've learned:

                - How ReLU maps negative inputs to zero
                - How to visualize the function across different input ranges
                - The relationship between input values and their corresponding outputs
                - How to test specific input values and observe their ReLU transformations
            """),
            kind="success"
        ),
        mo.accordion({
            "Next Steps": mo.md("""
                1. **Implementation:** Try implementing ReLU from scratch
                2. **Compare:** Explore how it differs from other activation functions
                3. **Experiment:** Test ReLU in a simple neural network
                4. **Advanced:** Learn about variants like Leaky ReLU and ELU
            """),
            "🎯 Common Applications": mo.md("""
                - Deep neural networks
                - Convolutional neural networks (CNNs)
                - Hidden layers in deep learning
                - Feature extraction
                - Modern computer vision models
            """),
        })
    ])
    return (conclusion,)


@app.cell
def _(mo):
    mo.md(f"""
    This interactive learning experience was designed to help you understand the ReLU activation function. Hope this helps in your deep learning journey!
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
