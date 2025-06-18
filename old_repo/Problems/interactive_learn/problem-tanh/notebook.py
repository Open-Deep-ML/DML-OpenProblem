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
        # Understanding the Hyperbolic Tangent (tanh) Activation Function

        The [tanh activation function](https://paperswithcode.com/method/tanh-activation) is a scaled and shifted version of the sigmoid, producing outputs between -1 and 1. It's particularly useful for handling negative inputs and is often used in neural networks. Let's explore it interactively!
        """
    ).center()
    return


@app.cell
def _(mo):
    value = mo.md(r"""
    For an input $x$, the tanh function is defined as:

    \[
    \text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
    \]

    This can also be written in terms of the exponential function:

    \[
    \text{tanh}(x) = 2\sigma(2x) - 1
    \]

    where $\sigma(x)$ is the sigmoid function. This creates a smooth, S-shaped function that maps inputs to the range [-1, 1].
    """)
    mo.accordion({"### Mathematical Definition" : value})
    return (value,)


@app.cell
def _(mo):
    steepness = mo.ui.slider(
        start=0.1,
        stop=2.0,
        value=1.0,
        step=0.1,
        label="Steepness"
    )

    _callout = mo.callout(
        mo.md("""
            Adjust steepness to control the shape of the tanh function.

            **Observe how:**

            - Higher values make the transition steeper
            - Lower values make the transition more gradual
            - Default value of 1.0 gives the standard tanh
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
                    "About Range": "Adjust to see different regions of the function."
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
            "About Testing": "Enter specific values to see their tanh outputs."
        })
    ])
    return input_controls, test_input


@app.cell
def _(mo, steepness):
    formula_display = mo.vstack([
        mo.md(
            f"""
            ### Current Tanh Configuration

            With steepness parameter $s = {steepness.value:.1f}$, the current tanh function is:

            \\[
            f(x) = \\text{{tanh}}({steepness.value:.1f}x) = \\frac{{e^{{{steepness.value:.1f}x}} - e^{{-{steepness.value:.1f}x}}}}{{e^{{{steepness.value:.1f}x}} + e^{{-{steepness.value:.1f}x}}}}
            \\]

            Key Properties:
            - Bounded between -1 and 1
            - Symmetric around the origin
            - Steeper gradient near origin compared to sigmoid
            - Outputs centered around zero
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
    def plot_tanh():
        if x_range.value[0] >= x_range.value[1]:
            raise ValueError("Invalid x_range: start value must be less than stop value.")

        x = np.linspace(x_range.value[0], x_range.value[1], 1000)
        y = np.tanh(steepness.value * x)

        plt.figure(figsize=(12, 7))

        # Plot main function
        plt.plot(x, y, 
                label='tanh function', 
                color='blue', 
                linewidth=2)

        # Plot derivative (serves as a reference)
        derivative = steepness.value * (1 - y**2)
        plt.plot(x, derivative, 
                label='Derivative', 
                color='red', 
                linestyle='--',
                alpha=0.5)

        # Plot test point if within range (adjust if needed accordingly)
        if x_range.value[0] <= test_input.value <= x_range.value[1]:
            test_output = np.tanh(steepness.value * test_input.value)
            plt.scatter([test_input.value], [test_output], 
                       color='green', s=100, 
                       label=f'Test point: f({test_input.value:.2f}) = {test_output:.2f}')

        plt.grid(True, alpha=0.3)
        plt.title(f'Tanh Function (steepness = {steepness.value:.1f})')
        plt.xlabel('Input (x)')
        plt.ylabel('Output (tanh(x))')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add horizontal bounds at -1 and 1
        plt.axhline(y=-1, color='gray', linestyle=':', alpha=0.5)
        plt.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
        # Add zero lines
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

        plot_display = mo.vstack([
            mo.as_html(plt.gca()),
        ])

        return plot_display
    return (plot_tanh,)


@app.cell
def _(controls):
    controls
    return


@app.cell
def _(input_controls):
    input_controls
    return


@app.cell
def _(plot_tanh):
    plot_tanh()
    return


@app.cell
def _(mo):
    conclusion = mo.vstack([
        mo.callout(
            mo.md("""
                **Key Takeaways!** 
                Through this interactive exploration of the tanh function, you've learned:

                - How tanh maps inputs to the range [-1, 1]
                - The effect of steepness on the function's shape
                - The symmetric nature of the function
            """),
            kind="success"
        ),
        mo.accordion({
            "Next Steps": mo.md("""
                1. **Implementation:** Try using tanh in your neural networks
                2. **Compare:** Analyze performance vs sigmoid and other activations
                3. **Experiment:** Study the vanishing gradient problem
                4. **Advanced:** Explore tanh in RNNs and LSTMs
            """),
            "🎯 Common Applications": mo.md("""
                - Recurrent Neural Networks (RNNs)
                - Long Short-Term Memory (LSTM) networks
                - Binary classification tasks
                - When zero-centered outputs are needed
                - Financial time series prediction
            """),
        })
    ])
    return (conclusion,)


@app.cell
def _(mo):
    mo.md(f"""
    This interactive learning experience was designed to help you understand the hyperbolic tangent (tanh) activation function. Hope this helps in your deep learning journey!
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
