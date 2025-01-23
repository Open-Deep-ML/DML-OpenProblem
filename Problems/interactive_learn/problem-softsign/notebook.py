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
        # Understanding the Softsign Activation Function

        The [Softsign activation function](https://www.gabormelli.com/RKB/Softsign_Activation_Function) is a smooth, bounded function that can be used in neural networks. 
        Let's explore its properties interactively!
        """
    ).center()
    return


@app.cell
def _(mo):
    value = mo.md(r"""
    For an input $x$, the Softsign function is defined as:

    \[
    \text{Softsign}(x) = \frac{x}{1 + |x|}
    \]

    For comparison, the hyperbolic tangent (tanh) function is defined as:

    \[
    \text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
    \]

    While both functions map inputs to the range (-1, 1), Softsign approaches its asymptotes more slowly than tanh,
    which can help maintain better gradient flow during training.
    """)
    mo.accordion({"### Mathematical Definition" : value})
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
            The Softsign function:
            
            - Approaches ±1 asymptotically
            
            - Has a smoother curve compared to tanh
            
            - Maintains non-zero gradients even for larger inputs
            
            - Shows slower saturation than tanh
        """),
        kind="info"
    )

    controls = mo.hstack([
        mo.vstack([
            mo.md("### Adjust Parameters"),
            x_range,
            mo.accordion({
                "About Range": "Adjust to see different regions of the function and how it approaches its asymptotes.",
            })
        ]), _callout.right()
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
            "About Testing": "Enter specific values to see their Softsign outputs."
        })
    ])
    return input_controls, test_input


@app.cell
def _(mo):
    formula_display = mo.vstack([
        mo.md(
            r"""
            ### Current Softsign Configuration

            The Softsign function is defined as:

            \[
            f(x) = \frac{x}{1 + |x|}
            \]

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
    def plot_softsign():
        if x_range.value[0] >= x_range.value[1]:
            raise ValueError("Invalid x_range: start value must be less than stop value.")

        x = np.linspace(x_range.value[0], x_range.value[1], 1000)
        y_softsign = x / (1 + np.abs(x))
        y_tanh = np.tanh(x)

        plt.figure(figsize=(12, 7))

        # Plot tanh first (as reference)
        plt.plot(x, y_tanh,
                label='tanh',
                color='red',
                linewidth=2,
                linestyle='--',
                alpha=0.6)

        # Plot Softsign
        plt.plot(x, y_softsign,
                label='Softsign',
                color='blue',
                linewidth=2)

        # Plot test point if within range (adjust sliders accordingly)
        if x_range.value[0] <= test_input.value <= x_range.value[1]:
            test_output = test_input.value / (1 + abs(test_input.value))
            plt.scatter([test_input.value], [test_output], 
                       color='green', s=100, 
                       label=f'Test point: f({test_input.value:.2f}) = {test_output:.2f}')

        plt.grid(True, alpha=0.3)
        plt.title('Softsign Activation Function')
        plt.xlabel('Input (x)')
        plt.ylabel('Output (Softsign(x))')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add zero lines and asymptotes
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.axhline(y=1, color='r', linestyle=':', alpha=0.3, label='Upper asymptote (y=1)')
        plt.axhline(y=-1, color='r', linestyle=':', alpha=0.3, label='Lower asymptote (y=-1)')

        plot_display = mo.vstack([
            mo.as_html(plt.gca()),
        ])

        return plot_display
    return (plot_softsign,)


@app.cell
def _(controls):
    controls
    return


@app.cell
def _(input_controls):
    input_controls
    return


@app.cell
def _(plot_softsign):
    plot_softsign()
    return


@app.cell
def _(mo):
    conclusion = mo.vstack([
        mo.callout(
            mo.md("""
                **Congratulations!** 
                You've explored the Softsign activation function interactively. You've learned:

                - How Softsign bounds outputs between -1 and 1
                - Its smooth behavior and slower saturation
                - Why it can be an alternative to tanh
            """),
            kind="success"
        ),
        mo.accordion({
            "Next Steps": mo.md("""
                1. **Implementation:** Try implementing Softsign from scratch
                2. **Compare:** Explore how it differs from tanh
                3. **Experiment:** Test it in a neural network
                4. **Advanced:** Learn about other bounded activation functions
            """),
            "🎯 Common Applications": mo.md("""
                - Deep neural networks
                - When smooth gradients are important
                - Situations where bounded outputs are needed
                - Alternative to tanh activation
                - When slower saturation is beneficial
            """),
        })
    ])
    return (conclusion,)


@app.cell
def _(mo):
    mo.md(f"""
    This interactive learning experience was designed to help you understand the Softsign activation function. Hope this helps in your deep learning journey!
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
