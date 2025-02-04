import marimo

__generated_with = "0.10.19"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Understanding Covariance Matrices Interactively 🔢

        Covariance matrices help us understand how different variables in our data relate to each other. 
        Think of it as a way to measure whether variables "move together" or "move opposite" to each other.

        ### Quick Intuition 💡
        - If two variables tend to increase together → Positive covariance
        - If one increases while other decreases → Negative covariance
        - If they move independently → Covariance near zero

        Let's explore this concept interactively!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # Mathematical definition accordion
    definition = mo.md(r"""

    A covariance matrix captures the relationships between variables through these key components:

    1. **Mean Calculation:**

    \[
    \bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i
    \]

    2. **Covariance Formula:**

    \[
    cov(X,Y) = \frac{1}{n-1}\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})
    \]

    3. **Covariance Matrix Structure:**

    \[
    \begin{bmatrix} 
    cov(X,X) & cov(X,Y) \\
    cov(Y,X) & cov(Y,Y)
    \end{bmatrix}
    \]

    where $X_i, Y_i$ are observations and $n$ is the number of samples.

    """)

    mo.accordion({"### Mathematical Formulation": definition})
    return (definition,)


@app.cell
def _(insights):
    insights
    return


@app.cell(hide_code=True)
def _(mo):
    # Key concepts accordion


    insights = mo.accordion({
        "🎯 Understanding Covariance": mo.md("""
        **Key Concepts:**
        
        1. **Variance**: Measures spread of a single variable
        
        2. **Covariance**: Measures relationship between two variables

        3. **Matrix Properties**: Symmetric, diagonal contains variances

        4. **Interpretation**: Direction and strength of relationships
        """),

        "📊 Matrix Properties": mo.md("""
        The covariance matrix has important properties:

        1. **Symmetry**: cov(X,Y) = cov(Y,X)

        2. **Diagonal Elements**: Represent variances

        3. **Off-diagonal Elements**: Show relationships

        4. **Positive Semi-definite**: All eigenvalues ≥ 0
        """)
    })
    return (insights,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Data Input""")
    return


@app.cell
def _(input_controls):
    input_controls.center()
    return


@app.cell(hide_code=True)
def _(Matrix, mo):
    # interactive matrix input using wigglystuff

    data_matrix = Matrix(
        matrix=[[1, 2, 3], [4, 5, 6]],  # default example data
        rows=2,
        cols=3,
        step=0.1
    )
    input_controls = mo.hstack([
        data_matrix,
    ])
    return data_matrix, input_controls


@app.cell
def _(calculate_button):
    calculate_button.center()
    return


@app.cell
def _(mo):
    calculate_button = mo.ui.run_button(label="Calculate Covariance Matrix")
    return (calculate_button,)


@app.cell
def _(calculate_button, data_matrix, mo, np, pd, px):
    results = None
    if calculate_button.value:
        try:
            # 1. Get input data from above
            data = np.array(data_matrix.matrix, dtype=float)
            if data.shape[0] != 2:
                print("Data must have 2 rows")
                
            # 2. key components calc
            means = np.mean(data, axis=1)
            centered_data = data - means[:, np.newaxis]
            cov_matrix = np.cov(data)
            
            # 3. visualization with covariance matrix
            df = pd.DataFrame({
                'x': data[0],
                'y': data[1]
            })
            
            scatter_fig = px.scatter(
                df,
                x='x',
                y='y',
                title="Variable Relationship Pattern",
                labels={'x': 'Variable 1', 'y': 'Variable 2'}
            ).update_layout(
                width=400,
                height=400,
                showlegend=False
            )

            # appropriate trendline
            coeffs = np.polyfit(data[0], data[1], 1)
            x_range = np.linspace(min(data[0]), max(data[0]), 100)
            scatter_fig.add_trace(
                dict(
                    type='scatter',
                    x=x_range,
                    y=coeffs[0] * x_range + coeffs[1],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Trend'
                )
            )
            # 4. results with relevant explanations
            results = mo.vstack([
                mo.md("## Understanding Your Data's Covariance"),
                
                # First row: Plot and Matrix
                mo.hstack([
                    # scatter plot
                    mo.vstack([scatter_fig]),
                    
                    # covariance matrix
                    mo.vstack([
                        mo.md(r"""
                        **Covariance Matrix:**
                        
                        $$
                        C = \begin{pmatrix} 
                        %.2f & %.2f \\
                        %.2f & %.2f
                        \end{pmatrix}
                        $$
                        """ % (
                            cov_matrix[0,0], cov_matrix[0,1],
                            cov_matrix[1,0], cov_matrix[1,1]
                        ))
                    ])
                ]),
                
                # interpretation and insights side by side
                mo.hstack([
                    # Left: Pattern Interpretation
                    mo.callout(
                        mo.md("""
                        **Pattern Interpretation:**
                        
                        
                        - Upward trend → Positive covariance
                        
                        - Downward trend → Negative covariance
                        
                        - No trend → Zero/Low covariance
                        
                        **Matrix Values:**
                        
                        - Diagonal: Variances show spread
                        
                        - Off-diagonal: Show relationship strength
                        """),
                        kind="info"
                    ),
                    
                    # Right: Key Insights
                    mo.callout(
                        mo.md(f"""
                        **Key Insights:**
                        
                        
                        1. Relationship: {"Positive" if cov_matrix[0,1] > 0 else "Negative" if cov_matrix[0,1] < 0 else "No"} covariance
                        
                        2. Strength: {"Strong" if abs(cov_matrix[0,1]) > 1 else "Moderate" if abs(cov_matrix[0,1]) > 0.5 else "Weak"}
                        
                        3. Variances: ({cov_matrix[0,0]:.2f}, {cov_matrix[1,1]:.2f})
                        
                        **Centered Data:**
                        ```python
                        
                        Var1: {np.round(centered_data[0], 2)}
                        
                        Var2: {np.round(centered_data[1], 2)}
                        ```
                        """),
                        kind="neutral"
                    )
                ])
            ], justify='center')
            
        except Exception as e:
            results = mo.md(f"⚠️ Error: {str(e)}").callout(kind="danger")
    results
    return (
        centered_data,
        coeffs,
        cov_matrix,
        data,
        df,
        means,
        results,
        scatter_fig,
        x_range,
    )


@app.cell
def _(exercises):
    exercises
    return


@app.cell(hide_code=True)
def _(mo):
    # Practice exercises

    exercises = mo.accordion({

        "🎯 Practice Exercises": mo.md("""

        Try these examples to understand covariance better:

        1. Enter perfectly correlated data: [1,2,3] and [2,4,6]

        2. Enter negatively correlated data: [1,2,3] and [3,2,1]

        3. Enter uncorrelated data: [1,2,3] and [2,2,2]

        What do you notice about the covariance matrices?

        """),


        "💡 Tips for Interpretation": mo.md("""

        - Large positive values: Strong positive relationship

        - Large negative values: Strong negative relationship

        - Values near zero: Weak or no relationship

        - Diagonal values: Spread of individual variables

        """)

    })
    return (exercises,)


@app.cell
def _(conclusion):
    conclusion
    return


@app.cell(hide_code=True)
def _(mo):
    # Conclusion

    conclusion = mo.vstack([

        mo.callout(
            mo.md("""
            **Congratulations!** 

            You've mastered the key concepts of covariance matrices:

            - How to calculate covariance between variables

            - How to interpret the covariance matrix

            - How to visualize relationships in data

            - The importance of centered data

            """),
            kind="success"
        ),

        mo.accordion({

            "🚀 Next Steps": mo.md("""

            1. Explore correlation matrices (normalized covariance)

            2. Apply to real-world datasets

            3. Use in dimensionality reduction (PCA)

            4. Implement in machine learning projects

            """)
        })
    ])
    return (conclusion,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import plotly.express as px
    from wigglystuff import Matrix
    import pandas as pd
    return Matrix, np, pd, px


if __name__ == "__main__":
    app.run()
