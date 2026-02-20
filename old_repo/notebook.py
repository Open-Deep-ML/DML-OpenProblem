# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.2.4",
#     "pandas==2.2.3",
#     "plotly==6.0.1",
#     "wigglystuff==0.1.12",
# ]
# ///

import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    # Key concepts accordion

    insights = mo.accordion(
        {
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
        """),
            "🔄 Covariance vs. Correlation": mo.md("""
        **Important Distinction:**

        1. **Covariance**: Depends on the scale of the variables
           - Changes if you convert units (e.g., inches to cm)
           - No fixed range of values
           - Hard to interpret magnitude directly

        2. **Correlation**: Normalized covariance (scale-independent)
           - Always ranges from -1 to 1
           - Easier to interpret: 1 (perfect positive), -1 (perfect negative), 0 (no relationship)
           - Formula: $corr(X,Y) = \\frac{cov(X,Y)}{\\sigma_X \\sigma_Y}$
        """),
        }
    )
    return (insights,)


@app.cell(hide_code=True)
def _(mo):
    # Practice exercises

    exercises = mo.accordion(
        {
            "🎯 Practice Exercises": mo.md("""

        Try these examples to understand covariance better:

        1. Enter perfectly correlated data: [1,2,3] and [2,4,6]
           - Expected: Strong positive correlation (r = 1.0)
           - Covariance depends on scale

        2. Enter negatively correlated data: [1,2,3] and [3,2,1]
           - Expected: Strong negative correlation (r = -1.0) 
           - Note: covariance will be negative

        3. Enter uncorrelated data: [1,2,3] and [2,2,2]
           - Expected: No correlation (r = 0.0)
           - Covariance should be close to zero

        What do you notice about the covariance vs correlation matrices?

        """),
            "💡 Tips for Interpretation": mo.md("""

        - Large positive values: Strong positive relationship

        - Large negative values: Strong negative relationship

        - Values near zero: Weak or no relationship

        - Diagonal values: Spread of individual variables

        - **Comparing scales:** 
          - Correlation is easier to interpret consistently (-1 to 1)
          - Covariance magnitude depends on your variable units

        """),
        }
    )
    return (exercises,)


@app.cell
def _(calculate_button, data_matrix, mo, np, pd, px):
    results = None
    if calculate_button.value:
        try:
            # 1. Get input data from above
            data = np.array(data_matrix.matrix, dtype=float)
            if data.shape[0] != 2:
                raise ValueError("Data must have exactly 2 rows (variables)")

            # Check if we have enough data points
            if data.shape[1] < 2:
                raise ValueError("Need at least 2 data points for analysis")

            # 2. key components calc
            means = np.mean(data, axis=1)
            centered_data = data - means[:, np.newaxis]
            cov_matrix = np.cov(data)

            # 3. visualization with covariance matrix
            df = pd.DataFrame({"x": data[0], "y": data[1]})

            scatter_fig = px.scatter(
                df,
                x="x",
                y="y",
                title="Variable Relationship Pattern",
                labels={"x": "Variable 1", "y": "Variable 2"},
            ).update_layout(width=400, height=400, showlegend=False)

            # appropriate trendline - with error handling for edge cases
            try:
                if len(data[0]) >= 2 and len(np.unique(data[0])) >= 2:
                    coeffs = np.polyfit(data[0], data[1], 1)
                    x_range = np.linspace(min(data[0]), max(data[0]), 100)
                    scatter_fig.add_trace(
                        dict(
                            type="scatter",
                            x=x_range,
                            y=coeffs[0] * x_range + coeffs[1],
                            mode="lines",
                            line=dict(color="red", dash="dash"),
                            name="Trend",
                        )
                    )
                else:
                    coeffs = (0, 0)  # Default no slope
                    x_range = (
                        np.array([min(data[0]), max(data[0])])
                        if len(data[0]) > 0
                        else np.array([0, 1])
                    )
            except Exception:
                coeffs = (0, 0)  # Default fallback
                x_range = np.array([0, 1])

            # Calculate correlation coefficient
            corr_matrix = np.corrcoef(data)

            # 4. results with relevant explanations
            results = mo.vstack(
                [
                    mo.md("## Understanding Your Data's Covariance"),
                    # First row: Plot and Matrix
                    mo.hstack(
                        [
                            # scatter plot
                            mo.vstack([scatter_fig]),
                            # covariance matrix
                            mo.vstack(
                                [
                                    mo.md(
                                        r"""
                        **Covariance Matrix:**

                        $$
                        C = \begin{pmatrix} 
                        %.2f & %.2f \\
                        %.2f & %.2f
                        \end{pmatrix}
                        $$
                        """
                                        % (
                                            cov_matrix[0, 0],
                                            cov_matrix[0, 1],
                                            cov_matrix[1, 0],
                                            cov_matrix[1, 1],
                                        )
                                    ),
                                    mo.md(
                                        r"""
                        **Correlation Matrix:**

                        $$
                        R = \begin{pmatrix} 
                        %.2f & %.2f \\
                        %.2f & %.2f
                        \end{pmatrix}
                        $$
                        """
                                        % (
                                            corr_matrix[0, 0],
                                            corr_matrix[0, 1],
                                            corr_matrix[1, 0],
                                            corr_matrix[1, 1],
                                        )
                                    ),
                                ]
                            ),
                        ]
                    ),
                    # interpretation and insights side by side
                    mo.hstack(
                        [
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
                                kind="info",
                            ),
                            # Right: Key Insights
                            mo.callout(
                                mo.md(f"""
                        **Key Insights:**


                        1. Relationship: {"Positive" if cov_matrix[0, 1] > 0 else "Negative" if cov_matrix[0, 1] < 0 else "No"} covariance

                        2. Strength: {"Strong" if abs(corr_matrix[0, 1]) > 0.7 else "Moderate" if abs(corr_matrix[0, 1]) > 0.3 else "Weak"} correlation ({corr_matrix[0, 1]:.2f})

                        3. Variances: ({cov_matrix[0, 0]:.2f}, {cov_matrix[1, 1]:.2f})

                        **Centered Data:**
                        ```python

                        Var1: {np.round(centered_data[0], 2)}

                        Var2: {np.round(centered_data[1], 2)}
                        ```
                        """),
                                kind="neutral",
                            ),
                        ]
                    ),
                ],
                justify="center",
            )

        except Exception as e:
            results = mo.md(f"⚠️ Error: {str(e)}").callout(kind="danger")
            # Initialize variables to None to avoid reference errors in case of exception
            centered_data = coeffs = cov_matrix = data = df = means = scatter_fig = (
                x_range
            ) = corr_matrix = None
    results
    return (
        centered_data,
        coeffs,
        cov_matrix,
        corr_matrix,
        data,
        df,
        means,
        results,
        scatter_fig,
        x_range,
    )


@app.cell(hide_code=True)
def _(mo):
    # Conclusion
    conclusion = mo.vstack(
        [
            mo.callout(
                mo.md("""
            **Congratulations!** 

            You've mastered the key concepts of covariance matrices:
        
            - How to calculate covariance between variables
            - How to interpret the covariance matrix
            - The relationship between covariance and correlation
            - How to visualize relationships in data
            - The importance of centered data

            """),
                kind="success",
            ),
            mo.accordion(
                {
                    "🚀 Next Steps": mo.md("""

            1. Work with multivariate datasets (3+ variables)
            2. Apply to real-world datasets with different scales
            3. Use in dimensionality reduction (PCA)
            4. Implement in machine learning projects

            """)
                }
            ),
        ]
    )
    return (conclusion,)
