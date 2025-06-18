# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "wigglystuff==0.1.7",
#     "numpy==2.2.1",
# ]
# ///

import marimo

__generated_with = "0.10.9"
app = marimo.App(css_file="/Users/adityakhalkar/Library/Application Support/mtheme/themes/deepml.css")


@app.cell(hide_code=True)
def problem_description(mo):
    mo.md(
        r"""
        # Matrix Operations: Sums and Means

        We explore how to compute sums and means for the rows and columns of a matrix. Let:

        \[
        A = \begin{bmatrix}
        a_{11} & a_{12} & \cdots & a_{1n} \\
        a_{21} & a_{22} & \cdots & a_{2n} \\
        \vdots & \vdots & \ddots & \vdots \\
        a_{m1} & a_{m2} & \cdots & a_{mn}
        \end{bmatrix}
        \]
        """
    )
    return


@app.cell
def _(column_content, mo, row_content):
    mo.hstack([row_content, column_content]).center()
    return


@app.cell(hide_code=True)
def _(mo):
    row_content = mo.md(
        r'''
        For a row \(i\), the sum is:

        \[
        R_i = \sum_{j=1}^{n} a_{ij}
        \]

        and the mean is:

        \[
        \overline{R}_i = \frac{R_i}{n}
        \]
        '''
    )

    column_content = mo.md(
        r'''
        For a column \(j\), the sum is:

        \[
        C_j = \sum_{i=1}^{m} a_{ij}
        \]

        and the mean is:

        \[
        \overline{C}_j = \frac{C_j}{m}
        \]
        '''
    )
    return column_content, row_content


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Interactive Matrix

        Experiment with the matrix below to understand how row and column operations work. Tip: click and drag to change the matrix values.
        """
    )
    return


@app.cell(hide_code=True)
def interactive_matrix(Matrix, mo, np):
    # Create single matrix widget
    matrix = mo.ui.anywidget(
        Matrix(matrix=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), step=0.1)
    )
    return (matrix,)


@app.cell
def _(matrix):
    matrix.center()
    return


@app.cell
def calculations(matrix, np):
    def calculate_stats(matrix, dimension):
        arr = np.array(matrix)
        if dimension == "row":
            sums = np.sum(arr, axis=1)
            means = np.mean(arr, axis=1)
            labels = [f"Row {i+1}" for i in range(arr.shape[0])]
        else:
            sums = np.sum(arr, axis=0)
            means = np.mean(arr, axis=0)
            labels = [f"Column {i+1}" for i in range(arr.shape[1])]

        return {
            "Dimension": labels,
            "Sum": list(sums),
            "Mean": list(means)
        }

    # Calculate both row and column statistics
    row_stats = calculate_stats(matrix.matrix, "row")
    col_stats = calculate_stats(matrix.matrix, "column")
    return calculate_stats, col_stats, row_stats


@app.cell
def step_by_step_display(col_stats, matrix, mo, np, row_stats):
    arr = np.array(matrix.matrix)


    def generate_row_steps():
        steps = ""
        for i in range(arr.shape[0]):
            row = arr[i]
            sum_expr = " + ".join([f"{x:.1f}" for x in row])
            sum_val = np.sum(row)
            mean_val = np.mean(row)

            steps += rf"""
            \[
            \text{{Row {i+1}}}: {sum_expr} = {sum_val:.1f} \text{{ (sum)}}, \frac{{{sum_val:.1f}}}{{{len(row)}}} = {mean_val:.1f} \text{{ (mean)}}
            \]
            """
        return mo.md(steps)


    def generate_column_steps():
        steps = ""
        for j in range(arr.shape[1]):
            col = arr[:, j]
            sum_expr = " + ".join([f"{x:.1f}" for x in col])
            sum_val = np.sum(col)
            mean_val = np.mean(col)

            steps += rf"""
            \[
            \text{{Column {j+1}}}: {sum_expr} = {sum_val:.1f} \text{{ (sum)}}, \frac{{{sum_val:.1f}}}{{{len(col)}}} = {mean_val:.1f} \text{{ (mean)}}
            \]
            """
        return mo.md(steps)


    # Create tabs for row and column calculations
    tabs_content = {
        "📊 Row Operations": mo.vstack(
            [
                mo.md("### Row-wise Calculations"),
                mo.ui.table(row_stats, show_download=False),
                mo.md("#### Step-by-Step Row Calculations"),
                generate_row_steps(),
            ]
        ),
        "📈 Column Operations": mo.vstack(
            [
                mo.md("### Column-wise Calculations"),
                mo.ui.table(col_stats, show_download=False),
                mo.md("#### Step-by-Step Column Calculations"),
                generate_column_steps(),
            ]
        ),
    }

    mo.ui.tabs(tabs_content)
    return arr, generate_column_steps, generate_row_steps, tabs_content


@app.cell
def _(success_callout):
    success_callout
    return


@app.cell
def _(mo):
    success_callout = mo.callout(
        "🎉 Great job! You've explored the interactive matrix operations. Now you can proceed to the Problem Description tab to solve the problem!",
        kind="success",
    )
    return (success_callout,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    from wigglystuff import Matrix
    return Matrix, np


if __name__ == "__main__":
    app.run()
