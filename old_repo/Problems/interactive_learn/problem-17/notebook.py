# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "drawdata==0.3.6",
#     "marimo",
#     "numpy==2.2.1",
#     "pandas==2.2.3",
#     "plotly==5.24.1",
#     "scikit-learn==1.6.0",
# ]
# ///

import marimo

__generated_with = "0.10.12"
app = marimo.App(css_file="/Users/adityakhalkar/Library/Application Support/mtheme/themes/deepml.css")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        f"""# Understanding K-means Clustering

        ## Overview
        K-means clustering is an [unsupervised learning algorithm](https://en.wikipedia.org/wiki/Unsupervised_learning) that partitions data into k distinct clusters. Each cluster is characterized by its centroid - the mean position of all points in that cluster.
        """
     )
    return


@app.cell(hide_code=True)
def _(intro):
    intro
    return


@app.cell(hide_code=True)
def _(mo):
    intro = mo.accordion({
        "🔄 Algorithm Steps": mo.md("""
            1. **Initialization**: Randomly place k centroids in the feature space
            2. **Assignment**: Assign each point to the nearest centroid using Euclidean distance (or a suitable distance metric like Manhattan (City Block distance), Minkowski distance, etc.):

           \\[
           d(x_i, \\mu_j) = \\sqrt{\\sum_{d=1}^{D} (x_{id} - \\mu_{jd})^2}
           \\]

            3. **Update**: Recompute centroids as the mean of assigned points
            4. **Repeat**: Steps 2-3 until convergence
            """),
        "📐 Mathematical Formulation": mo.md("""
            The objective function (inertia) that K-means minimizes:

            \\[
            J = \\sum_{i=1}^{n} \\min_{j \\in \\{1,\\ldots,k\\}} ||x_i - \\mu_j||^2
            \\]

            where:
            - $x_i$ is the i-th data point
            - $\\mu_j$ is the centroid of cluster $j$
            """)
    })

    return (intro,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(f"""
    ## Implementation Details

    This implementation uses:

    * **Distance Metric**: [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) (L2 norm)

    * **Initialization**: [k-means++](https://en.wikipedia.org/wiki/K-means%2B%2B) by default for better starting positions
    """)
    return


@app.cell
def _(implementation):
    implementation
    return


@app.cell(hide_code=True)
def _(mo):
    implementation = mo.accordion({
            "⚙️ Key Parameters": mo.md("""
            - **n_clusters**: Number of clusters (k)
            - **init**: Initialization method ('k-means++' or 'random')
            - **max_iter**: Maximum iterations (default=300)
            - **tol**: Tolerance for declaring convergence (default=1e-4)
            - **n_init**: Number of initializations to try (default=10)
            """)
        })
    return (implementation,)


@app.cell
def _(mo):
    method = mo.ui.dropdown(
        options=["Random", "Manual"],
        value="Random",
        label="Generation Method"
    )
    return (method,)


@app.cell
def _(method):
    method
    return


@app.cell
def _(mo):
    points = mo.ui.number(value=200, start=10, stop=1000, label="Number of Points")
    return (points,)


@app.cell
def _(mo):
    k_clusters = mo.ui.number(value=5, start=2, stop=15, label="Number of Clusters")
    k_clusters
    return (k_clusters,)


@app.cell
def _(mo):
    random_button = mo.ui.button(label="Generate new data")
    return (random_button,)


@app.cell
def _(run_button):
    run_button
    return


@app.cell
def _(method, mo, random_button, widget):
    random_button if method.value == "Random" else mo.md(
        f"""
        Draw a dataset of points, then click the run button above!

        {widget}
        """
    )
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run k-means!")
    return (run_button,)


@app.cell
def _(np, random, random_button):
    random_button


    def _generate_data():
        n_clusters = random.randint(2, 10)
        np.random.randn()

        points = []
        for i in range(n_clusters):
            points.append(
                np.random.randn(100, 2) * np.random.uniform(-2, 2)
                + np.random.uniform(-2, 2)
            )
        return np.vstack(points)


    generated_points = _generate_data()
    return (generated_points,)


@app.cell
def _(
    KMeans,
    generated_points,
    k_clusters,
    method,
    mo,
    np,
    pd,
    px,
    run_button,
    widget,
):
    fig = (
        px.scatter(
            x=generated_points[:, 0],
            y=generated_points[:, 1],
            title="Random Points",
        )
        if method.value == "Random"
        else None
    )

    if run_button.value and method.value == "Random":
        kmeans = KMeans(n_clusters=k_clusters.value, random_state=42)
        clusters = kmeans.fit_predict(generated_points)
        df = pd.DataFrame(generated_points, columns=["x", "y"])
        df["cluster"] = clusters

        # Create main clustering plot
        cluster_fig = px.scatter(
            df,
            x="x",
            y="y",
            color="cluster",
            title="K-means Clustering Results",
            color_continuous_scale="viridis",
        )
        cluster_fig.update(layout_coloraxis_showscale=False)

        # Add centroids to the plot
        centroids = kmeans.cluster_centers_
        for i, centroid in enumerate(centroids):
            cluster_fig.add_scatter(
                x=[centroid[0]],
                y=[centroid[1]],
                mode="markers",
                marker=dict(size=15, color="red", symbol="x"),
                name=f"Centroid {i}",
            )

        # Create elbow curve
        k_range = range(1, 11)
        inertias = []
        for k in k_range:
            temp_kmeans = KMeans(n_clusters=k, random_state=42)
            temp_kmeans.fit(generated_points)
            inertias.append(temp_kmeans.inertia_)

        elbow_fig = px.line(
            x=list(k_range),
            y=inertias,
            title="Elbow Method Analysis",
            labels={"x": "Number of Clusters (k)", "y": "Inertia"},
        )
        elbow_fig.add_scatter(
            x=list(k_range), y=inertias, mode="markers", name="Inertia Points"
        )

        # Algorithm progress information
        algo_info = mo.accordion(
            {
                "🔍 Algorithm Progress": mo.md(f"""
                **Intermediate Steps:**

                1. Initial random centroids placed

                2. Points assigned to nearest centroid using Euclidean distance

                3. Centroids recomputed {kmeans.n_iter_} times

                4. Algorithm converged with final inertia of {kmeans.inertia_:.2f}
            """)
            }
        )

        callouts_only = mo.md(f"""
        {mo.callout(f"Iterations: {kmeans.n_iter_}", kind="info")}
        {mo.callout(f"Final inertia: {kmeans.inertia_:.2f}", kind="success")}
        """)

        elbow_analysis_steps = mo.accordion(
            {
                "📈 Interpreting the Elbow Curve": mo.md("""
            The elbow curve helps determine the optimal number of clusters (k):

            1. **Finding the elbow**: Look for the point where adding more clusters [gives diminishing returns](https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/#:~:text=The%20elbow%20method%20is%20a%20graphical%20representation%20of%20finding%20the,cluster%20and%20the%20cluster%20centroid.)
            2. Keep trying various value of k in the Number of Clusters field to see how the inertia and iterations change
            3. **Interpretation**:

                   - Sharp decrease: Significant improvement in cluster fit

                   - Leveling off: Minimal improvement in fit

                   - Elbow point: Often the optimal k value
            """),
            }
        )

        # Display results vertically
        fig = mo.vstack(
            [
                mo.hstack([cluster_fig, callouts_only]),
                algo_info,
                elbow_fig,
                elbow_analysis_steps,
            ]
        )

    elif run_button.value and method.value == "Manual":
        df = widget.data_as_pandas
        if not df.empty:
            numeric_df = df.select_dtypes(include=[np.number])
            kmeans = KMeans(n_clusters=k_clusters.value, random_state=42)
            clusters = kmeans.fit_predict(numeric_df)
            df["cluster"] = clusters

            # Create main clustering plot
            cluster_fig = px.scatter(
                df,
                x="x",
                y="y",
                color="cluster",
                title="K-means Clustering (Manual Data)",
                color_continuous_scale="viridis",
            )

            # Add centroids to the plot
            centroids = kmeans.cluster_centers_
            for i, centroid in enumerate(centroids):
                cluster_fig.add_scatter(
                    x=[centroid[0]],
                    y=[centroid[1]],
                    mode="markers",
                    marker=dict(size=15, color="red", symbol="x"),
                    name=f"Centroid {i}",
                )

            # Create elbow curve
            k_range = range(1, 11)
            inertias = []
            for k in k_range:
                temp_kmeans = KMeans(n_clusters=k, random_state=42)
                temp_kmeans.fit(numeric_df)
                inertias.append(temp_kmeans.inertia_)

            elbow_fig = px.line(
                x=list(k_range),
                y=inertias,
                title="Elbow Method Analysis (Manual Data)",
                labels={"x": "Number of Clusters (k)", "y": "Inertia"},
            )
            elbow_fig.add_scatter(
                x=list(k_range), y=inertias, mode="markers", name="Inertia Points"
            )

            # Algorithm progress information
            algo_info = mo.accordion(
                {
                    "🔍 Algorithm Progress": mo.md(f"""
                    **Intermediate Steps:**

                    1. Initial random centroids placed

                    2. Points assigned to nearest centroid using Euclidean distance

                    3. Centroids recomputed {kmeans.n_iter_} times

                    4. Algorithm converged with final inertia of {kmeans.inertia_:.2f}
                """)
                }
            )

            callouts_only = mo.md(f"""
            {mo.callout(f"Iterations until convergence: {kmeans.n_iter_}", kind="info")}
            {mo.callout(f"Final inertia: {kmeans.inertia_:.2f}", kind="success")}
            """)

            elbow_analysis_steps = mo.accordion(
                {
                    "📈 Interpreting the Elbow Curve": mo.md("""
                The elbow curve helps determine the optimal number of clusters (k):

                1. **Finding the elbow**: Look for the point where adding more clusters [gives diminishing returns](https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/#:~:text=The%20elbow%20method%20is%20a%20graphical%20representation%20of%20finding%20the,cluster%20and%20the%20cluster%20centroid.)
                2. Keep trying various value of k in the Number of Clusters field to see how the inertia and iterations change
                3. **Interpretation**:

                       - Sharp decrease: Significant improvement in cluster fit

                       - Leveling off: Minimal improvement in fit

                       - Elbow point: Often the optimal k value
                """),
                }
            )

            # Display results vertically
            fig = mo.vstack(
                [
                    mo.hstack([cluster_fig, callouts_only]),
                    algo_info,
                    elbow_fig,
                    elbow_analysis_steps,
                ]
            )

    fig
    return (
        algo_info,
        callouts_only,
        centroid,
        centroids,
        cluster_fig,
        clusters,
        df,
        elbow_analysis_steps,
        elbow_fig,
        fig,
        i,
        inertias,
        k,
        k_range,
        kmeans,
        numeric_df,
        temp_kmeans,
    )


@app.cell
def _(ScatterWidget, mo):
    widget = mo.ui.anywidget(ScatterWidget())
    return (widget,)


@app.cell
def _(mo):
    mo.accordion(
        {
            "🎯 Common Applications": mo.md("""
            - **Customer Segmentation**: Group similar customers for targeted marketing
            - **Image Compression**: Reduce color palette by clustering similar colors
            - **Document Classification**: Group similar documents by content features
            - **Anomaly Detection**: Identify outliers and unusual patterns
            """)
        },
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        **Congratulations!**

        You've successfully explored the k-Means clustering algorithm through this what we hope was an interactive experience. 

        **Next Steps:**

        * **Problem solving:** Head over to the Problem Decsription tag and start solving the problem!
        * **Experiment:** Try experimenting with different datasets, varying the number of clusters (kk), and observing the impact on the clustering results.
        * **Explore:** Delve deeper into the nuances of k-Means, such as initialization strategies (e.g., k-means++), handling outliers, and its limitations.
        * **Apply:** Apply k-Means to real-world problems in common practical applications (see the accordion above for more details).

        **We hope this interactive experience has been a valuable learning journey. Happy clustering!**
        """
    )
    return


@app.cell
def _(mo):
    callout = mo.callout(
        mo.md(
            "This interactive learning experience was designed to help you understand K-Means clustering, a fundamental unsupervised learning algorithm in AI/ML. We hope this resource proves valuable in your exploration of this important topic."
        ),
        kind="success",
    )
    return (callout,)


@app.cell
def _(callout):
    callout
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # import libraries
    import random
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    import plotly.express as px
    from drawdata import ScatterWidget
    return KMeans, ScatterWidget, np, pd, px, random


if __name__ == "__main__":
    app.run()
