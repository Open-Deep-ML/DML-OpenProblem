# **Shifting and Scaling a Range (Rescaling Data)**

## **1. Motivation**

Rescaling (or shifting and scaling) is a common preprocessing step in data analysis and machine learning. It's often necessary to map data from an original range (e.g., test scores, pixel values, GPA) to a new range suitable for downstream tasks or compatibility between datasets. For example, you might want to shift a GPA from $[0, 10]$ to $[0, 4]$ for comparison or model input.

---

## **2. The General Mapping Formula**

Suppose you have input values in the range $[a, b]$ and you want to map them to the interval $[c, d]$.

- First, shift the lower bound to $0$ by applying $x \mapsto x - a$, so $[a, b] \rightarrow [0, b-a]$.
- Next, scale to unit interval: $t \mapsto \frac{1}{b-a} \cdot t$, yielding $[0, 1]$.
- Now, scale to $[0, d-c]$ with $t \mapsto (d-c)t$, and shift to $[c, d]$ with $t \mapsto c + t$.
- Combining all steps, the complete formula is:

$$
    f(x) = c + \left(\frac{d-c}{b-a}\right)(x-a)
$$

- $x$ = the input value
- $a = \min(x)$ and $b = \max(x)$
- $c$, $d$ = target interval endpoints

---

## **3. Applications**
- **Image Processing**: Rescale pixel intensities
- **Feature Engineering**: Normalize features to a common range
- **Score Conversion**: Convert test scores or grades between systems

---

## **4. Practical Considerations**
- Be aware of the case when $a = b$ (constant input); this may require special handling (e.g., output all $c$).
- For multidimensional arrays, use NumPy’s `.min()` and `.max()` to determine the full input range.

---

This formula gives a **simple, mathematically justified way to shift and scale data to any target range**—a core tool for robust machine learning pipelines.
