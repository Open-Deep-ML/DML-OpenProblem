# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.2.1",
#     "matplotlib==3.10.0",
#     "pillow==11.1.0",
# ]
# ///

import marimo

__generated_with = "0.10.11"
app = marimo.App(css_file="/Users/adityakhalkar/Library/Application Support/mtheme/themes/deepml.css")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Grayscale Image Contrast Calculator

        This interactive tool helps you explore different methods for calculating image contrast in grayscale images. Upload your own image or use our sample to analyze contrast using three different mathematical approaches.
        """
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "📏 Basic Contrast": mo.md(r"""

    The **Basic Contrast** method measures the difference between the maximum and minimum pixel intensity values in an image:

    \[
    C_{\text{Basic}} = I_{\text{max}} - I_{\text{min}}
    \]

    - \( I_{\text{max}} \): Maximum pixel intensity in the image.
    - \( I_{\text{min}} \): Minimum pixel intensity in the image.

    This method is simple and computationally efficient but may not fully capture subtle variations in intensity.
            """),
            "📊 RMS Contrast": mo.md(r"""

    The **Root Mean Square (RMS) Contrast** measures the standard deviation of pixel intensities from their mean. It is given by:

    \[
    C_{\text{RMS}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (I_i - \overline{I})^2}
    \]

    - \( N \): Total number of pixels.
    - \( I_i \): Intensity of pixel \( i \).
    - \( \overline{I} \): Mean intensity of all pixels.

    RMS contrast is sensitive to variations in intensity across the image, making it particularly useful in scenarios where subtle intensity changes are important, such as medical imaging and texture analysis.
            """),
            "🔄 Michelson Contrast": mo.md(r"""


    The **Michelson Contrast** method, commonly used in periodic patterns and signals, is defined as:

    \[
    C_{\text{Michelson}} = \frac{I_{\text{max}} - I_{\text{min}}}{I_{\text{max}} + I_{\text{min}}}
    \]

    - \( I_{\text{max}} \): Maximum pixel intensity.
    - \( I_{\text{min}} \): Minimum pixel intensity.

    This method is often applied in signal and image processing to evaluate the visibility of periodic structures, such as waveforms or stripes.
            """),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🛠️ Usage Instructions": mo.md(
                """
                    1. Choose your image source from the dropdown
                    2. If uploading, select your image file
                    3. Pick your preferred contrast calculation method
                    4. View the results and histogram analysis
                    """
            ),
        }
    )
    return


@app.cell
def _(source_selector):
    source_selector
    return


@app.cell
def _(mo):
    source_selector = mo.ui.dropdown(
        options=["Use Sample Image", "Upload Image"],
        value="Use Sample Image",
        label="Choose Image Source",
    )
    return (source_selector,)


@app.cell
def _(file_upload, source_selector):
    file_upload if source_selector.value == "Upload Image" else None
    return


@app.cell
def _(mo):
    file_upload = mo.ui.file(
        filetypes=[".png", ".jpg"], multiple=False, label="Upload an Image"
    )
    return (file_upload,)


@app.cell(hide_code=True)
def _(Image, io, mo, np):
    def compress_image(img, max_size=(400, 400)):
        """Compress image while maintaining aspect ratio"""
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        return img


    @mo.cache
    def process_image(image_data=None, use_sample=False):
        # TODO: Image path needs to be fixed for sample image; rerun when new version is rolled out
        img_path = mo.notebook_location() / "public" / "marimo x deep-ml.png"
        if use_sample:
            img = Image.open(img_path).convert("L")
        elif image_data:
            img = Image.open(io.BytesIO(image_data)).convert("L")
        else:
            return None, None

        img = compress_image(img)
        img_array = np.array(img)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True, quality=85)
        buf.seek(0)
        img_display = mo.image(buf.read(), width=400)

        return img_array, img_display


    def calculate_contrast(img_array, method):
        if img_array is None:
            return 0

        if method == "Basic Contrast":
            return np.max(img_array) - np.min(img_array)
        elif method == "RMS Contrast":
            mean = np.mean(img_array)
            return np.sqrt(np.mean((img_array - mean) ** 2))
        else:  # Michelson Contrast
            Imax, Imin = np.max(img_array), np.min(img_array)
            return (Imax - Imin) / (Imax + Imin + 1e-6)
    return calculate_contrast, compress_image, process_image


@app.cell
def _(file_upload, mo, process_image, source_selector):
    # Use selected source
    use_sample = source_selector.value == "Use Sample Image"
    img_data = file_upload.value[0].contents if file_upload.value else None

    info_callout = mo.callout("Please upload an image to proceed 📸", kind="warn")

    # Stop execution if conditions aren't met
    mo.stop(
        source_selector.value == "Upload Image" and not img_data,
        info_callout,
    )

    # Process the image
    img_array, img = process_image(img_data, use_sample)
    return img, img_array, img_data, info_callout, use_sample


@app.cell
def _(calculate_contrast, img_array, io, mo, np, plt):
    # Ensure image array exists
    mo.stop(
        img_array is None,
        mo.md("No image data available to display the histogram."),
    )

    # Generate histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(img_array.ravel(), bins=256, color="gray", alpha=0.7)
    ax.set_title("Pixel Intensity Histogram")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    histogram_display = mo.image(buf.read(), width=500)

    # Calculate contrasts for all methods
    basic_contrast = calculate_contrast(img_array, "Basic Contrast")
    rms_contrast = calculate_contrast(img_array, "RMS Contrast")
    michelson_contrast = calculate_contrast(img_array, "Michelson Contrast")

    # Create content for each tab
    basic_tab = mo.vstack(
        [
            mo.md(rf"""

            Formula:

            \[
            C = I_{{\text{{max}}}} - I_{{\text{{min}}}}
            \]

            Values:

            \[
            C = {np.max(img_array):.2f} - {np.min(img_array):.2f} = {basic_contrast:.2f}
            \]
        """),
            histogram_display,
        ]
    )

    rms_tab = mo.vstack(
        [
            mo.md(rf"""

            Formula:

            \[
            C = \sqrt{{\frac{{1}}{{N}} \sum (I - \overline{{I}})^2}}
            \]

            Expanded:

            \[
            C = \sqrt{{\frac{{1}}{{{img_array.size}}} \sum (I - {np.mean(img_array):.2f})^2}}
            \]

            Values:

            \[
            C = {rms_contrast:.2f}
            \]
        """),
            histogram_display,
        ]
    )


    michelson_tab = mo.vstack(
        [
            mo.md(rf"""

            Formula:

            \[
            C = \frac{{I_{{\text{{max}}}} - I_{{\text{{min}}}}}}{{I_{{\text{{max}}}} + I_{{\text{{min}}}}}}
            \]

            Values:

            \[
            C = \frac{{{np.max(img_array):.2f} - {np.min(img_array):.2f}}}{{{np.max(img_array):.2f} + {np.min(img_array):.2f}}} = {michelson_contrast:.2f}
            \]
        """),
            histogram_display,
        ]
    )

    # Create tabs
    method_tabs = mo.ui.tabs(
        {
            "Basic Contrast": basic_tab,
            "RMS Contrast": rms_tab,
            "Michelson Contrast": michelson_tab,
        },
        lazy=True,
    )

    # Final display layout
    display = method_tabs
    return (
        ax,
        basic_contrast,
        basic_tab,
        buf,
        display,
        fig,
        histogram_display,
        method_tabs,
        michelson_contrast,
        michelson_tab,
        rms_contrast,
        rms_tab,
    )


@app.cell
def _(img):
    img.center()
    return


@app.cell
def _(display):
    display.center()
    return


@app.cell
def _(mo):
    conclusion = mo.md(f"""
        **Congratulations!** 
        You've explored the Grayscale Image Contrast Calculator interactively. 

        Now that you understand how image contrast works, head over to the Problem Description tab to solve the challenge!

        You'll apply these concepts to:

        - Calculate contrast for a given grayscale image

        - Understand the relationship between pixel values and contrast

        - Implement your own contrast calculation function
        """)
    return (conclusion,)


@app.cell
def _(mo):
    mo.accordion({
        "🌟 Applications and Relevance": mo.md(r"""
    ### Relevance in Signal and Image Processing

    Contrast calculations play a vital role in enhancing image quality for better visualization and analysis. They are used in:

    - **Signal Processing** 📡: Contrast metrics help in analyzing waveforms and identifying anomalies in signals.
    - **Image Enhancement** ✨: Techniques like histogram equalization leverage contrast metrics to improve image clarity.
    - **Computer Vision** 🤖: Object detection, edge detection, and segmentation algorithms rely on contrast to distinguish objects and features.

    Contrast is a cornerstone of image and signal processing, enabling improved functionality across diverse fields from healthcare to autonomous systems.
            """),
    })
    return


@app.cell
def _(conclusion, mo):
    callout = mo.callout(conclusion, kind="success")
    return (callout,)


@app.cell
def _(callout, img_data, mo, source_selector):
    mo.stop(source_selector.value == "Upload Image" and not img_data, mo.md(""))
    callout
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import io
    return Image, io, np, plt


if __name__ == "__main__":
    app.run()
