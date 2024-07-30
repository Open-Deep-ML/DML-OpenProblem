import streamlit as st
from streamlit_ace import st_ace
import streamlit.components.v1 as components
import os
import re

def render_learn_section(learn_section):
    # Replace LaTeX delimiters with the appropriate format for MathJax
    learn_section = re.sub(r'\\\(', r'$', learn_section)
    learn_section = re.sub(r'\\\)', r'$', learn_section)
    learn_section = re.sub(r'\\\[', r'$$', learn_section)
    learn_section = re.sub(r'\\\]', r'$$', learn_section)
    
    st.subheader("Learn Section Preview")
    components.html(
        f"""
        <div id="mathjax-preview">
            {learn_section}
        </div>
        <script>
            window.MathJax = {{
                tex: {{
                    inlineMath: [['$', '$'], ['\\(', '\\)']],
                    displayMath: [['$$', '$$'], ['\\[', '\\]']]
                }},
                svg: {{
                    fontCache: 'global'
                }}
            }};
            document.addEventListener("DOMContentLoaded", function() {{
                MathJax.typesetPromise();
            }});
        </script>
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" defer></script>
        """,
        height=1000,
    )

# Define the directory containing the learn.html files
LEARN_HTML_DIR = "Problems"

def load_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        st.error(f"Error loading file {file_path}: {e}")
        return ""

def save_file(file_path, content):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
    except Exception as e:
        st.error(f"Error saving file {file_path}: {e}")

# Streamlit app
st.title("Learn Section Editor")

# List HTML files in the directory
html_files = []
for root, dirs, files in os.walk(LEARN_HTML_DIR):
    for file in files:
        if file.endswith(".html"):
            html_files.append(os.path.join(root, file))

if not html_files:
    st.warning("No learn.html files found.")
else:
    # File selector
    selected_file = st.selectbox("Select an HTML file to edit", html_files)

    if selected_file:
        # Load the content of the selected file
        content = load_file(selected_file)

        # Use the code editor for editing the content
        edited_content = st_ace(value=content, language='html', theme='monokai', key="editor")

        if st.button("Save changes"):
            save_file(selected_file, edited_content)
            st.success(f"Changes saved to {selected_file}")
            st.session_state["rendered_html"] = edited_content

        # Render the content
        if "rendered_html" in st.session_state:
            render_learn_section(st.session_state["rendered_html"])
        else:
            render_learn_section(content)
