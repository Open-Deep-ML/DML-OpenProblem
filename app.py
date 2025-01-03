import streamlit as st
from streamlit_ace import st_ace
import streamlit.components.v1 as components
import os
import re

# Set page configuration to wide layout
st.set_page_config(layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stTextArea textarea {
        font-family: monospace;
    }
    .stCodeBlock {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
    }
    .split-container {
        display: flex;
        gap: 20px;
    }
    .left-panel, .right-panel {
        flex: 1;
    }
    .left-panel {
        max-width: 50%;
    }
    .right-panel {
        max-width: 50%;
    }
    </style>
    """, unsafe_allow_html=True)

def render_learn_section(learn_section):
    # Replace LaTeX delimiters with the appropriate format for MathJax
    learn_section = re.sub(r'\\\(', r'$', learn_section)
    learn_section = re.sub(r'\\\)', r'$', learn_section)
    learn_section = re.sub(r'\\\[', r'$$', learn_section)
    learn_section = re.sub(r'\\\]', r'$$', learn_section)
    
    st.subheader("Problem Description")
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
        height=500,
    )

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
st.title("LeetCode-like Problem Solver")

# Define the directory containing the learn.html files
LEARN_HTML_DIR = "Problems"

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
    selected_file = st.selectbox("Select a problem to solve", html_files, key="file_selector")

    if selected_file:
        # Load the content of the selected file
        content = load_file(selected_file)

        # Split layout into two columns
        col1, col2 = st.columns([1, 1])

        # Left column: Problem description
        with col1:
            render_learn_section(content)

        # Right column: Code editor and execution
        with col2:
            st.subheader("Code Editor")
            editor_key = f"editor_{selected_file}"
            edited_content = st_ace(
                value="// Write your code here",
                language='python',
                theme='monokai',
                key=editor_key,
                height=500  # Increased height for better visibility
            )

            # Run button
            if st.button("Run Code"):
                st.subheader("Output")
                try:
                    # Simulate code execution (for demonstration purposes)
                    exec(edited_content)
                    st.success("Code executed successfully!")
                except Exception as e:
                    st.error(f"Error executing code: {e}")

            # Save button
            if st.button("Save Code"):
                save_file(selected_file, edited_content)
                st.success("Code saved successfully!")

        # Feedback section (below the split layout)
        st.subheader("Feedback")
        feedback = st.text_area("Leave your feedback here")
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback!")