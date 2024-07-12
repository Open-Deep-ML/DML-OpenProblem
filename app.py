import ast
import streamlit as st
from streamlit_ace import st_ace
from pistonpy import PistonApp
import streamlit.components.v1 as components
import re
from problems import problems
# Example problems dictionary


# Instantiate the piston client
piston = PistonApp()

def execute_code(user_code):
    # Execute the user code using pistonpy
    result = piston.run(language="python", version="3.10.0", code=user_code)
    return result

def run_test_cases(user_code, test_cases):
    results = []
    for test_case in test_cases:
        # Modify user_code to include the test case at the end
        code_to_run = f"{user_code}\n\n{test_case['test']}"
        result = execute_code(code_to_run)
        
        stdout = result['run']['stdout'].strip()
        expected_output = test_case['expected_output'].strip()

        # Check if the test case passed
        if '{' in stdout:
            passed = ast.literal_eval(stdout) == ast.literal_eval(expected_output)
        else:
            passed = stdout == expected_output
        results.append((test_case['test'], expected_output, stdout, passed))

    return results

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

def output_problem_dict():
    if 'problems' not in st.session_state:
        st.session_state.problems = {}

    problem_id = len(st.session_state.problems) + 1
    problem_dict = {
        'id': problem_id,
        'title': st.session_state.problem_title,
        'description': st.session_state.problem_description,
        'example': st.session_state.problem_example,
        'video': st.session_state.problem_video,
        'learn': st.session_state.problem_learn,
        'starter_code': st.session_state.problem_starter_code,
        'solution': st.session_state.problem_solution,
        'test_cases': [{'test': tc['test'], 'expected_output': tc['expected_output']} for tc in st.session_state.custom_test_cases]
    }

    if 'contributor_name' in st.session_state and st.session_state.contributor_name:
        problem_dict['contributor'] = {
            'name': st.session_state.contributor_name,
            'profile_link': st.session_state.contributor_profile_link
        }

    st.subheader("Generated Python Dictionary:")
    st.code(f"{problem_id}: {repr(problem_dict)}", language="python")

def populate_fields(problem):
    st.session_state.problem_title = problem['title']
    st.session_state.problem_description = problem['description']
    st.session_state.problem_example = problem['example']
    st.session_state.problem_video = problem.get('video', '')
    st.session_state.problem_learn = problem['learn']
    st.session_state.problem_starter_code = problem['starter_code']
    st.session_state.problem_solution = problem['solution']
    st.session_state.custom_test_cases = problem['test_cases']

def reset_fields():
    st.session_state.problem_title = ''
    st.session_state.problem_description = ''
    st.session_state.problem_example = ''
    st.session_state.problem_video = ''
    st.session_state.problem_learn = ''
    st.session_state.problem_starter_code = ''
    st.session_state.problem_solution = ''
    st.session_state.custom_test_cases = []

def app():
    st.title("Problem Creation Platform")
    st.subheader("Create and test new problems for the ML Challenge Platform")

    if 'custom_test_cases' not in st.session_state:
        st.session_state.custom_test_cases = []

    if 'test_results' not in st.session_state:
        st.session_state.test_results = []

    if 'problem_title' not in st.session_state:
        reset_fields()

    st.header("Create or Edit a Problem")

    # Dropdown menu for selecting an existing problem
    problem_selection = st.selectbox("Select a problem to edit or create a new one", [None] + list(problems.keys()), format_func=lambda x: problems[x]['title'] if x else "Create new problem")
    if st.session_state.get('selected_problem') != problem_selection:
        if problem_selection:
            st.session_state.selected_problem = problem_selection
            problem = problems[problem_selection]
            populate_fields(problem)
            st.rerun()  # Reload the page
        else:
            st.session_state.selected_problem = None
            reset_fields()
            st.rerun()  # Reload the page

    # Problem Title
    problem_title = st.text_input("Problem Title", key="problem_title", placeholder="Enter the problem title")

    # Problem Description
    problem_description = st.text_area("Description", key="problem_description", placeholder="Enter the problem description")

    # Problem Example
    problem_example = st.text_area("Example", key="problem_example", placeholder="Enter example inputs and outputs")

    # Problem Video URL
    problem_video = st.text_input("Video URL (optional)", key="problem_video", placeholder="Enter a video URL (optional)")

    # Learn Section
    st.subheader("Learn Section")
    problem_learn = st_ace(language="html", theme="twilight", key="problem_learn", placeholder="Enter the learn section in HTML format", height=200, value=st.session_state.problem_learn)

    if st.button("Preview Learn Section"):
        with st.expander("Learn Section Preview"):
            render_learn_section(st.session_state.problem_learn)

    # Starter Code
    st.subheader("Starter Code")
    problem_starter_code = st_ace(language="python", theme="twilight", key="problem_starter_code", placeholder="Enter starter code", height=200, value=st.session_state.problem_starter_code)

    # Solution Code
    st.subheader("Solution Code")
    problem_solution = st_ace(language="python", theme="twilight", key="problem_solution", placeholder="Enter solution code", height=200, value=st.session_state.problem_solution)

    # Test Cases
    st.subheader("Test Cases")
    if st.button("Add Test Case"):
        st.session_state.custom_test_cases.append({'test': '', 'expected_output': ''})

    for idx, test_case in enumerate(st.session_state.custom_test_cases):
        test_case['test'] = st.text_area(f"Test Case {idx + 1} Code", value=test_case['test'], key=f"test_case_{idx}")
        test_case['expected_output'] = st.text_area(f"Expected Output {idx + 1}", value=test_case['expected_output'], key=f"expected_output_{idx}")

        if st.button(f"Delete Test Case {idx + 1}"):
            st.session_state.custom_test_cases.pop(idx)

    # Run Test Cases Button
    if st.button("Run Test Cases"):
        user_code = st.session_state.problem_solution
        st.session_state.test_results = run_test_cases(user_code, st.session_state.custom_test_cases)

    # Display Test Results
    if st.session_state.test_results:
        st.subheader("Test Results")
        for idx, (test_code, expected_output, actual_output, passed) in enumerate(st.session_state.test_results):
            st.write(f"Test Case {idx + 1}")
            st.code(test_code, language="python")
            st.write(f"Expected Output: {expected_output}")
            st.write(f"Actual Output: {actual_output}")
            st.write(f"Passed: {passed}")

    # Contributor Information
    st.subheader("Contributor Information")
    contributor_name = st.text_input("Contributor Name (optional)", key="contributor_name", placeholder="Enter your name (optional)")
    contributor_profile_link = st.text_input("Profile Link (optional)", key="contributor_profile_link", placeholder="Enter your profile link (optional)")

    if st.button("Generate Python Dictionary"):
        output_problem_dict()

if __name__ == "__main__":
    app()
