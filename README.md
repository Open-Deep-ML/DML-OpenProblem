# DML-OpenProblem

DML-OpenProblem is an open-source repository of problems focused on linear algebra, machine learning, and deep learning. The problems are designed to be solved from scratch, providing a robust learning experience. This project powers the website [Deep-ML](https://www.deep-ml.com/).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with DML-OpenProblem, clone the repository and install the necessary dependencies.

```sh
git clone https://github.com/yourusername/DML-OpenProblem.git
cd DML-OpenProblem
pip install -r requirements.txt
```
## Usage
You can use the repository to create, edit, and solve problems related to linear algebra, machine learning, and deep learning. The problems are structured in directories, each containing relevant files such as learn.html for the learning section and solution.py for the solution code.

### Running the Streamlit App

To launch the Streamlit application for editing and viewing problems, use the following command:

```sh
streamlit run app.py
```

#### Features
- Problem Editor: Edit the learn.html and solution.py files for each problem using a web-based code editor.
- Preview Section: Preview the learning section with LaTeX rendering for mathematical expressions.
- Save Changes: Save your edits to the corresponding files in the repository.

## Project Structure
```sh
DML-OpenProblem/
│
├── Problems/
│   ├── 1_matrix_times_vector/
│   │   ├── learn.html
│   │   └── solution.py
│   ├── 2_transpose_matrix/
│   │   ├── learn.html
│   │   └── solution.py
│   └── ... (additional problem directories)
│
├── app.py
├── requirements.txt
└── README.md
```
- **Problems/**: Contains directories for each problem. Each directory includes:
  - `learn.html`: HTML file containing the learning section with explanations and examples.
  - `solution.py`: Python file containing the solution to the problem along with tests.
- **app.py**: The main Streamlit application file for editing the learn sections.
- **requirements.txt**: Lists the dependencies required for the project.
- **README.md**: This file.

## Contributing

We welcome contributions to improve DML-OpenProblem and [deep-ml.com](https://www.deep-ml.com). If you have a new problem to add or improvements to existing problems, please fork the repository and submit a pull request. All contributions will be displayed on [deep-ml.com](https://www.deep-ml.com). For example, check out this problem: [Divide Dataset Based on Feature Threshold](https://www.deep-ml.com/problem/Divide%20Dataset%20Based%20on%20Feature%20Threshold). A helpfull tool to work on the learn section and know what it would look like on the front end is [https://openproblem-r4vsjwuthdl9a3qzrd4p3m.streamlit.app/](https://dml-openproblem-a5bwuwjh2xeyt5ta5wdiw9.streamlit.app/).


### Steps to Contribute

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear and concise messages.
4. Push your changes to your fork.
5. Submit a pull request with a detailed description of your changes.

## License

This project is for educational reasons only. See the [LICENSE](LICENSE) file for details.
