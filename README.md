# Deep-ML Open Problem Bank

A community-maintained collection of machine learning coding challenges.  
Each problem lives in its own folder (`questions/<id>_<slug>/`) so contributors can edit Markdown, Python, and JSON files naturally.  
A build script assembles everything into a single JSON file used by [deep-ml.com](https://deep-ml.com).

---

## 📁 Repository Layout

```
.
├─ questions/
│  ├─ _template/                ← Copy this to start a new problem
│  ├─ 101_grpo_objective/
│  │   ├─ meta.json
│  │   ├─ description.md
│  │   ├─ learn.md
│  │   ├─ starter_code.py
│  │   ├─ solution.py
│  │   ├─ example.json
│  │   ├─ tests.json
│  │   ├─ tinygrad/
│  │   │   ├─ starter_code.py
│  │   │   ├─ solution.py
│  │   │   └─ tests.json
│  │   └─ pytorch/
│  │       ├─ starter_code.py
│  │       ├─ solution.py
│  │       └─ tests.json
│  └─ ...
│
├─ schemas/
│  └─ question.schema.json     ← JSON-Schema used for validation
│
├─ utils/
│  ├─ build_bundle.py          ← folder → build/*.json bundler
│  ├─ validate_questions.py    ← schema validator
│  └─ make_question_template.py← template folder generator
│
└─ .github/workflows/
   └─ format_questions.yml     ← GitHub Action: validate on PR/push
```

---

## 🛠️ Adding a New Question

1. **Copy the template**

```bash
cp -r questions/_template questions/123_my_problem
```

2. **Fill in the fields**

- `meta.json`: question ID, title, category, difficulty, etc.
- `description.md`: problem statement
- `learn.md`: explanation and background
- `starter_code.py`, `solution.py`: reference implementation
- `example.json`: input/output + reasoning
- `tests.json`: list of `{ "test": "...", "expected_output": "..." }`
- Optional language support under `tinygrad/` and `pytorch/`

3. **Run local validation**

```bash
python utils/build_bundle.py && python utils/validate_questions.py
```

4. **Open a Pull Request**

CI will build and validate your changes automatically.

---

## 🧪 Schema Validation

The schema ensures:

- Required fields are present
- Optional `tinygrad_*`, `pytorch_*` are allowed
- No invalid or extra fields

Each question must pass validation before it can be merged.

---

## 🤖 GitHub Actions

Located in `.github/workflows/format_questions.yml`, this runs:

1. `build_bundle.py` – compiles all question folders
2. `validate_questions.py` – checks for schema and structure errors

CI fails if anything is invalid.

---

## 📜 License

All problems are for **educational use only**.  
See `LICENSE` file for full terms. 

---

## 🙋 Need Help?

Open an issue or visit our Discord: https://discord.gg/JwMePfMZAV
