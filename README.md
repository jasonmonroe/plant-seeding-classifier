# plant-seeding-classifier
Classified plant seed images


---

## 🚀 Setup & Execution

Follow these steps to isolate your dependencies, configure your environment, and execute the analytical pipeline.

### 1. Clone & Navigate to the Project Directory
```sh
git clone <your-repository-url>
cd plant-seedling-classifier

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to keep dependencies isolated.
```sh
python3 -m venv venv
```

### 3. Activate the Environment
```sh
source venv/bin/activate
```

### 4. Configure your IDE
To resolve "unresolved reference" warnings for built-ins like `print`:
- **PyCharm**: Go to `Settings > Project > Python Interpreter` and select the `python` executable inside your `venv` folder.
- **VS Code**: Type `Cmd+Shift+P`, search for `Python: Select Interpreter`, and choose the one in your project `venv`.

### 5. Install Dependencies
Install the required data science libraries:
```sh
pip install -r requirements.txt
```

### 6. Run the Pipeline
Execute the main script to process the data and generate visualizations:
```sh
python main.py
```

### 7. Deactivate
When you are finished, you can exit the virtual environment by running `deactivate`.

## Troubleshooting

### "zsh: no such file or directory" error
If you see an error pointing to a missing Python path in `/opt/homebrew/`, your virtual environment link is likely broken due to a Homebrew update. To fix it:
```sh
rm -rf venv
python3 -m venv venv
source venv/bin/activate
```