# Plant Seedling Classifier

This project implements a machine learning pipeline to classify images of plant seedlings. Utilizing deep learning techniques and image processing, it aims to accurately identify different species of seedlings, which can be crucial for agricultural automation and research. The project leverages popular Python libraries such as TensorFlow, scikit-learn, and OpenCV for model development, training, and evaluation.

---

## 🚀 Setup & Execution

Follow these steps to isolate your dependencies, configure your environment, and execute the analytical pipeline.

### 1. Clone & Navigate to the Project Directory
```sh
git clone <your-repository-url>
cd plant-seedling-classifier
```

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
- **PyCharm/IntelliJ IDEA**: Go to `Settings > Project > Python Interpreter` and select the `python` executable inside your `venv` folder.
- **VS Code**: Type `Cmd+Shift+P`, search for `Python: Select Interpreter`, and choose the one in your project `venv`.

### 5. Install Dependencies
Install the required data science libraries. For macOS users, `tensorflow-metal` will be automatically installed to leverage GPU acceleration on Apple Silicon, as specified in `requirements.txt`.
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
