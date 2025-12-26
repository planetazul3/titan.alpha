# Kaggle Training Instructions

## 1. Prepare Data & Code

You have two options for the project files:

### Option A: Upload Code as Dataset (Recommended)
1.  Zip your `xtitan` folder (excluding `venv` and `__pycache__`).
2.  Create a **New Dataset** on Kaggle.
3.  Upload the `xtitan.zip`.

### Option B: Upload Raw Data (Optional)
If you want to save download time:
1.  Zip your `data_cache/` folder.
2.  Add it to the same Dataset or a new one.

---

## 2. Setup the Notebook

1.  Create a **New Notebook** in Kaggle.
2.  **Settings:**
    *   **Accelerator:** GPU P100 or T4 x2.
    *   **Internet:** ON (REQUIRED for installations).
    *   **Persistence:** Files in `/kaggle/working` are output.

3.  **Add Input:**
    *   Add the Dataset you created in Step 1.
    *   It will appear under `/kaggle/input/`.

---

## 3. Run the Training

Open the `kaggle_training.ipynb` (included in this repo) or copy the cells below.

**Key Steps performed by the notebook:**
1.  **Install TA-Lib:** Compiles and installs the required C-library.
2.  **Install Dependencies:** Installs `requirements.txt`.
3.  **Download Data:** (If not uploaded) Downloads from Deriv.
4.  **Train:** Runs the training pipeline with GPU acceleration.

## 4. Download Results

After training:
1.  The notebook zips `checkpoints/` and `logs/`.
2.  Download `training_artifacts.zip` from the **Output** tab.
