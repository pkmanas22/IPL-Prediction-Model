# 🏏 IPL Match Winner Prediction

This project predicts the winner of an IPL match using machine learning models based on historical match and delivery data.

---

## 📁 Project Structure

```
IPL-PREDICTION/
│
├── .venv/                     # Python virtual environment
├── datasets/                  # Contains CSV datasets
│   ├── deliveries.csv
│   └── matches.csv
│
├── models/                    # Trained ML models
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
│
├── UI/                        # Streamlit web app UI
│   └── app.py
│
├── ipl_prediction.ipynb       # Jupyter notebook for training and experimentation
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Git ignored files
```

---

## 🛠️ Setup Instructions

Follow these steps to set up the project locally:

### 1. Clone the repository

```bash
git clone https://github.com/pkmanas22/IPL-Prediction-Model.git
cd IPL-Prediction-Model
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

**Linux/macOS:**

```bash
source .venv/bin/activate
```

**Windows (CMD):**

```cmd
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Application

```bash
streamlit run UI/app.py
```

---

## 📊 Dataset Source

- `matches.csv` and `deliveries.csv` from [Kaggle IPL dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)

---

## 📌 Notes

- Models are pre-trained and saved in `models/` as `.pkl` files.
- You can retrain models using the Jupyter Notebook: `ipl_prediction.ipynb`.

---

## 🤝 Contributing

Feel free to open issues to improve the model or UI!
