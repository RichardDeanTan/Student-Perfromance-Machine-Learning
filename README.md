# Student Performance - Machine Learning

This project predicts student performance using machine learning techniques. It includes a deployed web app built with Streamlit for easy interaction.

## 📂 Project Structure

- `app.py` — The main Streamlit app for making predictions.
- `ml-project.ipynb` — The Jupyter Notebook used for data exploration, preprocessing, and model building.
- `StudentPerformanceFactors.csv` — The dataset used for training.
- `best_model.pkl` — The trained model saved using `joblib`.
- `transformer_data.pkl` — Contains the preprocessing pipeline.
- `requirements.txt` — All Python dependencies required to run the project.
- `run-web.txt` — Containts the website link to run the project.

## 🚀 How to Run the App

### Clone the Repository
```bash
git clone https://github.com/RichardDeanTan/Student-Perfromance-Machine-Learning.git
cd Student-Perfromance-Machine-Learning
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Run the Streamlit App
```bash
streamlit run app.py
```
## 💡 Features
Predicts student performance based on various factors.
Interactive web interface using Streamlit.
Model and transformer are preloaded for real-time predictions.

## 📦 Tech Stack
- Python
- Streamlit
- Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn
- Joblib

## 🧠 Model
The model was trained in ml-project.ipynb using:
Data preprocessing pipeline (transformer_data.pkl)
Final model saved as best_model.pkl

## 📝 License
This project is open-source and free to use.
