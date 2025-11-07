# Personal Loan Marketing Dashboard (Streamlit)

This package contains a Streamlit app (app.py) that trains ML models (Decision Tree, Random Forest, Gradient Boosting) on the Universal Bank dataset to predict whether a customer will accept a personal loan.

How to use
1. Create a new GitHub repository and upload all files from this ZIP to the root (no folders).
2. In Streamlit Cloud, connect the GitHub repo and set the main file to `app.py`.
3. Streamlit Cloud will install dependencies from `requirements.txt` (no versions specified).
4. Run the app. You can use the provided `UniversalBank.csv` sample dataset or upload your own CSV.

Files included
- `app.py` : Main Streamlit application
- `UniversalBank.csv` : Sample dataset
- `requirements.txt` : Packages required
- `README.md` : This file

Notes and tips
- The app trains models on the fly. For larger datasets you may want to increase resources or pre-train and load models.
- Ensure uploaded datasets have the same column names as the sample (or compatible names). The app attempts simple preprocessing.
