import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
import plotly.express as px
import plotly.graph_objects as go
import io

st.set_page_config(layout="wide", page_title="Personal Loan - Marketing Dashboard")

@st.cache_data
def load_data(path="UniversalBank.csv"):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    if "Personal_Loan" not in df.columns:
        for c in df.columns:
            if c.lower().replace("_","")=="personalloan":
                df = df.rename(columns={c:"Personal_Loan"})
                break
    return df

def train_and_evaluate(df):
    df = preprocess(df)
    drop_cols = [c for c in df.columns if c.lower().startswith("id") or "zip" in c.lower()]
    X = df.drop(columns=drop_cols+["Personal_Loan"], errors='ignore')
    y = df["Personal_Loan"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
    }
    results = {}
    fitted = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = (model, X_train.columns.tolist())
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else model.decision_function(X_test)
        res = {
            "Train Accuracy": accuracy_score(y_train, y_pred_train),
            "Test Accuracy": accuracy_score(y_test, y_pred_test),
            "Precision": precision_score(y_test, y_pred_test, zero_division=0),
            "Recall": recall_score(y_test, y_pred_test, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred_test, zero_division=0),
            "AUC": roc_auc_score(y_test, y_proba_test),
            "CV Accuracy Mean": cross_val_score(model, X, y, cv=skf, scoring='accuracy').mean()
        }
        res["Confusion Train"] = confusion_matrix(y_train, y_pred_train)
        res["Confusion Test"] = confusion_matrix(y_test, y_pred_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba_test)
        res["ROC"] = (fpr, tpr)
        results[name]=res
    return results, fitted, X_train, X_test, y_train, y_test

def metrics_df(results):
    rows = []
    for name, r in results.items():
        rows.append({
            "Algorithm": name,
            "Train Accuracy": r["Train Accuracy"],
            "Test Accuracy": r["Test Accuracy"],
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "F1-Score": r["F1-Score"],
            "AUC": r["AUC"]
        })
    return pd.DataFrame(rows).set_index("Algorithm")

st.title("Personal Loan — Marketing Dashboard (Streamlit)")
st.markdown("Upload dataset or use provided sample to explore and run models.")

df = load_data("UniversalBank.csv")
st.sidebar.header("Controls")
source = st.sidebar.selectbox("Data source", ["Sample dataset (provided)", "Upload CSV"])
if source=="Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success("File loaded")
        except Exception as e:
            st.error("Failed to read CSV: " + str(e))

tab = st.tabs(["Dashboard (Insights)","Train Models & Metrics","Predict & Download","Data Preview & Download"])
with tab[0]:
    st.header("Marketing Insights — 5 strategic charts")
    dfp = preprocess(df)
    st.subheader("1) Conversion rate by Income decile (target higher-income segments)")
    dfp["Income_decile"] = pd.qcut(dfp["Income"], q=10, duplicates='drop')
    dec = dfp.groupby("Income_decile")["Personal_Loan"].agg(['count','mean']).reset_index().sort_values("mean", ascending=False)
    dec.columns = ["Income Decile","Count","Acceptance Rate"]
    fig1 = px.bar(dec, x="Income Decile", y="Acceptance Rate", hover_data=["Count"], title="Acceptance rate by Income decile", labels={"Acceptance Rate":"Acceptance Rate"})
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("Action: target top deciles with specialized offers and pre-approved communications.")

    st.subheader("2) Education level vs Online banking adoption — Acceptance heatmap")
    pivot = dfp.pivot_table(index="Education", columns="Online", values="Personal_Loan", aggfunc='mean')
    pivot = pivot.fillna(0)
    hm = go.Figure(data=go.Heatmap(z=pivot.values, x=["Online=0","Online=1"], y=["Edu=1","Edu=2","Edu=3"], hoverongaps=False, colorbar=dict(title="Acceptance Rate")))
    hm.update_layout(title="Acceptance rate by Education and Online banking", xaxis_title="Online", yaxis_title="Education")
    st.plotly_chart(hm, use_container_width=True)
    st.markdown("Action: if online users in certain education brackets convert highly, prioritize digital channels for those cohorts.")

    st.subheader("3) Income vs CCAvg — size by acceptance & color by family size")
    fig3 = px.scatter(dfp, x="Income", y="CCAvg", color="Family", symbol="Personal_Loan", size="Personal_Loan", hover_data=["Age","Education"], title="Income vs CCAvg (accepted loan highlighted)")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("Action: customers with high CCAvg and mid-to-high income are promising; push targeted cross-sell offers.")

    st.subheader("4) Age distribution by Personal Loan acceptance (who to target)")
    fig4 = px.violin(dfp, y="Age", x="Personal_Loan", box=True, points="all", hover_data=["Income","Education"], labels={"Personal_Loan":"Accepted (1) / No (0)"})
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("Action: tailor messaging by age segments (e.g., younger customers may prefer digital-first offers).")

    st.subheader("5) Feature importance (Random Forest) — actionable drivers")
    try:
        X = dfp.drop(columns=[c for c in dfp.columns if c.lower().startswith("id") or "zip" in c.lower()] + ["Personal_Loan"], errors='ignore')
        y = dfp["Personal_Loan"].astype(int)
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
        fig5 = px.bar(importances, x=importances.values, y=importances.index, orientation='h', title="Top feature importances (Random Forest)")
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("Action: focus marketing levers on the top features (e.g., Income, CCAvg, Education).")
    except Exception as e:
        st.error("Failed to compute feature importances: " + str(e))

with tab[1]:
    st.header("Train models (Decision Tree, Random Forest, Gradient Boosting) and show metrics")
    st.markdown("Click **Train & Evaluate** to run models on current dataset (will perform a train/test split and 5-fold CV).")
    if st.button("Train & Evaluate"):
        with st.spinner("Training models..."):
            results, fitted, X_train, X_test, y_train, y_test = train_and_evaluate(df)
        st.success("Training complete")
        metrics = metrics_df(results)
        st.dataframe(metrics.style.format("{:.4f}"))
        fig = go.Figure()
        for name, r in results.items():
            fpr, tpr = r["ROC"]
            auc = r["AUC"]
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={auc:.3f})"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random Chance'))
        fig.update_layout(title="ROC Curves (Test set)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Confusion matrices (train & test)")
        cols = st.columns(3)
        for i,(name,r) in enumerate(results.items()):
            cm_test = r["Confusion Test"]
            cm_train = r["Confusion Train"]
            with cols[i]:
                st.markdown(f"**{name} (Test)**")
                st.write(cm_test)
                st.markdown(f"**{name} (Train)**")
                st.write(cm_train)
        st.subheader("Feature importances")
        for name,(model,feat_names) in fitted.items():
            if hasattr(model, "feature_importances_"):
                fi = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False).head(20)
                figfi = px.bar(fi, x=fi.values, y=fi.index, orientation='h', title=f"{name} - Feature importances (top 20)")
                st.plotly_chart(figfi, use_container_width=True)

with tab[2]:
    st.header("Upload new data to predict Personal_Loan and download labeled file")
    st.markdown("Upload a CSV with the same feature columns (ID optional). Choose model and click Predict.")
    uploaded = st.file_uploader("Upload CSV file for prediction", type=["csv"])
    model_choice = st.selectbox("Choose model for prediction", ["Random Forest","Gradient Boosting","Decision Tree"])
    if uploaded is not None:
        try:
            to_pred = pd.read_csv(uploaded)
            st.write("Preview of uploaded data:")
            st.dataframe(to_pred.head())
            if st.button("Predict and provide download"):
                dfp = preprocess(df)
                drop_cols = [c for c in dfp.columns if c.lower().startswith("id") or "zip" in c.lower()]
                X_full = dfp.drop(columns=drop_cols+["Personal_Loan"], errors='ignore')
                y_full = dfp["Personal_Loan"].astype(int)
                models = {
                    "Decision Tree": DecisionTreeClassifier(random_state=42),
                    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
                    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
                }
                m = models[model_choice]
                m.fit(X_full, y_full)
                X_upload = to_pred.copy()
                X_upload = X_upload.drop(columns=[c for c in X_upload.columns if c.lower().startswith("id") or "zip" in c.lower()], errors='ignore')
                for c in X_full.columns:
                    if c not in X_upload.columns:
                        X_upload[c]=0
                X_upload = X_upload[X_full.columns]
                preds = m.predict(X_upload)
                out = to_pred.copy()
                out["Predicted_Personal_Loan"] = preds.astype(int)
                csv = out.to_csv(index=False).encode('utf-8')
                st.success("Prediction done — download below")
                st.download_button("Download labeled CSV", data=csv, file_name="predicted_personal_loan.csv", mime="text/csv")
        except Exception as e:
            st.error("Failed to process uploaded file: " + str(e))

with tab[3]:
    st.header("Dataset Preview & Download")
    st.dataframe(df.head(200))
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("Download current dataset CSV", data=buf.getvalue().encode('utf-8'), file_name="UniversalBank_used.csv", mime="text/csv")

st.sidebar.markdown("---")
st.sidebar.markdown("Built for: Marketing Head — actionable charts + model training + predictions")
st.sidebar.markdown("Notes: Models train on the fly (no persisted pickles).")
