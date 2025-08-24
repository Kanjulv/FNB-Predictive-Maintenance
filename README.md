#  Predictive Operations System for F&B Manufacturing

This project is a real-time, interactive dashboard designed to detect process anomalies in Food & Beverage manufacturing. It was developed as a submission for the [Name of Hackathon]. The system provides a unified "Process Health Score," forecasts the business impact of anomalies, and guides operator actions through an interactive digital playbook, complete with a regulatory-ready audit log.

**Live Demo URL:** `https://f891ee8325b4.ngrok-free.app/`
*(Note: This is a temporary link and is only active while the Google Colab notebook is running the final launch cell.)*

---

## ✨ Key Features

* **Unified Health Monitoring:** A real-time "Process Health Score" gauge provides an at-a-glance understanding of the production line's stability.
* **Quality Impact Forecast:** The system forecasts the direct business impact of anomalies, such as projected yield loss or a Cpk drift.
* **Interactive Digital Playbooks:** Upon an alert, operators are guided by an interactive checklist of standard operating procedures (SOPs).
* **Full Audit Trail:** All acknowledged events are logged in a persistent, downloadable report, providing traceability for HACCP and ISO 22000 compliance.
* **Multi-SKU Capability:** The system can dynamically load different process profiles ("Golden Batches") for various products (e.g., Chocolate Chip vs. Oatmeal Raisin cookies).

---

## 🛠️ Tech Stack

* **Language:** Python
* **Data Science:** Pandas, Scikit-learn, XGBoost, SHAP
* **Dashboard:** Streamlit, Plotly
* **Environment:** Google Colab, `pyngrok` for temporary deployment

---

## 🚀 How to Run

There are two ways to run this project:

### 1. Self-Contained Demo (Recommended)
This is the quickest way to see the project in action.

1.  Upload the `.ipynb` file (`HoneywellDraft1.ipynb`) to Google Colab.
2.  Run all the cells in order from top to bottom.
3.  The final cell will output a public `ngrok` URL. Click this link to open the live dashboard.

### 2. Local Environment Setup
To run the project on your local machine:

1.  Clone the repository:
    ```bash
    git clone https://github.com/Kanjulv/FNB-Predictive-Maintenance
    ```
2.  Navigate to the project directory:
    ```bash
    cd FNB-Predictive-Maintenance
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
