"""
Script to generate a formatted Word Document (.docx) for FinGuard AI
Summarizing the entire application, all pages, architecture, and viva answers.
"""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from pathlib import Path

def set_cell_background(cell, fill_hex):
    """Sets background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets internal padding for a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def create_callout(doc, text, title="KEY TAKEAWAY FOR VIVA"):
    """Creates a highlighted callout box."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run_t = p.add_run(f"📌 {title}: ")
    run_t.bold = True
    run_t.font.name = "Calibri"
    run_t.font.size = Pt(10.5)
    run_t.font.color.rgb = RGBColor(30, 58, 138)
    
    run_c = p.add_run(text)
    run_c.font.name = "Calibri"
    run_c.font.size = Pt(10.5)
    run_c.font.color.rgb = RGBColor(15, 23, 42)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def build_document(output_path: Path):
    doc = docx.Document()
    
    # Page setup: Standard Margins
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Styles
    # Title
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    title_run = title_p.add_run("🛡️ FinGuard AI — Quick Viva & Project Guide")
    title_run.bold = True
    title_run.font.name = "Arial"
    title_run.font.size = Pt(22)
    title_run.font.color.rgb = RGBColor(30, 58, 138) # Deep Navy

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(16)
    sub_run = sub_p.add_run("Intelligent Credit Card Fraud Detection & Personal Spending Analytics System\nThird-Year Computer Science Diploma Semester Mini-Project Guide")
    sub_run.font.name = "Calibri"
    sub_run.font.size = Pt(11)
    sub_run.font.color.rgb = RGBColor(100, 116, 139)

    # SECTION 1
    h1 = doc.add_heading("1. Project Summary in Simple Words", level=1)
    h1.style.font.color.rgb = RGBColor(30, 58, 138)
    
    p = doc.add_paragraph(
        "FinGuard AI is a machine learning web application built with Python and Streamlit. "
        "Its main job is to solve a major problem in modern fintech: detecting whether a credit card transaction "
        "is Legitimate (Normal) or Fraudulent (Stolen/Unauthorized) before the bank processes it."
    )
    p.style.font.size = Pt(11)

    p2 = doc.add_paragraph(
        "The application also includes a supplementary Personal Spending Analytics module, where users can track "
        "their personal expenses (in Indian Rupees ₹), view category budgets, analyze monthly trends, and catch unusual expenses."
    )
    p2.style.font.size = Pt(11)

    create_callout(
        doc,
        "If the examiner asks 'What is your project in one sentence?': Say: 'FinGuard AI is an interactive fintech machine learning application that detects credit card fraud using genuine supervised classification algorithms while handling severe class imbalance, and provides personal spending intelligence.'",
        "ONE-SENTENCE SUMMARY"
    )

    # SECTION 2
    h2 = doc.add_heading("2. The Dataset (What You Must Know for Viva)", level=1)
    h2.style.font.color.rgb = RGBColor(30, 58, 138)

    doc.add_paragraph("Examiners always ask about the dataset. Here are the exact facts to state:").style.font.size = Pt(11)

    bullet_pts = [
        ("Source:", "Official Kaggle MLG-ULB Credit Card Fraud Detection benchmark dataset (European cardholders)."),
        ("Total Transactions:", "284,807 transactions recorded over a 48-hour period."),
        ("Fraudulent Transactions:", "Only 492 transactions are fraud (~0.172% of total)."),
        ("Class Imbalance Ratio:", "Approximately 578 legitimate transactions for every 1 fraudulent transaction!"),
        ("Features (30 inputs):", "Time (seconds elapsed), Amount (transaction amount), and V1 through V28."),
        ("Target Variable:", "Class (0 = Legitimate transaction, 1 = Fraudulent transaction)."),
        ("What are V1 to V28?:", "They are numerical variables resulting from Principal Component Analysis (PCA). The bank anonymized them to protect customer privacy and comply with GDPR/PCI-DSS rules. They are NOT fake data; they are mathematical combinations of original banking features.")
    ]

    for label, desc in bullet_pts:
        bp = doc.add_paragraph(style='List Bullet')
        r1 = bp.add_run(f"{label} ")
        r1.bold = True
        r1.font.size = Pt(10.5)
        r2 = bp.add_run(desc)
        r2.font.size = Pt(10.5)

    create_callout(
        doc,
        "Never invent real-world meanings for V1-V28 (e.g. do not say V1 is age or V2 is location). Tell the examiner they are anonymized PCA components to protect customer confidentiality.",
        "CRITICAL VIVA TIP"
    )

    # SECTION 3
    h3 = doc.add_heading("3. Machine Learning Pipeline & Leakage Prevention", level=1)
    h3.style.font.color.rgb = RGBColor(30, 58, 138)

    doc.add_paragraph(
        "Our system strictly follows genuine machine learning methodology with ZERO data leakage:"
    ).style.font.size = Pt(11)

    steps = [
        ("1. Stratified 80/20 Train/Test Split:", "We split 284,807 rows into 80% Training (227,845 rows) and 20% Testing (56,962 rows). We use Stratified splitting so both training and testing sets have the exact same 0.17% fraud percentage."),
        ("2. Leakage-Free Preprocessing:", "Features Time and Amount have wild ranges, so we scale them using RobustScaler (which uses Median and Interquartile Range, making it immune to extreme spending outliers). The scaler is FITTED ONLY ON TRAINING DATA. The test set is transformed using those fitted values. This ensures zero data leakage."),
        ("3. Model Training (3 Supervised Classifiers):", "\n  • Logistic Regression: Baseline linear model with balanced class weighting.\n  • Decision Tree: Non-linear interpretable model (max_depth=6) with if-else decision rules.\n  • Random Forest: Powerful ensemble of 100 decorrelated decision trees using bagging."),
        ("4. Imbalance Handling:", "We used class_weight='balanced' (giving high penalties for missing fraud) and added an optional SMOTE oversampling experiment strictly on the training set.")
    ]

    for label, desc in steps:
        sp = doc.add_paragraph()
        r1 = sp.add_run(label + "\n")
        r1.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = RGBColor(30, 58, 138)
        r2 = sp.add_run(desc)
        r2.font.size = Pt(10.5)

    # SECTION 4
    h4 = doc.add_heading("4. Actual Model Performance & Why Random Forest Won", level=1)
    h4.style.font.color.rgb = RGBColor(30, 58, 138)

    doc.add_paragraph("These are the actual experimental results on the untouched 56,962 test records:").style.font.size = Pt(11)

    # Table
    table = doc.add_table(rows=4, cols=6)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    headers = ["Model", "Accuracy", "Recall", "Precision", "PR-AUC", "F1-Score"]
    col_widths = [Inches(1.8), Inches(0.9), Inches(0.9), Inches(0.9), Inches(0.9), Inches(0.9)]

    # Header Row
    for i, h_text in enumerate(headers):
        cell = table.cell(0, i)
        cell.width = col_widths[i]
        set_cell_background(cell, "1E3A8A")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h_text)
        run.bold = True
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(255, 255, 255)

    data_rows = [
        ("Logistic Regression", "97.55%", "91.84%", "6.09%", "0.7175", "0.1142"),
        ("Decision Tree", "97.63%", "86.73%", "5.97%", "0.4643", "0.1118"),
        ("Random Forest 🏆", "99.92%", "84.69%", "73.45%", "0.8358", "0.7867")
    ]

    for row_idx, data in enumerate(data_rows, start=1):
        bg_color = "E0F2FE" if "🏆" in data[0] else ("FFFFFF" if row_idx % 2 != 0 else "F8FAFC")
        for col_idx, text in enumerate(data):
            cell = table.cell(row_idx, col_idx)
            cell.width = col_widths[col_idx]
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(text)
            run.font.size = Pt(9.5)
            if "🏆" in data[0]:
                run.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    create_callout(
        doc,
        "Why is Random Forest the best model? Say: 'Random Forest achieved the highest PR-AUC (0.8358) and F1-score (0.7867). While Logistic Regression caught slightly more fraud (Recall 91.8%), it generated 1,388 false alerts on innocent cardholders (Precision was only 6%). Random Forest caught 84.7% of fraud while maintaining 73.5% precision (only 30 false alerts), making it the most practical production model.'",
        "WHY RANDOM FOREST IS PREFERRED"
    )

    # SECTION 5
    h5 = doc.add_heading("5. Page-by-Page Application Walkthrough", level=1)
    h5.style.font.color.rgb = RGBColor(30, 58, 138)

    doc.add_paragraph("Here is exactly what each page in your Streamlit application does:").style.font.size = Pt(11)

    pages_info = [
        ("Page 1: Home / Executive Dashboard (app.py)", 
         "• Shows system telemetry cards (284,807 transactions analyzed, 492 frauds, 0.17% fraud rate).\n"
         "• Live indicator showing dataset source (creditcard.csv loaded) and trained models status.\n"
         "• Interactive Plotly Boxplot showing the distribution of transaction amounts for legit vs fraud.\n"
         "• Upload box to load new CSVs or generate synthetic demo data with one click."),
        
        ("Page 2: Fraud Detection (pages/1_Fraud_Detection.py)", 
         "• Single Transaction Inspector: Pick an existing transaction or test known fraud/legit examples.\n"
         "• Classification Threshold Slider (0.05 to 0.95): Allows user to change the decision boundary. If fraud probability >= threshold, the transaction is flagged.\n"
         "• Displays risk banner ('Potentially Suspicious' in red or 'Likely Legitimate' in green).\n"
         "• Batch CSV Inference: Upload any CSV batch of transactions, score thousands of rows in bulk, and download the annotated CSV with predicted fraud probabilities."),
        
        ("Page 3: Transaction Explorer (pages/2_Transaction_Explorer.py)", 
         "• Searchable and filterable table displaying Time, Amount, Ground Truth Class, Predicted Class, and Fraud Probability.\n"
         "• Filter by Amount range, ground truth label, or predicted fraud status.\n"
         "• Pagination (25, 50, 100, 200 rows per page) and CSV export for filtered records."),
        
        ("Page 4: Model Lab & Model Comparison (pages/3_Model_Lab.py)", 
         "• Side-by-side benchmark table comparing Logistic Regression, Decision Tree, and Random Forest.\n"
         "• Grouped Bar Chart comparing Accuracy, Precision, Recall, F1, and PR-AUC.\n"
         "• Interactive Confusion Matrix Heatmap (True Negatives, False Positives, False Negatives, True Positives).\n"
         "• Precision-Recall Curves (PR-AUC) and ROC Curves.\n"
         "• Retraining Suite: Allows retraining all models from the browser with optional SMOTE."),
        
        ("Page 5: Personal Spending Analytics (pages/4_Spending_Analytics.py)", 
         "• Supplementary personal finance module (in Indian Rupees ₹ by default).\n"
         "• Interactive Category Donut Chart showing where money was spent (Groceries, Dining, Shopping, etc.).\n"
         "• Monthly Spending Trajectory line chart.\n"
         "• Top largest expenses table and date range filtering.\n"
         "• Experimental Anomaly Detector (Z-Score > 2.5) that automatically flags unusually large expenses."),
        
        ("Page 6: Model Insights & Explainability (pages/5_Model_Insights.py)", 
         "• Gini Feature Importance chart showing which PCA variables had the highest split power in the Random Forest.\n"
         "• Explains why features V14, V12, V10, V17 are consistently the most influential.\n"
         "• Educational section on PCA and correlation vs causation."),
        
        ("Page 7: About Project & Documentation (pages/6_About_Project.py)", 
         "• Complete academic project report specification suitable for semester presentations.\n"
         "• Architecture diagram, methodology, problem definition, and student contribution log.")
    ]

    for p_title, p_desc in pages_info:
        doc.add_heading(p_title, level=2).style.font.color.rgb = RGBColor(15, 23, 42)
        p = doc.add_paragraph(p_desc)
        p.style.font.size = Pt(10.5)

    # SECTION 6
    h6 = doc.add_heading("6. Top 15 Viva Questions & Ready Answers", level=1)
    h6.style.font.color.rgb = RGBColor(30, 58, 138)

    viva_qa = [
        ("Q1: What is the Accuracy Paradox in your project?",
         "Because 99.8% of transactions are legitimate, a useless model that blindly predicts every transaction is legitimate would achieve 99.8% accuracy but catch ZERO fraud! Therefore, accuracy is deceptive, and we must use Precision, Recall, and PR-AUC."),
        
        ("Q2: What is the difference between Recall and Precision?",
         "Recall is the percentage of actual fraud cases caught by the model (high recall prevents financial loss). Precision is the percentage of flagged alerts that were actually fraud (high precision prevents embarrassing false alarms on genuine customers)."),
        
        ("Q3: What is PR-AUC and why is it better than ROC-AUC here?",
         "In extreme class imbalance, ROC-AUC can look artificially high (0.98+) because True Negatives dominate. PR-AUC (Precision-Recall Area Under Curve) evaluates precision directly across all thresholds and is the standard metric for rare-event detection."),
        
        ("Q4: How did you prevent Data Leakage?",
         "We performed an 80/20 Stratified Split first. Then we fitted the RobustScaler exclusively on the training partition. The test partition was scaled using the fitted training parameters without re-fitting."),
        
        ("Q5: What is RobustScaler and why did you use it instead of StandardScaler?",
         "Transaction amounts have extreme outliers (e.g. $1 vs $15,000). RobustScaler scales features using the Median and Interquartile Range (IQR), making it resistant to extreme outliers."),
        
        ("Q6: What is a Confusion Matrix?",
         "A 2x2 table showing True Negatives (correct legit), False Positives (legit flagged as fraud), False Negatives (fraud missed), and True Positives (fraud caught)."),
        
        ("Q7: What is SMOTE?",
         "Synthetic Minority Over-sampling Technique. It creates synthetic minority records by interpolating between nearest neighbors in feature space. In our pipeline, it is applied strictly to training data."),
        
        ("Q8: Why is Random Forest an ensemble method?",
         "Because it combines predictions from 100 individual decision trees trained on random subsets of data and features (bagging), reducing variance and avoiding overfitting."),
        
        ("Q9: What is the role of the Decision Threshold slider in Page 2?",
         "The model outputs a probability (0.0 to 1.0). The threshold determines when to classify as fraud. Default is 0.50. Lowering it (e.g. 0.20) increases recall (catches more fraud) but increases false positives. Raising it reduces false alarms."),
        
        ("Q10: What is Joblib used for?",
         "Joblib serializes Python objects and trained Scikit-learn pipelines into .joblib binary files, so models do not have to be retrained on every user click."),
        
        ("Q11: Why are features V1 to V28 anonymized?",
         "They are Principal Components (PCA) created by the bank to protect cardholder privacy under GDPR and PCI-DSS rules."),
        
        ("Q12: Which features are most important in detecting fraud?",
         "Based on Gini importance in our Random Forest, PCA features V14, V12, V10, and V17 are the most influential in separating fraud from normal transactions."),
        
        ("Q13: What does the Spending Analytics module do?",
         "It is an independent personal expense tracker that takes Date, Amount, and Category, calculating monthly budgets, category distributions, and statistical spending anomalies using Z-scores."),
        
        ("Q14: How many unit tests do you have and what do they verify?",
         "We have 17 automated Pytest unit tests covering schema validation, leakage prevention, probability outputs, threshold classifications, and spending calculations."),
        
        ("Q15: What are future improvements for this system?",
         "Adding Explainable AI with SHAP, deploying inference as a REST API using FastAPI and Docker, and streaming live transactions through Apache Kafka.")
    ]

    for q, a in viva_qa:
        qp = doc.add_paragraph()
        q_run = qp.add_run(q + "\n")
        q_run.bold = True
        q_run.font.size = Pt(10.5)
        q_run.font.color.rgb = RGBColor(30, 58, 138)
        a_run = qp.add_run(a)
        a_run.font.size = Pt(10)

    # SECTION 7
    h7 = doc.add_heading("7. How to Launch & Demonstrate the App in Viva", level=1)
    h7.style.font.color.rgb = RGBColor(30, 58, 138)

    doc.add_paragraph("Follow these exact steps when demonstrating to teachers or external examiners:").style.font.size = Pt(11)

    demo_steps = [
        "1. Open Terminal (PowerShell) in project folder: cd 'c:/DIPLOMA PROJECTS/finguard-ai'",
        "2. Launch Streamlit: streamlit run app.py",
        "3. Open Browser at: http://localhost:8501",
        "4. Show the Executive Dashboard: Highlight the 284,807 transactions and the real Kaggle dataset badge.",
        "5. Go to Page 1 (Fraud Detection): Show the Single Transaction Inspector, click 'Load an Example Known Fraudulent Transaction', click 'Run Fraud Risk Evaluation', and show the RED alert banner. Then slide the threshold to show how it changes.",
        "6. Go to Page 3 (Model Lab): Show the benchmark comparison table and explain why Random Forest (PR-AUC 0.8358) beats Logistic Regression and Decision Tree.",
        "7. Go to Page 5 (Model Insights): Show the Gini Importance bar chart and explain that V14, V12, and V10 are the primary mathematical indicators.",
        "8. Show tests in terminal: Run 'pytest -v' to demonstrate all 17 passing automated unit tests."
    ]

    for ds in demo_steps:
        dsp = doc.add_paragraph(style='List Bullet')
        r = dsp.add_run(ds)
        r.font.size = Pt(10.5)

    doc.save(str(output_path))
    print(f"Document successfully created at: {output_path}")

if __name__ == "__main__":
    out = Path("c:/DIPLOMA PROJECTS/finguard-ai/docs/FinGuard_AI_Project_Summary_and_Viva_Guide.docx")
    build_document(out)
