import os
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from datetime import datetime

def generate_dashboard_pdf(file_path, tumor_labels, tumor_values, trend_dates, trend_counts, history):

    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>AI Hospital Analytics Report</b>", styles["Title"]))
    elements.append(Paragraph(f"Generated on: {datetime.now()}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    # Generate Tumor Chart Image
    chart_path = file_path.replace(".pdf", "_chart.png")

    plt.figure()
    plt.pie(tumor_values, labels=tumor_labels, autopct='%1.1f%%')
    plt.title("Tumor Distribution")
    plt.savefig(chart_path)
    plt.close()

    elements.append(Image(chart_path, width=4*inch, height=4*inch))
    elements.append(Spacer(1, 0.3 * inch))

    # Add History Table
    table_data = [["Tumor", "Confidence", "Date"]]

    for row in history:
        table_data.append([row["tumor"], row["confidence"], row["date"]])

    table = Table(table_data)
    elements.append(table)

    doc.build(elements)

    # Remove temp chart image
    if os.path.exists(chart_path):
        os.remove(chart_path)
