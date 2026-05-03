def calculate_severity(tumor_type, confidence):
    if tumor_type == "notumor":
        return "None", "Safe"

    if confidence > 85:
        return "High", "Critical"
    elif confidence > 65:
        return "Medium", "Moderate"
    else:
        return "Low", "Early Stage"
