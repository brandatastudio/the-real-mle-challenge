def map_price_category_code_to_text(code):
    if code == 0:
        resulting_text = "Low"
    elif code == 1:
        resulting_text = "Mid"
    elif code == 2:
        resulting_text = "High"
    elif code == 3:
        resulting_text = "Lux"

    return resulting_text
