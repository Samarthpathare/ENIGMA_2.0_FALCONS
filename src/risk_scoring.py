def calculate_risk(model, sample):
    probability = model.predict_proba([sample])[0][1]
    return probability * 100