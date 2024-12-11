from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

# Charger le modèle ANN et CNN
modelANN = load_model("C:/Users/etudiant/Project_ML/model.h5")

# Initialiser Flask
app = Flask(__name__)

# Route pour effectuer des prédictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Requête reçue")
        data = request.json
        print("Données reçues :", data)

        # Prétraitement des données
        input_data = np.array(data["input"]).reshape(1, 32, 32, 3)
        print("Données traitées :", input_data.shape)

        # Faire une prédiction
        predictions = modelANN.predict(input_data)
        print("Prédiction effectuée")

        predicted_class = int(np.argmax(predictions))
        print("Classe prédite :", predicted_class)

        # Retourner la réponse
        return jsonify({
            "predictions": predictions.tolist(),
            "predicted_class": predicted_class
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Lancer le serveur Flask
if __name__ == "__main__":
    app.run(debug=True, port=5001)
