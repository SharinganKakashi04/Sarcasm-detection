# UI/app.py (Example using Flask)
from flask import Flask, request, jsonify, render_template
import os
import torch
# Assuming your InteractiveLearningSystem is importable from its scripts location
from scripts.interactive_learning_system import InteractiveLearningSystem 

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# --- INITIALIZE MODEL GLOBALLY ---
try:
    # Adjust the path to load the model file from the project root
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'roberta_best.pt')
    system = InteractiveLearningSystem(model_path=MODEL_PATH)
    print("Backend model loaded and ready for predictions.")
except FileNotFoundError:
    system = None
    print("WARNING: Model file not found. Run training script first.")

# --- API ENDPOINT FOR PREDICTION ---
@app.route('/predict', methods=['POST'])
def predict():
    if not system:
        return jsonify({'error': 'Model not initialized'}), 503
        
    data = request.get_json()
    user_text = data.get('text', '')
    
    if not user_text:
        return jsonify({'prediction': 'Error', 'confidence': 0}), 400

    # Run prediction using your loaded system
    label, confidence_scores = system.predict(user_text)
    
    return jsonify({
        'text': user_text,
        'prediction': label.upper(),
        'confidence': confidence_scores
    })

# --- ROUTE TO SERVE THE MAIN PAGE ---
@app.route('/')
def index():
    return render_template('main.html')

if __name__ == '__main__':
    # You will need to install Flask: pip install Flask
    app.run(debug=True)