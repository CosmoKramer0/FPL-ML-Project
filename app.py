from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from best_squad_picks import fetch_data, train_model, select_best_11_with_captain, determine_formation

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Fetch player data and precompute predictions when the app starts
player_data = fetch_data()
player_data_with_predictions = train_model(player_data)

@app.route('/api/fetch_data', methods=['GET'])
def get_player_data():
    try:
        data_dict = player_data_with_predictions.to_dict(orient="records")
        return jsonify(data_dict)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/select_best_11', methods=['POST'])
def select_best_11_route():
    data = request.get_json()
    try:
        # Filter the precomputed predictions for the selected players
        selected_players = pd.DataFrame(data)
        best_11, captain = select_best_11_with_captain(selected_players)
        best_11_dict = best_11.to_dict(orient="records")
        captain_dict = captain.to_dict() if captain is not None else None
        return jsonify({"best_11": best_11_dict, "captain": captain_dict})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/determine_formation', methods=['POST'])
def determine_formation_route():
    data = request.get_json()
    try:
        df = pd.DataFrame(data)
        formation = determine_formation(df)
        return jsonify({"formation": formation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)