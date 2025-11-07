from flask import Flask, request, jsonify
from predict import Recommender

app = Flask(__name__)
recommender = Recommender()

@app.route('/recommend', methods=['GET'])
def recommend():
    """Endpoint to get movie recommendations for a user."""
    user_id_str = request.args.get('user_id')
    
    if not user_id_str:
        return jsonify({"error": "user_id parameter is required"}), 400
        
    try:
        user_id = int(user_id_str)
    except ValueError:
        return jsonify({"error": "user_id must be an integer"}), 400

    recommendations, rec_type = recommender.get_recommendations(user_id)
    
    response = {
        "user_id": user_id,
        "recommendations": recommendations,
        "type": rec_type
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
