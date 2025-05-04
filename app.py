from flask import Flask, request, jsonify, render_template
from rag_recommender.modules.rag_pipeline import search_assessments

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# @app.route("/search", methods=["POST"])
# def search():
#     data = request.get_json()
#     query = data.get("query", "")
#     try:
#         results = search_assessments(query)
#         return jsonify({"results": results})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
# @app.route('/search', methods=['POST'])
# def search():
#     data = request.get_json()
#     query = data.get("query", "")
#     results = search_assessments(query)

#     return jsonify({"results": results}) 


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query must not be empty"}), 400

    try:
        results = search_assessments(query)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)
