from flask import Flask, request, jsonify
app = Flask(__name__)
import os
import recommend_AI

@app.route('/getmsg/', methods=['GET'])
def respond():
    # Retrieve the name from the url parameter /getmsg/?name=
    name = request.args.get("name", None)

    # For debugging
    print(f"Received: {name}")

    response = {}

    # Check if the user sent a name at all
    if not name:
        response["ERROR"] = "No shoes's name found. Please send a name."
    # Check if the user entered a number
    elif str(name).isdigit():
        response["ERROR"] = "The shoes's name can't be numeric. Please send a string."
    else:
        # print(recommend_AI.recommend(name))
        response["recommend_products"] = recommend_AI.recommend(name)
        # Return the response in json format
        return jsonify(response)

@app.route('/')
def index():
    # A welcome message to test our server
    return "<h1>Welcome to our recommend-AI-api!</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)