from flask import Flask, render_template, request
from predict import predict_email

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def main_page():
    output = None

    if request.method == "POST":
        # Get text from the textarea in index.html
        user_text = request.form["email_content"]

        # Get results from our prediction script
        res_label, res_score, res_risk, res_words = predict_email(user_text)

        output = {
            "label": res_label,
            "score": res_score,
            "risk": res_risk,
            "bad_words": res_words
        }

    return render_template("index.html", result=output)

# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    app.run()