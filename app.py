from flask import Flask, render_template, request
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)

# Hugging Face Inference Client
client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"]
)


def translate_to_french(english_text):
    result = client.translation(
        english_text,
        model="Helsinki-NLP/opus-mt-en-fr"
    )

    return result.translation_text


@app.route("/", methods=["GET", "POST"])
def home():
    translation = ""

    if request.method == "POST":
        english_text = request.form["english"]

        if english_text.strip():
            translation = translate_to_french(english_text)

    return render_template(
        "index.html",
        translation=translation
    )


if __name__ == "__main__":
    app.run(debug=True)