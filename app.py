from flask import Flask, render_template, request
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)

# Hugging Face client
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
    english_text = ""

    if request.method == "POST":

        english_text = request.form.get("english", "").strip()

        if english_text:

            # Maximum 1000 characters
            english_text = english_text[:1000]

            try:
                translation = translate_to_french(english_text)

            except Exception as e:
                print("Translation error:", e)
                translation = "Translation service is temporarily unavailable."

    return render_template(
        "index.html",
        translation=translation,
        english_text=english_text
    )


if __name__ == "__main__":
    app.run(debug=True)