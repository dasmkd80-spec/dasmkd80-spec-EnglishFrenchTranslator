from flask import Flask, render_template, request
from transformers import MarianTokenizer, MarianMTModel

app = Flask(__name__)

model_name = "Helsinki-NLP/opus-mt-en-fr"

# Lazy loading
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

def translate_to_french(english_text):
    load_model()   # Model first request appudu matrame load avutundi

    encoded = tokenizer(
        english_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    translated_tokens = model.generate(**encoded)
    french_text = tokenizer.decode(
        translated_tokens[0],
        skip_special_tokens=True
    )

    return french_text


@app.route("/", methods=["GET", "POST"])
def home():
    translation = ""

    if request.method == "POST":
        english_text = request.form["english"]
        translation = translate_to_french(english_text)

    return render_template("index.html", translation=translation)


if __name__ == "__main__":
    app.run(debug=True)