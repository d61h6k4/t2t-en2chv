from transformers import AutoModelForSeq2SeqLM
from transformers.tools.translation import AutoTokenizer


def main():
    checkpoint = "models/umt5_ru2chv/checkpoint-120000"
    # checkpoint = "models/umt5_en2chv/checkpoint-20000"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    text = "Ну я и заскучал, даже вша на мне появилась от тоски."

    inputs = tokenizer(text, return_tensors="pt").input_ids
    outputs = model.generate(
        inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
