import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    logging,
)

logging.set_verbosity_warning()


def main():
    checkpoint = "google/umt5-small"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    metric = evaluate.load("sacrebleu")

    dataset = load_dataset("alexantonov/chuvash_english_parallel")[
        "train"
    ].train_test_split(test_size=0.2)
    source_lang = "en"
    target_lang = "chv"

    def preprocess_function(examples):
        inputs = examples[source_lang]
        targets = examples[target_lang]

        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=128, truncation=True
        )
        return model_inputs

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    training_args = Seq2SeqTrainingArguments(
        output_dir="models/umt5_en2chv",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=2,
        predict_with_generate=True,
        skip_memory_metrics=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
