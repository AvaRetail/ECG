from datasets import load_dataset
from transformers import AutoImageProcessor
from torchvision.transforms import Resize, Compose, Normalize, ToTensor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, ViTForImageClassification
import evaluate
import numpy as np
# from sklearn.model_selection import train_test_split

from transformers import DefaultDataCollator


def main():

    # with split returns Dataset
    # without split return DatasetDict
    data = load_dataset(data_root_fldr,split="train") 
    data = data.train_test_split(test_size=0.02)

    labels = data["train"].features["label"].names
    label2id, id2label = {}, {}

    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[f"{i}"] = label

    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

    def transform(examples):

        h, w = 224, 224
        _transform = Compose(
            [
                Resize((h, w)),
                ToTensor(),
                normalize
            ]
        )

        examples["pixel_values"] = [_transform(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]

        return examples

    data = data.with_transform(transform)

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    data_collator = DefaultDataCollator()

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=r"C:\Users\ATI-G2\Documents\python\ECG\weights\viT-b-16",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        num_train_epochs=100,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    data_root_fldr = r"C:\Users\ATI-G2\Documents\python\ECG\data\vit-data"

    checkpoint = "google/vit-base-patch16-224-in21k"
    # checkpoint = r"C:\Users\ATI-G2\Documents\python\ECG\my_awesome_food_model\checkpoint-4976"

    main()