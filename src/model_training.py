from transformers import TrainingArguments, Trainer
import torch
from data import *

def collate_fn(examples):
    return {
        "pixel_values":   torch.stack([ex["pixel_values"]   for ex in examples]),
        "input_ids":      torch.stack([ex["input_ids"]      for ex in examples]),
        "attention_mask": torch.stack([ex["attention_mask"] for ex in examples]),
        "return_loss":    True,
    } 

def Training(model, data):
    training_args = TrainingArguments(output_dir = "../content/drive/MyDrive/siglip_model",
                                    num_train_epochs = 5,
                                    per_device_train_batch_size = 64,
                                    learning_rate = 1e-5,
                                    warmup_steps = 50,
                                    weight_decay = 0.01,
                                    lr_scheduler_type = "cosine",
                                    fp16 = torch.cuda.is_available(),
                                    save_strategy = "epoch",
                                    logging_steps = 20,
                                    eval_strategy = "epoch"
                                    )



    trainer = Trainer(model,
                    training_args,
                    train_dataset = data["train"],
                    eval_dataset = data["test"],
                    data_collator = collate_fn
                    )

    trainer.train()

# save the model into the local disk directory will be same as output dir as given in training arguments.
    trainer.save_model()