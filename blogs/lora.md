## Introduction

This is a straightforward guide to help you fine-tune [ModernBERT](https://arxiv.org/abs/2412.13663) on any classification task. Any number of labels. Any type of input (code, text, etc.). We'll walk through the full code and explain everything.

Whether you're using a single GPU or multiple GPUs, this guide has you covered.

[ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) is a small model, and when combined with **LoRA**, it can be fine-tuned on a free Colab T4 (15GB VRAM). We ran the process on 4 Tesla V100s, but we believe it can work on a single 8GB GPU — just expect to use a smaller batch size (around 8 or 4), which trades memory usage for time.

We'll be using low-level PyTorch and `accelerate` to handle multi-GPU training.

First, let's import some packages:

We'll use `pandas` and `torch` for data preparation and model training, `sklearn` for evaluation metrics, `peft` for LoRA, `transformers` for loading the model and tokenizer, and `accelerate` for multi-GPU training. And of course, `os` to manage directories and store logs.

Don't forget to install these packages if they're not already in your environment.

```python
import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score
from peft import LoraConfig, TaskType, get_peft_model
from transformers import ModernBertForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from accelerate import Accelerator
```

Now let's define a few variables. I like to keep them here at the top so it's easy to tweak later.

In this tutorial, our dataset is already uploaded to Hugging Face. If you're using a local file, that's fine too! Just expect a few tweaks in the dataset loading code.

Also, we won't be using the `dataset_id` directly. Instead, we use the Parquet path, which you can grab from the "Use this dataset" → "pandas" tab on Hugging Face. This allows us to load the dataset with pandas without converting from the Hugging Face `Dataset` object.

We'll also define a few hyperparameters here, but we'll come back to those later.

```python
MODEL_NAME = "answerdotai/ModernBERT-base"

DATASET_NAME = "hf://datasets/lemon42-ai/minified-diverseful-multilabels/data/train-00000-of-00001.parquet"
MAX_LENGTH = 600
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 5e-4
LOGGING_STEPS = 100
```

## Dataset Definition

Next, we need to prepare the data. We'll create a class that inherits from PyTorch's `Dataset`. We named it `TextDataset`, but feel free to choose any name—just remember to update all references to it accordingly.

```python
class TextDataset(Dataset):
    def __init__(self, dataset_id, tokenizer, max_length):

        self.data = pd.read_parquet(dataset_id) #read the parquet (adapt to your situation)
        self.tokenizer = tokenizer 
        self.max_length = max_length 

        self.data = self.data[["func", "cwe"]] #in our dataset, we have more than two columns, so we truncate to keep inputs (func) & outputs (cwe) columns only.
        self.data.columns = ["Text", "Label"] #rename columns to "Text" & "Label"

        self.data = self.data[self.data["Label"].notnull()] #remove rows with null labels, if not done beforehand.
        unique_labels = sorted(self.data["Label"].unique()) #get the unique labels

        #create tha mappings from label to idx (we'll need them later)
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)} 
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
```

We also define the `__len__()` and `__getitem__()` methods under the same class for use during training:

```python
class TextDataset(Dataset):
    def __init__(self, dataset_id, tokenizer, max_length):
        ...

    def __len__(self):
        return len(self.data)

    #the __getitem__() method takes a row idx and returns a dictionary {"input_id", "labels"}
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["Text"]
        label = row["Label"]

        # Tokenize text & truncate to max_length (maxiumum number of tokens allowed per sequence)
        token_ids = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=self.max_length)

        label_idx = self.label2idx[label]
        return {
            "input_ids": token_ids,
            "labels": label_idx,
        }
```

Note that we do not manually create attention masks here, as we'll use a `DataCollator` to handle padding later. The attention mask tells the model which tokens to attend to (1 = real token, 0 = padding). Since the collator handles this automatically, we don't need to worry about it here.

## Main Training and Evaluation Loop

Now that the dataset class is ready, let's set up some hyperparameters

```python
#get the values of model_name, dataset_name & max_length that we fixed in the beginning of our code
model_name = MODEL_NAME
dataset_id = DATASET_NAME
max_length = MAX_LENGTH 

# Training hyperparameters
num_train_epochs = NUM_EPOCHS
learning_rate = LEARNING_RATE
weight_decay = 0.01
per_device_train_batch_size = BATCH_SIZE
per_device_eval_batch_size = BATCH_SIZE
logging_steps = LOGGING_STEPS
```

Let's discuss the key hyperparameters:

- `num_train_epochs`: the total number of passes over the training dataset. We set it to 20, but you can experiment with lower values.
- `learning_rate`: the step size for updating weights. Typically, values around `10**5` are used, but since we employ LoRA, we can afford larger values (~`10**4` or `10**3`).
- `weight_decay`: a regularization term to penalize large weights.
- `per_device_train_batch_size`: batch size per GPU during training. When using multiple GPUs, the effective batch size equals `per_device_train_batch_size * num_gpus`. For GPUs with limited VRAM, a smaller batch size is recommended.
- `per_device_eval_batch_size`: batch size per GPU during evaluation. This can be the same or larger than the training batch size since gradients are not computed, freeing up VRAM.
- `logging_steps`: the interval (in steps) between logging metrics such as loss. Avoid very small values to keep logs clean and manageable.

Since there are only general guidelines for setting these hyperparameters, we encourage you to experiment and find the values that best suit your situation.

Next, we'll use HuggingFace's `accelerate` to set up multi-GPU training. This won't cause any issues even if you have just a single GPU. In fact, using `accelerate` makes it easier to scale to multiple GPUs later and also simplifies enabling mixed precision with fp16.

```python
accelerator = Accelerator()
```

Now that we have our `accelerator` set up & ready, we can move on to load the tokenizer & prepare the data with a `DataCollator`

```python
# ----- Load Tokenizer and Prepare Dataset -----
accelerator.print("Loading tokenizer...") #we use accelerator.print() instead of print() when using multiple gpus
tokenizer = AutoTokenizer.from_pretrained(model_name) #load tokenizer

data_collator = DataCollatorWithPadding(tokenizer=tokenizer) #prepare a data_collator

accelerator.print("Instantiating the full dataset...")
full_dataset = TextDataset(dataset_id, tokenizer, max_length)
num_labels = len(full_dataset.label2idx)
accelerator.print(f"Number of labels: {num_labels}")
```

The `data_collator` is very useful because it performs **dynamic padding**: it pads each sequence in the batch to match the length of the longest sequence **within that batch** (instead of a fixed maximum length). This avoids unnecessary padding and saves memory. We pass the tokenizer as an argument so it uses the tokenizer's pad token. It also takes care of converting inputs to tensors for us.

Next, let's split our dataset into training and evaluation sets—the usual way.

```python
# Split into training (90%) and validation (10%) sets.
dataset_size = len(full_dataset)
train_size = int(0.9 * dataset_size)
val_size = dataset_size - train_size
accelerator.print(
    f"Total dataset size: {dataset_size}, Training size: {train_size}, Validation size: {val_size}"
)
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)
```

And then create two dataloaders: one for training & the other for evaluation (or validation)

```python
# Create DataLoaders.
train_dataloader = DataLoader(
    train_dataset,
    batch_size=per_device_train_batch_size,
    shuffle=True,
    collate_fn=data_collator,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=per_device_eval_batch_size,
    shuffle=False,
    collate_fn=data_collator,
)
```

Next, let's load our model using `ModernBertForSequenceClassification` which will add a final layer to the model's architecture including `num_labels` neurons.

```python
accelerator.print("Loading model...")
model = ModernBertForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels, ignore_mismatched_sizes=True
)
```

Let's setup our LoRA configuration

```python
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  #sequence classification task
    inference_mode=False,
    r=8,           #rank of the LoRA update matrices (hyperparameter)
    lora_alpha=32, #scaling factor
    lora_dropout=0.1,  # dropout probability applied to LoRA layers
    target_modules=["attn.Wqkv"],  #this is how the layer is called in ModernBERT.
)

model = get_peft_model(model, lora_config)
accelerator.print("LoRA-modified model loaded.")

device = accelerator.device
model.to(device)
```

Notice that we didn't setup the device manually. This is because `accelerate` does device placement automatically.

**Why LoRA?** LoRA reduces the number of trainable parameters. Instead of updating all the weights during backpropagation, we only update small trainable matrices. This means most of the model stays frozen and untouched. As a result, we don't need to store gradients or optimizer states for the majority of parameters, significantly reducing memory requirements. This makes the fine-tuning process much better suited for low-resource environments. We'll share some great resources to learn more about LoRA in the Resources section.

Next, let's create our optimizer & scheduler.

```python
# Create optimizer. Here we use AdamW with weight decay.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Optionally: set up a learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

#Most importantly, don't forget to prepare everything with your accelerator
model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader, scheduler
)
```

Now that everything is ready, let's write our training loop using plain PyTorch.

```python
best_val_accuracy = 0.0
global_step = 0

# ----- Training Loop -----
for epoch in range(num_train_epochs):
    model.train()
    running_loss = 0.0
    for step, batch in enumerate(train_dataloader):
    
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss) #backprop using the accelerator

        optimizer.step()

        scheduler.step()                               

        optimizer.zero_grad()                  
                   
        running_loss += loss.item()
        global_step += 1

        if global_step % logging_steps == 0:
            avg_loss = running_loss / logging_steps
            accelerator.print(f"Epoch [{epoch+1}/{num_train_epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {avg_loss:.4f}")
            running_loss = 0.0
```

Don't forget to run evaluation at the end of each epoch.

```python
for epoch in range(num_train_epochs):
    model.train()
    running_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        ... #previous code

    # ----- Validation at the End of Each Epoch -----
    model.eval()
    all_preds = []
    all_labels = []
    eval_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            outputs = model(**batch)
            eval_loss += outputs.loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            num_batches += 1

    avg_eval_loss = eval_loss / num_batches
    val_accuracy = accuracy_score(all_labels, all_preds)
    accelerator.print(f"Epoch [{epoch+1}/{num_train_epochs}] Validation Loss: {avg_eval_loss:.4f} | Accuracy: {val_accuracy:.4f}")
```

And of course, save the best-performing model after every epoch.

```python
for epoch in range(num_train_epochs):
    model.train()
    running_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        ... #previous code

    with torch.no_grad():
        for batch in val_dataloader:
            ... #previous code

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = model.state_dict()
        accelerator.print("Best model updated.")
```

Once training is complete, we can save the final best model. Note that we're not saving the full model, but only the LoRA adapters—since we wrapped our model with `get_peft_model()`. This further reduces memory usage.

```python
# ----- Load and Save the Best Model -----
model.load_state_dict(best_model_state)

output_dir = "./lora_modernbert_finetuned"
os.makedirs(output_dir, exist_ok=True)
accelerator.print("Saving best model and tokenizer...")
if hasattr(model, "module"):
    model.module.save_pretrained(output_dir)
else:
    model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
accelerator.print(f"Model saved to {output_dir}")
```

Finally, let's wrap everything in a `main()` function and place it in a `train.py` file (or any name you prefer).

```python
def main():
    ...

if __name__ == "__main__":
    main()
```

## Launch with Accelerate

```bash
#ON A SINGLE GPU
accelerate launch --mixed_precision=fp16 train.py #you can remove mixed_precision if you want

#ON MULTIPLE GPUs
accelerate launch --multi_gpu --mixed_precision "fp16" --num_processes 4 train.py
```

Once you've got a solid checkpoint, you can merge it with the base model for inference or further evaluation.

```python
from peft import PeftModel
base_model = ModernBertForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
model = PeftModel.from_pretrained(base_model, checkpoint_dir)
```

Although we didn't mention it earlier, `accelerate` will automatically use DDP (Distributed Data Parallel). Each GPU hosts a model copy that processes different batches. While the forward passes run independently, gradients are averaged across GPUs during backpropagation.

## Resources

Here are some great resources to dive deeper into DDP and LoRA.

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685): The original LoRA paper by Hu et al.
- [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704): The original DDP Pytorch implementation paper by Li et al.

Thanks for reading! If you have any questions, feel free to open an issue in this [repository](https://github.com/lemon42-ai/ThreatDetect-code-vulnerability-detection/tree/main).
