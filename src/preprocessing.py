from torchvision import transforms
from global_vars import *

processor = load_processor()

# image scaling, normalizing and resizing
img_transformer = transforms.Compose([
    transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std)
])

def preprocess(examples):
  pixel_values = [img_transformer(img.convert("RGB")) for img in examples["image"]]
  labels = [class_labels[label] for label in examples["label"]]

  tokens = processor.tokenizer(labels,  max_length = 64, padding = "max_length", truncation = True, return_attention_mask = True)

  tokens["pixel_values"] = pixel_values
  return tokens
