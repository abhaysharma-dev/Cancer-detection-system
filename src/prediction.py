import evaluate
from torch.utils.data import DataLoader
import torch 
from global_vars import load_processor , load_model, class_labels

model = load_model()
processor = load_processor()

def image_collate_fn(batch):
    return batch

def prediction(images):
    device  =  "cuda" if torch.cuda.is_available() else "cpu"
    # test_loader = DataLoader(
    #     images["image"],
    #     batch_size = 64,
    #     shuffle = False,
    #     collate_fn = image_collate_fn
    # )

    model = model.to(device)
    model.eval()
    
    inputs = processor(text = class_labels,
                    images = images,
                    padding = "max_length",
                    return_tensors = "pt"
                    ).to(device)

    with torch.no_grad():
        output = model(**inputs)
    predictions = output.logits_per_image.argmax(dim = 1)
    return predictions