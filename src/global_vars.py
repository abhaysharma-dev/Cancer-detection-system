import torch
import streamlit as st
from datasets import load_dataset, load_from_disk
from global_vars import load_processor,load_model
from transformers import AutoModel, AutoProcessor

processor = load_processor()
dataset = load_dataset()

class_labels = dataset["train"].features["label"].names
std = processor.image_processor.image_std
mean = processor.image_processor.image_mean
path = "../models"

@st.cache_resource
def load_model():
    return AutoModel.from_pretrained(path)

@st.cache_resource
def load_processor():
    return AutoProcessor.from_pretrained(path)

@st.cache_data
def load_dataset():
    return load_dataset("1aurent/NCT-CRC-HE",split= "CRC_VAL_HE_7K")