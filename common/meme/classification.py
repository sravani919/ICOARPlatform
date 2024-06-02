import json
import os
import ssl
from collections import Counter

import clip
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoConfig, AutoModel, AutoTokenizer, MMBTConfig, MMBTForClassification

ssl._create_default_https_context = (
    ssl._create_default_https_context
) = ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
print("Loading model completed")
for p in clip_model.parameters():
    p.requires_grad = False

num_image_embeds = 4
num_labels = 1
gradient_accumulation_steps = 20
data_dir = "/home/psilimk"
max_seq_length = 80
max_grad_norm = 0.5
train_batch_size = 16
eval_batch_size = 16
image_encoder_size = 288
image_features_size = 640
num_train_epochs = 5


def slice_image(im, desired_size):
    """
    Resize and slice image
    """
    old_size = im.size
    ratio = float(desired_size) / min(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)
    ar = np.array(im)
    images = []
    if ar.shape[0] < ar.shape[1]:
        middle = ar.shape[1] // 2
        half = desired_size // 2

        images.append(Image.fromarray(ar[:, :desired_size]))
        images.append(Image.fromarray(ar[:, middle - half : middle + half]))
        images.append(Image.fromarray(ar[:, ar.shape[1] - desired_size : ar.shape[1]]))
    else:
        middle = ar.shape[0] // 2
        half = desired_size // 2

        images.append(Image.fromarray(ar[:desired_size, :]))
        images.append(Image.fromarray(ar[middle - half : middle + half, :]))
        images.append(Image.fromarray(ar[ar.shape[0] - desired_size : ar.shape[0], :]))

    return images


def resize_pad_image(im, desired_size):
    """
    Resize and pad image to a desired size
    """
    old_size = im.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))

    return new_im


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, max_seq_length):
        self.data = [json.loads(d) for d in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = torch.LongTensor(self.tokenizer.encode(self.data[index]["text"], add_special_tokens=True))
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        sentence = sentence[: self.max_seq_length]

        label = torch.FloatTensor([self.data[index]["label"]])

        image = Image.open(os.path.join(self.data_dir, self.data[index]["img"])).convert("RGB")
        sliced_images = slice_image(image, 288)
        sliced_images = [np.array(self.transforms(im)) for im in sliced_images]
        image = resize_pad_image(image, image_encoder_size)
        image = np.array(self.transforms(image))

        sliced_images = [image] + sliced_images
        sliced_images = torch.from_numpy(np.array(sliced_images)).to(device)

        return {
            "image_start_token": start_token,
            "image_end_token": end_token,
            "sentence": sentence,
            "image": sliced_images,
            "label": label,
        }

    def get_label_frequencies(self):
        label_freqs = Counter()
        for row in self.data:
            label_freqs.update([row["label"]])
        return label_freqs

    def get_labels(self):
        labels = []
        for row in self.data:
            labels.append(row["label"])
        return labels


def collate_fn(batch):
    lens = [len(row["sentence"]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = input_row["sentence"]
        mask_tensor[i_batch, :length] = 1

    img_tensor = torch.stack([row["image"] for row in batch])
    tgt_tensor = torch.stack([row["label"] for row in batch])
    img_start_token = torch.stack([row["image_start_token"] for row in batch])
    img_end_token = torch.stack([row["image_end_token"] for row in batch])

    return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, tgt_tensor


class ClipEncoderMulti(nn.Module):
    def __init__(self, num_embeds, num_features=image_features_size):
        super().__init__()
        self.model = clip_model
        self.num_embeds = num_embeds
        self.num_features = num_features

    def forward(self, x):
        # 4x3x288x288 -> 1x4x640
        out = self.model.encode_image(x.view(-1, 3, 288, 288))
        out = out.view(-1, self.num_embeds, self.num_features).float()
        return out  # Bx4x640


model_name = "Hate-speech-CNERG/bert-base-uncased-hatexplain"
transformer_config = AutoConfig.from_pretrained(model_name)
transformer = AutoModel.from_pretrained(model_name, config=transformer_config)
img_encoder = ClipEncoderMulti(num_image_embeds)
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

config = MMBTConfig(transformer_config, num_labels=num_labels, modal_hidden_size=image_features_size)
model = MMBTForClassification(config, transformer, img_encoder)
model.to(device)

model = MMBTForClassification(config, transformer, img_encoder)
state_dict = torch.load("model/model-embs4-seq80-auc0.760-loss1.175-acc0.671.pt", map_location=device)
print("Model Loaded")
model.load_state_dict(state_dict["model_state_dict"])
model.to(device)

num_labels = 1
data_dir = "data"
test_batch_size = 16


class TestJsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, max_seq_length):
        self.data = [json.loads(dp) for dp in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = torch.LongTensor(self.tokenizer.encode(self.data[index]["text"], add_special_tokens=True))
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        sentence = sentence[: self.max_seq_length]

        id = torch.LongTensor([self.data[index]["id"]])
        image = Image.open(os.path.join(self.data_dir, self.data[index]["img"])).convert("RGB")
        sliced_images = slice_image(image, 288)
        sliced_images = [np.array(self.transforms(im)) for im in sliced_images]
        image = resize_pad_image(image, image_encoder_size)
        image = np.array(self.transforms(image))
        sliced_images = [image] + sliced_images
        sliced_images = torch.from_numpy(np.array(sliced_images)).to(device)

        return {
            "image_start_token": start_token,
            "image_end_token": end_token,
            "sentence": sentence,
            "image": sliced_images,
            "id": id,
        }


def final_collate_fn(batch):
    lens = [len(row["sentence"]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = input_row["sentence"]
        mask_tensor[i_batch, :length] = 1

    img_tensor = torch.stack([row["image"] for row in batch])
    id_tensor = torch.stack([row["id"] for row in batch])
    img_start_token = torch.stack([row["image_start_token"] for row in batch])
    img_end_token = torch.stack([row["image_end_token"] for row in batch])

    return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, id_tensor


def load_test_examples(test_file="test_seen2.jsonl"):
    path = os.path.join(data_dir, test_file)
    dataset = TestJsonlDataset(path, tokenizer, preprocess, max_seq_length - num_image_embeds - 2)
    return dataset


def final_prediction(model, dataloader):
    preds = None
    proba = None
    all_ids = None
    for batch in dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            ids = batch[5]
            inputs = {
                "input_ids": batch[0],
                "input_modal": batch[2],
                "attention_mask": batch[1],
                "modal_start_tokens": batch[3],
                "modal_end_tokens": batch[4],
                "return_dict": False,
            }
            outputs = model(**inputs)
            logits = outputs[0]
        if preds is None:
            all_ids = ids.detach().cpu().numpy()
            preds = torch.sigmoid(logits).detach().cpu().numpy() > 0.5
            proba = torch.sigmoid(logits).detach().cpu().numpy()
        else:
            all_ids = np.append(all_ids, ids.detach().cpu().numpy(), axis=0)
            preds = np.append(preds, torch.sigmoid(logits).detach().cpu().numpy() > 0.5, axis=0)
            proba = np.append(proba, torch.sigmoid(logits).detach().cpu().numpy(), axis=0)

    result = {
        "ids": all_ids,
        "preds": preds,
        "probs": proba,
    }

    return result


def classify_memes():
    final_test = load_test_examples()

    final_test_sampler = SequentialSampler(final_test)

    final_test_dataloader = DataLoader(
        final_test, sampler=final_test_sampler, batch_size=test_batch_size, collate_fn=final_collate_fn
    )

    results = final_prediction(model, final_test_dataloader)

    results["ids"] = results["ids"].reshape(-1)
    results["preds"] = results["preds"].reshape(-1)
    results["probs"] = results["probs"].reshape(-1)
    return results
