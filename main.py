import os

from transformers import AutoImageProcessor, ResNetForImageClassification, ViTForImageClassification

import torchvision
from torchvision import models
import torchvision.datasets as datasets_torchvision

from torch.utils.data import DataLoader

import torch
from datasets import load_dataset

from atk_method.PGD import Attack_PGD

import hydra
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from transformers import DefaultDataCollator
from tqdm.auto import tqdm
from transformers import AdamW, get_scheduler
from torch.nn import functional as F

from torchvision.utils import save_image

from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_metric

import json

def setup_optimizer(cfg, model, train_dataloader):
    optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate)
    num_training_steps = cfg.training.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    return optimizer, lr_scheduler, model, num_training_steps

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def preprocess_dataset(cfg, train_dataset, test_dataset):
    '''
        intput:
        - cfg (hydra config)
        - train_dataset
        - test_dataset
    
        output:
        - train_dataset (preprocessed by image processor)
        - test_dataset (preprocessed by image processor)
    '''

    ## huggingface image pre-processor
    processor = AutoImageProcessor.from_pretrained(cfg.model.pretrained_weights)

    mu, sigma = processor.image_mean, processor.image_std #get default mu,sigma
    size = processor.size

    norm = Normalize(mean=mu, std=sigma) #normalize image pixels range to [-1,1]

    # resize 3x32x32 to 3x224x224 -> convert to Pytorch tensor -> normalize
    _train_transforms = Compose([
        Resize(size['height']),
        ToTensor(),
        norm
    ])
    _val_transforms = Compose([
        Resize(size['height']),
        ToTensor(),
        norm
    ]) 

    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    def val_transforms(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    ## preprocessing images
    train_dataset.set_transform(train_transforms)
    test_dataset.set_transform(val_transforms)

    return train_dataset, test_dataset

def prepare_models_atk(cfg, num_labels, id2label, label2id, is_attack=True):
    '''
        intput:
        - cfg (hydra config)
        - num_labels
        - id2label
        - label2id
    
        output:
        - model
        - attack method class
        - device
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained(cfg.model.pretrained_weights, num_labels=num_labels, ignore_mismatched_sizes=True, id2label=id2label, label2id=label2id)
    model = model.to(device)

    ## adversarial attack method
    atk = None
    if is_attack:
        atk = Attack_PGD(model, device, targeted=False, eps=8/255, alpha=2/225, steps=10, random_start=True)
        atk = atk.to(device)

    return model, atk, device

def prepare_dataloader(cfg):
    '''
        intput:
        - cfg (hydra config)
        output:
        - train_dataloader
        - test_dataloader
        - num_labels
        - id2label
        - label2id
    '''
    ## load dataset load from huggingface
    train_dataset, test_dataset = load_dataset(cfg.dataset.name, split=["train[:5000]", "test[:2000]"])  # dataset from huggingface (sample number should be parameterized later)

    label_names = train_dataset.features["label"].names
    num_labels = len(label_names)
    # print(train_dataset)

    id2label = {id:label for id, label in enumerate(train_dataset.features['label'].names)}
    label2id = {label:id for id,label in id2label.items()}
    # print(label2id)

    train_dataset, test_dataset = preprocess_dataset(cfg, train_dataset, test_dataset)

    ## dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)

    return train_dataloader, test_dataloader, num_labels, id2label, label2id

def full_pipeline(cfg, is_finetune=True, is_attack=True):
    '''
        intput:
        - cfg (hydra config)
        - finetune (whether finetune or not; model and datasets are given in the hydra config)
        - attack (whether attack or not; attack method is given in the hydra config)
        output:
        - model
    '''

    train_dataloader, test_dataloader, num_labels, id2label, label2id = prepare_dataloader(cfg)

    model, atk, device = prepare_models_atk(cfg, num_labels, id2label, label2id, is_attack)

    if is_finetune:
        model, optimizer, lr_scheduler = finetune(cfg, model, train_dataloader, device)

    evaluate(cfg, device, test_dataloader, model, atk, save_log=False)

    return

def finetune(cfg, model, train_dataloader, device):

    '''
        intput:
        - cfg (hydra config)
        - model to classify
        - train dataloader
        - device
        output:
        - model (finetuned on the data)
    '''

    optimizer, lr_scheduler, model, num_training_steps = setup_optimizer(cfg, model, train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    #### training on the target dataset
    for epoch in range(cfg.training.num_epochs):
        model.train()
        for ind, batch in enumerate(train_dataloader):

            image = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            logits = model(image).logits  # (1, num_classes)
            predicted = logits  # to use cross entropy loss
            
            # update loss and optimizer
            loss = F.cross_entropy(predicted, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    
    torch.cuda.empty_cache()  # not gauranteed

    return model, optimizer, lr_scheduler

def evaluate(cfg, device, test_dataloader, model, atk=None, save_log=False):

    '''
        input:
        - cfg (hydra config)
        - device
        - testa_dataloader
        - model to classify
        - atk (attack method, if none, then do not evaluate with attack but only evaluate on the original dataset)
        - save_log (whether to save the log predictions and attacked images)
        output:
        - adverb images
    '''
    model.eval()

    # outputs = trainer.predict(test_dataset)
    # print(outputs.metrics)
    # print(elwkfjlwewef)
    
    correct = 0
    correct_adv = 0
    total = 0
    preds_origin = []
    preds_adver = []
    labels_list = []
    log_path = os.path.join(cfg.dataset.logs, cfg.model.attack_method)

    adverb_image_list = []
    for ind, batch in enumerate(test_dataloader):
        image = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        logits = model(image).logits  # (1, num_classes)
        predicted_origin = logits.argmax(-1).item()

        correct += (predicted_origin == labels).sum().item()
        total += len([predicted_origin])

        ####
        # adversarial attack
        if atk is not None:
            adv_images = atk(image, labels)

            logits = model(adv_images).logits
            predicted_adver = logits.argmax(-1).item()

            correct_adv += (predicted_adver == labels).sum().item()   

            adverb_image_list.append(adv_images)  

        ## save log
        preds_origin.append(predicted_origin)
        if atk is not None:
            preds_adver.append(predicted_adver)
        labels_list.append(labels)
        if cfg.training.save_log:
            save_image(image, os.path.join(log_path, 'img_origin_' + ind +'.png'))
            save_image(image, os.path.join(log_path, 'img_adver_' + ind +'.png'))

    # make prediction logs    
    if cfg.training.save_log:
        output_path = os.path.join(log_path, "predictions.log")  # pred_origin; pred_adversarial; labels
        with open(output_path, "w") as outfile:
            json.dump(preds_origin + preds_adver + labels_list, outfile)

    accuracy_origin = correct / float(total)
    print("{} / {}".format(correct, total))
    print("Original Accuracy: {}".format(accuracy_origin))
    if atk is not None:
        accuracy_adver = correct_adv / float(total)
        print("Adversarial Accuracy: {}".format(accuracy_adver))

    outputs = []
    outputs.append(adverb_image_list)

    return outputs


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg : DictConfig) -> None:

    full_pipeline(cfg)


if __name__ == '__main__':
    main()
