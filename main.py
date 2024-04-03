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

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg : DictConfig) -> None:

    ## huggingface model
    processor = AutoImageProcessor.from_pretrained(cfg.model.pretrained_weights)

    # inputs = processor(image, return_tensors="pt")

    ## dataset from robustbench
    # import robustbench
    # from robustbench.data import load_cifar10
    # from robustbench.utils import load_model, clean_accuracy

    # images, labels = load_cifar10(n_examples=5)  # tensor, tesnor
    # print('[Data loaded]')

    ## dataset load from huggingface
    train_dataset, test_dataset = load_dataset(cfg.dataset.name, split=["train[:5000]", "test[:2000]"])  # dataset from huggingface

    label_names = train_dataset.features["label"].names
    num_labels = len(label_names)

    # train_dataset = dataset["train[:5000]"]
    # test_dataset = dataset["test[:2000]"]
    print(train_dataset)

    # image = dataset["test"].features["img"]

    id2label = {id:label for id, label in enumerate(train_dataset.features['label'].names)}
    label2id = {label:id for id,label in id2label.items()}
    # print(label2id)

    # image_mean, image_std = processor.image_mean, processor.image_std
    # size = processor.size["height"]

    # normalize = Normalize(mean=image_mean, std=image_std)
    # _train_transforms = Compose(
    #         [
    #             RandomResizedCrop(size),
    #             RandomHorizontalFlip(),
    #             ToTensor(),
    #             normalize,
    #         ]
    #     )

    # _val_transforms = Compose(
    #         [
    #             Resize(size),
    #             CenterCrop(size),
    #             ToTensor(),
    #             normalize,
    #         ]
    #     )

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

    # print(image)

    ## dataset load from torchvision
    # transform = transforms.Compose(
    # [transforms.ToTensor(),
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # train_dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, num_workers=2)

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                     download=True, transform=transform)
    # train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                         shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                     download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                         shuffle=False, num_workers=2)

    # print(train_dataset)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)

    #### Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained(cfg.model.pretrained_weights, num_labels=num_labels, ignore_mismatched_sizes=True, id2label=id2label, label2id=label2id)
    model = model.to(device)
    atk = Attack_PGD(model, device, targeted=False, eps=8/255, alpha=2/225, steps=10, random_start=True)
    atk = atk.to(device)

    # only test
    # model.eval()

    ##########################

    optimizer, lr_scheduler, model, num_training_steps = setup_optimizer(cfg, model, train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    #### training on the target dataset
    for epoch in range(cfg.training.num_epochs):
        model.train()
        for ind, batch in enumerate(train_dataloader):
            # print(batch)
            image = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            logits = model(image).logits  # (1, num_classes)
            predicted = logits  # to use cross entropy loss

            # print(labels)
            # print(predicted)
            # print(welkfjlew)
            
            # update loss and optimizer
            loss = F.cross_entropy(predicted, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    
    torch.cuda.empty_cache()  # not gauranteed

    # args = TrainingArguments(
    #     f"test-cifar-10",
    #     save_strategy="epoch",
    #     evaluation_strategy="epoch",
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=10,
    #     per_device_eval_batch_size=4,
    #     num_train_epochs=1,
    #     weight_decay=0.01,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="accuracy",
    #     logging_dir='logs',
    #     remove_unused_columns=False,
    # )

    # metric = load_metric('accuracy')

    # def compute_metrics(eval_pred):
    #     predictions, labels = eval_pred
    #     predictions = np.argmax(predictions, axis=1)
    #     return metric.compute(predictions=predictions, references=labels)
        
    # trainer = Trainer(
    #     model,
    #     args, 
    #     train_dataset=train_dataset,
    #     eval_dataset=test_dataset,
    #     data_collator=collate_fn,
    #     compute_metrics=compute_metrics,
    #     tokenizer=processor,
    # )

    # trainer.train()

    #############################################################

    #### test accuracy for adversarial attack
    model.eval()

    # outputs = trainer.predict(test_dataset)
    # print(outputs.metrics)
    # print(elwkfjlwewef)
    
    correct = 0
    correct_adv = 0
    total = 0
    preds_origin = []
    preds_adver = []
    log_path = os.path.join(cfg.dataset.logs, cfg.model.attack_method)
    for ind, batch in enumerate(test_dataloader):
        # print(batch)
        # logits = model(**inputs).logits
        image = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        logits = model(image).logits  # (1, num_classes)
        predicted_origin = logits.argmax(-1).item()

        correct += (predicted_origin == labels).sum().item()
        # total += predicted.size(0)
        total += len([predicted_origin])

        ####
        # adversarial attack
        adv_images = atk(image, labels)

        logits = model(adv_images).logits
        predicted_adver = logits.argmax(-1).item()

        correct_adv += (predicted_adver == labels).sum().item()     

        ## save log
        preds_origin.append(predicted_origin)
        preds_adver.append(predicted_adver)
        # if save_log:
            # save_image(image, os.path.join(log_path, 'img_origin_' + ind +'.png'))
            # save_image(image, os.path.join(log_path, 'img_adver_' + ind +'.png'))

    ## make prediction logs    
    # if save_log:


    accuracy_origin = correct / float(total)
    print("{} / {}".format(correct, total))
    print("Original Accuracy: {}".format(accuracy_origin))
    accuracy_adver = correct_adv / float(total)
    print("Adversarial Accuracy: {}".format(accuracy_adver))

    print(weklfjwel)


if __name__ == '__main__':
    main()
