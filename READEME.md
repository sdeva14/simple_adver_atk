# Simple Framework of Adversarial Attack on Images

Goal: THe framework of maniplulating images by adding adversarial noises

## Functions
full_pipeline(cfg, is_finetune=True, is_attack=True)
- cfg (hydra config -> config.yaml)
- is_finetune: whether to finetune on the target dataset; the model and dataset are given in the hydra config
- is_attack: whether to applying adversarial attack; the attack method is given in the hydra config

prepare_dataloader(cfg)
- return train and test dataloaders

prepare_models_atk(cfg, num_labels, id2label, label2id, is_attack)
- return model, attak_class, device

finetune(cfg, model, train_dataloader, device)
- return fine-tuned model, optimizer, and scheduler

evaluate(cfg, device, test_dataloader, model, atk, save_log=False)
- evaluate the model on the test dataloader, optionally with the attack method
- print accuracy

## Future Work
Compatibility
- Huggingface libraries are used to maximize compatibility with other datasets and models, supported by Huggingface. Still, it can be prepared to deal with the modules from other libraries

More attack method
- At this point, only PGD is implemented. More attack methods can be implemented, and more pre-processing methods can be implemented

Log functions
- At this point, basic log function is provided, saving adversarial images and prediction logs. More log functions can be implemented as the requirements.

## Acknowledge
Huggingface
"adversarial-attacks-pytorch" library; https://github.com/Harry24k/adversarial-attacks-pytorch