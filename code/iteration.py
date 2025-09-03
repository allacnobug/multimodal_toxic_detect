import torch
import torch.nn as nn
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os

def train_one_epoch(model, train_dataloader, optimizer, config):
    model = model.to(config.device)
    model.train()
    total_loss = 0.0
    step=0

    for batch in tqdm(train_dataloader):
        step+=1
        batch = {key: value.to(config.device) for key, value in batch.items()}
 
        optimizer.zero_grad()
        outputs = model(batch)
        loss = outputs["loss"]
        
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        
        total_loss += loss.item()
        if step % 50 == 0:
            avg_loss_so_far = total_loss / step
            print(f"Step {step}, Avg Loss: {avg_loss_so_far:.4f}")

    average_loss = total_loss / len(train_dataloader)
    
    return average_loss
    

def validate(model, val_dataloader, config, devices=None):
    model = model.to(config.device)
    model.eval()
    task_metrics = {task: {"accuracy": 0.0, "f1": 0.0} for task in config.tasks}
    all_labels = {task: [] for task in config.tasks}
    all_predictions = {task: [] for task in config.tasks}

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            inputs = {key: value.to(config.device) for key, value in batch.items()}
            
            outputs = model(inputs)

            for task in config.tasks:
                # print(outputs[task], batch[task])

                task_predictions = torch.argmax(outputs[task], dim=1).detach().cpu().numpy()
                # task_labels = torch.argmax(batch[task], dim=1).detach().cpu().numpy()
                task_labels = batch[task].detach().cpu().numpy()

                # task_accuracy = accuracy_score(task_labels, task_predictions)
                # task_f1 = f1_score(task_labels, task_predictions, average="macro")

                # task_metrics[task]["accuracy"] += task_accuracy
                # task_metrics[task]["f1"] += task_f1

                all_labels[task].extend(task_labels)
                all_predictions[task].extend(task_predictions)

    for task in config.tasks:
        # task_metrics[task]["accuracy"] /= len(val_dataloader)
        # task_metrics[task]["f1"] /= len(val_dataloader)
        task_metrics[task]["accuracy"] = accuracy_score(all_labels[task], all_predictions[task])
        task_metrics[task]["f1"] = f1_score(all_labels[task], all_predictions[task], average="macro")
        task_metrics[task]["classification_report"] = classification_report(all_labels[task], all_predictions[task])
        print(task, " " , task_metrics[task]["accuracy"] , " ", task_metrics[task]["f1"], "\n", task_metrics[task]["classification_report"], "\n")
        
    return task_metrics, all_labels, all_predictions

def train_model(model, train_dataloader, val_dataloader, config, num_epochs, track_task, track_metric, devices=None):
    
    model = model.to(config.device)
    # 冻结预训练模型
    # for param in model.video_encoder.parameters():
    #     param.requires_grad = False

    # for param in model.audio_encoder.parameters():
    #     param.requires_grad = False

    history = {"train_validation": []}
    
    print("Video encoder frozen:", all(not p.requires_grad for p in model.video_encoder.parameters()))
    print("Audio encoder frozen:", all(not p.requires_grad for p in model.audio_encoder.parameters()))

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=1e-5, weight_decay=1e-3)
    
    best_val_metric = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} out of {num_epochs}")

        # Training
        train_loss = train_one_epoch(model, train_dataloader, optimizer, config)
        
        print(train_loss)

        # Validation
        val_metrics, _, _ = validate(model, val_dataloader, config)
        
        # print(val_metrics)

        # Save metrics to history
        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_metrics": val_metrics
        }
        
        history["train_validation"].append(epoch_data)
        
        json_save_path = config.results_directory + config.file_name + ".json"
        os.makedirs(config.results_directory, exist_ok=True)

        # Save to JSON file
        with open(json_save_path, 'w') as json_file:
            json.dump(history, json_file)

        track_task = 'offensive'
        track_metric = 'f1'
        if val_metrics[track_task][track_metric] > best_val_metric:
            best_val_metric = val_metrics[track_task][track_metric]
            torch.save(model.state_dict(), config.directory + config.file_name + ".pth")
            print(f"✅ Saved best model with {track_task}-{track_metric}: {best_val_metric:.4f}")

    
    print("Training finished!")


def test(model, test_dataloader, config, devices=None):
    model = model.to(config.device)
    model.eval()
    all_labels = {task: [] for task in config.tasks}
    all_predictions = {task: [] for task in config.tasks}

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            inputs = {key: value.to(config.device) for key, value in batch.items()}
            outputs = model(inputs)

            for task in config.tasks:
                task_predictions = torch.argmax(outputs[task], dim=1).detach().cpu().numpy()
                task_labels = batch[task].detach().cpu().numpy()
                all_labels[task].extend(task_labels)
                all_predictions[task].extend(task_predictions)
    print("labels",all_labels) 
    print("predictions",all_predictions)           
    return all_labels, all_predictions