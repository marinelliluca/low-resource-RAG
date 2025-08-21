import os
import numpy as np
import pandas as pd
# import librosa
import torch

torch.manual_seed(42)
torch.set_float32_matmul_precision('medium')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model definition
import torch.nn as nn
from torch.utils.data import Dataset  
from pytorch_lightning.core import LightningModule  
from torch.nn import functional as F

# Training, inference and evaluation
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.special import softmax

###################
# Load embeddings #
###################


def load_embeddings_and_labels(
    groundtruth_df,
    mid_dict,
    emo_dict,
    cls_dict,
    embeddings_dir="music/clap_2023",
    input_dim=1024,
):
    """Load embeddings and labels for a given modality and embedding type.
    Args:
        groundtruth_df (pd.DataFrame): ground truth dataframe
        <something>_dict (dict): dictionaries containing the SELECTED classes for each classification task
    Returns:
        X (np.array): these should be embeddings (depends on what your .npy files contain)
        y_<something> (dict): dictionary containing the labels for each classification task
        ... (more dictionaries if needed)

    Example of <something>_dict:
    <something>_dict = {
        "target_of_toy_ad": [
            "Girls/women",
            "Boys/men"
            ], # skip mixed (assign -1)
        "another_column_in_grountruth_df": [
            'selected_class1',
            'selected_class2',
            ...
            ],
        }
    """

    # load embeddings
    X = np.empty((groundtruth_df.shape[0], input_dim))
    for i, stimulus_id in enumerate(groundtruth_df.index):
        emb = np.load(open(os.path.join(embeddings_dir, f"{stimulus_id}.npy"), 'rb'))
        if emb.ndim == 2:
            emb = emb.mean(axis=0) # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> !!!assuming that 0 is the time axis!!!!
        X[i] = emb

    y_mid = {}
    y_emo = {}
    y_cls = {}

    for task, classes in mid_dict.items():
        y_mid[task] = groundtruth_df[task].values
        y_mid[task] = [classes.index(x) if x in classes else -1 for x in y_mid[task]]
        y_mid[task] = np.array(y_mid[task])

    for task, classes in emo_dict.items():
        y_emo[task] = groundtruth_df[task].values
        y_emo[task] = [classes.index(x) if x in classes else -1 for x in y_emo[task]]
        y_emo[task] = np.array(y_emo[task])

    for task, classes in cls_dict.items():
        y_cls[task] = groundtruth_df[task].values
        y_cls[task] = [classes.index(x) if x in classes else -1 for x in y_cls[task]]
        y_cls[task] = np.array(y_cls[task])

    X = torch.from_numpy(X).float()
    y_mid = {k: torch.from_numpy(v).long() for k, v in y_mid.items()}
    y_emo = {k: torch.from_numpy(v).long() for k, v in y_emo.items()}
    y_cls = {k: torch.from_numpy(v).long() for k, v in y_cls.items()}

    return X, y_mid, y_emo, y_cls


#########
# Model #
#########

class DynamicDataset(Dataset):
    def __init__(self, X, y_mid, y_emo, y_cls):
        self.X = X
        self.y_mid = y_mid
        self.y_emo = y_emo
        self.y_cls = y_cls

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        y_mid = {k: v[idx] for k, v in self.y_mid.items()}
        y_emo = {k: v[idx] for k, v in self.y_emo.items()}
        y_cls = {k: v[idx] for k, v in self.y_cls.items()}
        return self.X[idx], y_mid, y_emo, y_cls

class DynamicClassifier(LightningModule):
    def __init__(
        self,
        input_dim,
        mid_dict,
        emo_dict,
        cls_dict,
    ):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.hidden = nn.Linear(input_dim, 512)

        self.bn_mid = nn.BatchNorm1d(512)
        self.bn_emo = nn.BatchNorm1d(512)
        self.bn_cls = nn.BatchNorm1d(512)

        self.hidden_mid = nn.Linear(512, 256)
        self.hidden_emo = nn.Linear(512, 256)
        self.hidden_cls = nn.Linear(512, 256)

        self.out = nn.ModuleDict(
            {
                "mid": nn.ModuleDict(
                    {k: nn.Linear(256, len(v)) for k, v in mid_dict.items()}
                ),
                "emo": nn.ModuleDict(
                    {k: nn.Linear(256, len(v)) for k, v in emo_dict.items()}
                ),
                "cls": nn.ModuleDict(
                    {k: nn.Linear(256, len(v)) for k, v in cls_dict.items()}
                ),
            }
        )

    def compute_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y, ignore_index=-1)

    def forward(self, x):
        x = self.batch_norm(x)
        x = F.relu(self.hidden(x))

        x_mid = self.bn_mid(x)
        x_mid = F.relu(self.hidden_mid(x_mid))

        x_emo = self.bn_emo(x)
        x_emo = F.relu(self.hidden_emo(x_emo))

        x_cls = self.bn_cls(x)
        x_cls = F.relu(self.hidden_cls(x_cls))

        x_mid = {k: v(x_mid) for k, v in self.out["mid"].items()}
        x_emo = {k: v(x_emo) for k, v in self.out["emo"].items()}
        x_cls = {k: v(x_cls) for k, v in self.out["cls"].items()}

        return x_mid, x_emo, x_cls

    def shared_step(self, batch, which_loss):
        x, y_mid, y_emo, y_cls = batch
        y_hat_mid, y_hat_emo, y_hat_cls = self(x)

        main_task = "target_of_toy_ad"
        loss_main = self.compute_loss(y_hat_cls[main_task], y_cls[main_task])

        loss_mid = sum([self.compute_loss(y_hat_mid[k], v) for k, v in y_mid.items()])

        loss_emo = sum([self.compute_loss(y_hat_emo[k], v) for k, v in y_emo.items()])

        loss_voice = sum([self.compute_loss(y_hat_cls[k], v) for k, v in y_cls.items() if k != main_task])

        main_task_weighing = (len(y_mid.keys()) + len(y_emo.keys()) + len(y_cls.keys()))
        loss_main = main_task_weighing * loss_main

        loss = loss_mid + loss_emo + loss_voice + loss_main

        self.log(which_loss, loss)

        return loss


    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train_loss")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val_loss")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=5e-3, amsgrad=True)

######################################
# Training, inference and evaluation #
######################################

def inference(model, X, device=device):
    """Inference for a given model and input data.
    Args:
        model (LightningModule): model must be in eval mode
        X (np.array or torch.Tensor): input data
    Returns:
        y_mid_logits (dict): dictionary containing the logits for each classification task
        y_emo_logits (dict): ...
        y_cls_logits (dict): ...
    """
    with torch.no_grad():
        y_mid_logits, y_emo_logits, y_cls_logits = model(
            X.to(device) if isinstance(X, torch.Tensor) 
            else torch.from_numpy(X).float().to(device)
        )
    return y_mid_logits, y_emo_logits, y_cls_logits

def evaluate(model, config, X, y_mid_true, y_emo_true, y_cls_true):
    f1s = {
        "mid": {k: None for k in config["mid_dict"]},
        "emo": {k: None for k in config["emo_dict"]},
        "cls": {k: None for k in config["cls_dict"]}
    }
    y_mid_pred, y_emo_pred, y_cls_pred = inference(model, X)

    for y_true, y_pred, task_name in zip(
        [y_mid_true, y_emo_true, y_cls_true], 
        [y_mid_pred, y_emo_pred, y_cls_pred], 
        ["mid", "emo", "cls"]
    ):
        for k in config[task_name + "_dict"]:
            y_true_temp = y_true[k].cpu().numpy()
            y_pred_temp = y_pred[k].cpu().numpy()

            skip_unlabelled = y_true_temp != -1
            y_true_temp = y_true_temp[skip_unlabelled]
            y_pred_temp = y_pred_temp[skip_unlabelled]

            y_pred_temp = np.argmax(y_pred_temp, axis=1)
            
            f1s[task_name][k] = f1_score(y_true_temp, y_pred_temp, average="weighted")

    return f1s

def train_music_model(config, X, y_mid, y_emo, y_cls, device=device, return_metrics=False):

    # get validation set
    music_train_index, music_val_index = train_test_split(range(X.shape[0]), test_size=0.3, random_state=42)

    # get the split
    X_train, X_val = X[music_train_index], X[music_val_index]
    get_split_from_dict = lambda y_dict, train_index, val_index: (
        {k: y_dict[k][train_index] for k in y_dict},{k: y_dict[k][val_index] for k in y_dict}
    )
    y_mid_train, y_mid_val = get_split_from_dict(y_mid, music_train_index, music_val_index)
    y_emo_train, y_emo_val = get_split_from_dict(y_emo, music_train_index, music_val_index)
    y_cls_train, y_cls_val = get_split_from_dict(y_cls, music_train_index, music_val_index)

    # move all to device
    X_train, X_val = X_train.to(device), X_val.to(device)
    y_mid_train = {k: v.to(device) for k, v in y_mid_train.items()}
    y_mid_val = {k: v.to(device) for k, v in y_mid_val.items()}
    y_emo_train = {k: v.to(device) for k, v in y_emo_train.items()}
    y_emo_val = {k: v.to(device) for k, v in y_emo_val.items()}
    y_cls_train = {k: v.to(device) for k, v in y_cls_train.items()}
    y_cls_val = {k: v.to(device) for k, v in y_cls_val.items()}

    train_loader = DataLoader(
        DynamicDataset(X_train, y_mid_train, y_emo_train, y_cls_train), 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=0, 
        drop_last=True
    )
    val_loader = DataLoader(
        DynamicDataset(X_val, y_mid_val, y_emo_val, y_cls_val), 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=0, 
        drop_last=False
    )

    # model params
    params = {
        "input_dim": X.shape[1],
        "mid_dict": config["mid_dict"],
        "emo_dict": config["emo_dict"],
        "cls_dict": config["cls_dict"],
    }

    # train
    model = DynamicClassifier(**params) 

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    trainer = Trainer(
        max_epochs=config["max_epochs"],
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="val_loss", patience=50),
        ],
        enable_progress_bar=False,
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(model, train_loader, val_loader)

    # load best model
    model = DynamicClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path, **params
    )
    model.eval()
    
    if return_metrics:
        f1s_val = evaluate(model, config, X_val, y_mid_val, y_emo_val, y_cls_val)
        return model, f1s_val
    
    return model

#################################
# Inference to text description #
#################################

logits_to_probs = lambda y_logits: {k: softmax(logit, axis=1).flatten() for k, logit in y_logits.items()}
logits_to_preds = lambda y_logits: {k: int(np.argmax(prob)) for k, prob in logits_to_probs(y_logits).items()}
logits_to_labels = lambda y_logits, task_dict: {k: task_dict[k][pred] for k, pred in logits_to_preds(y_logits).items() if k in task_dict}

def id_to_labels(model, config, stimulus_id, embeddings_dir="music/clap_2023"):

    if not os.path.exists(embeddings_dir):
        raise ValueError(f"Embeddings directory {embeddings_dir} does not exist")

    # load embedding
    embedding = np.load(open(os.path.join(embeddings_dir, f"{stimulus_id}.npy"), 'rb'))
    if embedding.ndim == 2:
        embedding = embedding.mean(axis=0) # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> !!!assuming that 0 is the time axis!!!!

    # compute music summary
    y_mid_logits, y_emo_logits, y_cls_logits = inference(model, embedding[np.newaxis,:])

    y_mid_logits = {k: v.cpu().numpy() for k, v in y_mid_logits.items()}
    y_emo_logits = {k: v.cpu().numpy() for k, v in y_emo_logits.items()}
    y_cls_logits = {k: v.cpu().numpy() for k, v in y_cls_logits.items()}

    # get labels
    y_mid_label = logits_to_labels(y_mid_logits, config["mid_dict"])
    y_emo_label = logits_to_labels(y_emo_logits, config["emo_dict"])
    y_cls_label = logits_to_labels(y_cls_logits, config["cls_dict"])

    return y_mid_label, y_emo_label, y_cls_label, (y_mid_logits, y_emo_logits, y_cls_logits)

def keep_only_relevant_labels(y_logits, mn_prb, task_dict):
    y_probs = logits_to_probs(y_logits)
    return {
        k: task_dict[k][pred] for k, pred in logits_to_preds(y_logits).items()
        if k in task_dict and np.max(y_probs[k][pred]) > mn_prb
    }

def logits_to_text(y_mid_logits, y_emo_logits, y_cls_logits, config):

    y_mid_label = keep_only_relevant_labels(y_mid_logits, 0.5, config["mid_dict"])
    y_emo_label = keep_only_relevant_labels(y_emo_logits, 0.5, config["emo_dict"])
    y_cls_label = keep_only_relevant_labels(y_cls_logits, 0.6, config["cls_dict"])

    y_mid_label = {k: v for k, v in y_mid_label.items() if v != "skip_this"} # see music/config_inference.yaml
    y_emo_label = {k: v for k, v in y_emo_label.items() if v != "skip_this"}
    y_cls_label = {k: v for k, v in y_cls_label.items() if v != "skip_this"}

    mid_text = "The music is: "
    txts = []
    for k, v in y_mid_label.items():
        txts.append(f"{v}")

    if txts:
        mid_text += ", ".join(txts) + "."
    else:
        mid_text += ""

    emo_text = ""
    txts = []
    for k, v in y_emo_label.items():
        txts.append(f"{v} {k}")

    if txts:
        emo_text += ", ".join(txts) + "."
    else:
        emo_text += ""

    cls_text = "The voices of the narrators are: "
    txts = []
    for k, v in y_cls_label.items():
        if k != "target_of_toy_ad":
            txts.append(v)

    if txts:
        cls_text += ", ".join(txts) + ". "
    else:
        cls_text += ""

    # NB: do NOT add "target_of_toy_ad" to cls_text to the soundtrack description, as it introduces noise
    #if "target_of_toy_ad" in y_cls_label:
    #    cls_text += f"Overall, this soundtrack could be aimed at {y_cls_label['target_of_toy_ad']}."
    
    return mid_text, emo_text, cls_text