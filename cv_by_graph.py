import pandas as pd
import torch
import tqdm
from sklearn.model_selection import KFold
#from langgraph.graph.graph import CompiledGraph
import logging
from workflow import WorkflowHandler
import os
import json

# Music training
from music.tools import (
    load_embeddings_and_labels, 
    train_music_model,
    id_to_labels,
    logits_to_probs,
    logits_to_text
)

torch.manual_seed(42)
torch.set_float32_matmul_precision('medium')

#######
# TODO: add metrics for cues detection and correction!
#######

class CVbyGraph():
    def __init__(
        self, 
        df: pd.DataFrame, 
        workflow: WorkflowHandler, 
        classes_key: str,
        music_config_training: dict,
        music_config_inference: dict,
        music_df: pd.DataFrame,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = 42
    ):  
        self.logger = logging.getLogger(__name__)

        # Main graph and data
        self.df = df
        self.workflow = workflow
        self.classes_key = classes_key

        # Cross-validation
        self.folds = {
            "train_ids": [],
            "test_ids": [],
        }
        kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        for train_idx, test_idx in kf.split(self.df):
            self.folds["train_ids"].append(train_idx)
            self.folds["test_ids"].append(test_idx)

        # Music model
        self.music_config_training = music_config_training
        self.music_config_inference = music_config_inference
        self.music_df = music_df

        self.logger.info(f"CV by graph initialized with {n_folds}-fold cross-validation.")

    def _music_training(self, fold_idx: int):
        test_idx = self.folds["test_ids"][fold_idx]

        # exclude test data from training of music model
        test_original_ids = self.df.iloc[test_idx].index
        fold_music_df = self.music_df[~self.music_df.index.isin(test_original_ids)]

        X, y_mid, y_emo, y_cls = load_embeddings_and_labels(
            fold_music_df, 
            self.music_config_training['mid_dict'],
            self.music_config_training['emo_dict'],
            self.music_config_training['cls_dict']
        )

        # Train music model
        model, f1s_val = train_music_model(self.music_config_training, X, y_mid, y_emo, y_cls, return_metrics=True)

        self.logger.info(f"Fold {fold_idx+1}/{len(self.folds['train_ids'])} music validation F1: {f1s_val['cls'][self.classes_key]:.2f}")
        # NB: the main task (definitions/main_task.json) should be the same for both self.music_df and self.df

        return model

    
    def __iter__(self):
        for fold_idx in range(len(self.folds["train_ids"])):
            train_idx = self.folds["train_ids"][fold_idx]
            test_idx = self.folds["test_ids"][fold_idx]
            self.logger.info(f"Processing fold {fold_idx+1}/{len(self.folds['train_ids'])}")

            # Train music model
            music_model = self._music_training(fold_idx)
            music_model_path = f"music_model_fold_{fold_idx+1}.pt"
            
            # Split data
            test_df = self.df.iloc[test_idx] # this is excluded from the RAG pool
            test_original_ids = test_df.index.tolist() # see "exclude_from_vector_store": test_original_ids
            train_df = self.df.iloc[train_idx] # this coincides with the resulting reduced RAG pool

            # precompute audio descriptions for the reduced RAG pool
            for idx in tqdm.tqdm(train_df.index, desc="Precomputing audio descriptions for RAG pool"):
                y_mid_label, y_emo_label, y_cls_label, y_logits = id_to_labels(music_model, self.music_config_inference, idx)
                mid_text, emo_text, cls_text = logits_to_text(y_logits[0], y_logits[1], y_logits[2], self.music_config_inference)

                self.workflow.auxiliary_data["music_description_rag_inputs"][idx] = {
                    "soundtrack_description": " ".join([mid_text, emo_text, cls_text])
                }

            # Run the graph on the test data
            predictions = []
            music_only_predictions = []
            actual = test_df[self.classes_key].values.tolist()
            for idx in tqdm.tqdm(test_original_ids, desc="Running graph on test data"):
                # compute audio description and target prediction for current datapoint
                y_mid_label, y_emo_label, y_cls_label, y_logits = id_to_labels(music_model, self.music_config_inference, idx)
                mid_text, emo_text, cls_text = logits_to_text(y_logits[0], y_logits[1], y_logits[2], self.music_config_inference)
                music_only_predictions.append(y_cls_label[self.classes_key]) # TODO: add more music metrics

                # pass music description and target prediction to the workflow
                self.workflow.auxiliary_data["music_only_predictions"][idx] = y_cls_label[self.classes_key]
                self.workflow.auxiliary_data["music_description_llm_inputs"][idx] = {
                    "soundtrack_description": " ".join([mid_text, emo_text, cls_text])
                }
                
                # run the graph
                for output in self.workflow.workflow.stream({"current_id": idx, "exclude_from_vector_store": test_original_ids}):
                    if "target_classification" in output:
                        predictions.append(output["target_classification"]["target_class"])

            music_model_path = os.path.join(self.workflow.o_folder, music_model_path)
            torch.save(music_model, music_model_path)

            # save fold results to disk
            folds_path = os.path.join(self.workflow.o_folder, "cv_folds.json")
            to_dump = {
                "music_model_paths": [],
                "test_original_ids": [],
                "actual": [],
                "predictions": [],
                "music_only_predictions": []
            }
            if os.path.exists(folds_path):
                with open(folds_path, "r") as f:
                    to_dump = json.load(f)

            self.logger.info(f"Saving folds to {folds_path}")
            to_dump["music_model_paths"].append(music_model_path)
            to_dump["test_original_ids"].append(test_original_ids)
            to_dump["actual"].append(actual)
            to_dump["predictions"].append(predictions)
            to_dump["music_only_predictions"].append(music_only_predictions)

            with open(folds_path, "w") as f:
                json.dump(to_dump, f)
            
            yield actual, predictions