import os
import json
from pprint import pprint
from dynamic_parser import parse_args
import argparse
from tools import load_data, convert_df_to_documents
from llm import Chain, get_huggingface_model
from vector_db import VectorDB
from workflow import WorkflowHandler
import numpy as np
import logging
import torch
import time
import yaml
import pandas as pd
import tqdm

# How to use: CUDA_VISIBLE_DEVICES=? python app_cv.py 
# (define parameters in config/base_parameters.json)

# Music training
from music.tools import (
    load_embeddings_and_labels, 
    train_music_model,
    id_to_labels
)

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # set up timing
    start_time = time.time()

    # enable flash-attention
    torch.backends.cuda.enable_flash_sdp(True)

    # parse arguments
    args = parse_args()

    if args.old_run_folder is not None:
        # KEEP some of the current args (change as needed for debugging)
        old_run = args.old_run_folder
        datapoint_id = args.datapoint_id
        
        # and keep all relevant for the new run (mainly target classification)
        model_name_tc = args.model_name_tc
        max_new_tokens_tc = args.max_new_tokens_tc
        examples_per_class_tc = args.examples_per_class_tc
        provide_theme_classes = args.provide_theme_classes
        do_sample_tc = args.do_sample_tc
        attn_implementation_tc = args.attn_implementation_tc
        recompute_this_theme = args.recompute_this_theme

        # replace the rest of the parsed configuration with the one from the old run
        # just to be sure that for the rest of the configuration we are using the same parameters as in the old run
        logger.info(f"Replacing the parsed configuration with the one from {args.old_run_folder}")
        with open(os.path.join(args.old_run_folder, "config.json"), "r") as f:
            args = json.load(f)
        args = argparse.Namespace(**args)

        # replace the kept arguments
        args.old_run_folder = old_run
        args.datapoint_id = datapoint_id
        args.model_name_tc = model_name_tc
        args.max_new_tokens_tc = max_new_tokens_tc
        args.examples_per_class_tc = examples_per_class_tc
        args.provide_theme_classes = provide_theme_classes
        args.do_sample_tc = do_sample_tc
        args.attn_implementation_tc = attn_implementation_tc
        args.recompute_this_theme = recompute_this_theme

    if args.frame != "huggingface":
        raise ValueError(f"Only \"huggingface\" is supported for now, but got {args.frame}")
    else:
        logger.info("Using Huggingface models")

        # login to huggingface (necessary for some of the models)
        from huggingface_hub.hf_api import HfFolder 

        fp = os.path.join("config", "huggingface.json") 
        if os.path.exists(fp):
            with open(fp) as f:
                token = json.load(f)["tkn"]
            HfFolder.save_token(token)
        else:
            # config/huggingface_token.json should be a json like this: {"tkn": "your_token_here"}
            logger.error(f"No huggingface token found. Please create a token and save it in {fp})")

    #########################################
    # Load data and create the vector store #
    #########################################

    rag_pool_df = load_data(args.groundtruth)

    vector_store = VectorDB(args.embedding_model_name, args.device)

    docs, ids = convert_df_to_documents(
        rag_pool_df, args.text_column, args.themes_definitions,
    )

    vector_store.create_vector_database(docs, ids)

    # load raw data to be processed by the workflow
    unseen_datapoints_df = pd.read_csv(args.unseen_data, index_col=0)

    ##########################
    # Build graph components #
    ##########################

    # check if we can save resources
    skip_cues_model = False
    if args.old_run_folder is not None and args.recompute_this_theme is None:
        logger.info("ATTENTION: Reusing precomputed states")
        logger.info(f"Old run folder: {args.old_run_folder}")

        # check if the old run folder exists
        if not os.path.exists(args.old_run_folder):
            raise FileNotFoundError(f"Old run folder does not exist: {args.old_run_folder}")
        
        if args.recompute_this_theme is None:
            logger.info("... checking if we can free-up resources ...")

            # check if the old run contains the state for all of the datapoints 
            # for any of the following steps (the tasks defined in the folder prompts/<task_name>)
            
            # TODO: make everything even more modular 
            # so that once defined a task in the prompts folder everything else is done automatically
            # tasks = [os.path.split(f.path)[1] for f in os.scandir("prompts") if f.is_dir()]

            tasks = ["cues_detection"] #"cues_correction",] # "target_classification"]
            
            hook = WorkflowHandler(
                config=vars(args),
                vector_store=vector_store,
                cues_detector=None,
                cues_corrector=None,
                target_classifier=None,
            )

            we_can_skip_these = []
            for task in tasks:
                for idx in unseen_datapoints_df.index:
                    precomp_state = hook._load_precomputed_state(task, {"current_id": idx})
                    if precomp_state is None:
                        logger.info(f"State for {idx} in {task} was either not found or has different parameters in the configuration than the current run.")
                        break
                else: # A For Else appeared!
                    logger.info(f"Reusing all of the states for {task}")
                    we_can_skip_these.append(task)

            if "cues_detection" in we_can_skip_these:# and "cues_correction" in we_can_skip_these:
                logger.info("We therefore don't need to load the cues model at all")
                skip_cues_model = True

    # load cues detection (and checker) models
    if not skip_cues_model:
        model_cd, tokenizer_cd, assistant_token_cd = get_huggingface_model(
            model_name=args.model_name_cd,
            device=args.device,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation_cd,
            four_bit=args.four_bit_cd,
        )

        # initialize and build Chains
        chain_kwargs = {
            "frame": args.framework,
            "max_new_tokens": args.max_new_tokens_cd,
            "do_sample": args.do_sample_cd,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": args.attn_implementation_cd,
            "four_bit": args.four_bit_cd,
            "model_name": args.model_name_cd,
            "model": model_cd,  # NB: not passing the model name here, but the model itself
            "tokenizer": tokenizer_cd,  # and the tokenizer
            "assistant_token": assistant_token_cd,
        }

        if args.do_sample_cd:
            chain_kwargs["temperature"] = args.temperature_cd
            chain_kwargs["top_k"] = args.top_k_cd

        cues_detector = Chain(
            **chain_kwargs
        )

        cues_checker = Chain( # reusing the same model and tokenizer
            **chain_kwargs
        )
    else:
        model_cd, tokenizer_cd, assistant_token_cd = None, None, None
        cues_detector, cues_checker = None, None

    # load gender target detection model
    if args.model_name_cd == args.model_name_tc and model_cd is not None:
        model_tc, tokenizer_tc, assistant_token_tc = model_cd, tokenizer_cd, assistant_token_cd
    else:
        model_tc, tokenizer_tc, assistant_token_tc = get_huggingface_model(
            model_name=args.model_name_tc,
            device=args.device,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation_tc,
            four_bit=args.four_bit_tc,
        )

    chain_kwargs = {
        "frame": args.framework,
        "max_new_tokens": args.max_new_tokens_tc,
        "do_sample": args.do_sample_tc,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation_tc,
        "four_bit": args.four_bit_tc,
        "model_name": args.model_name_tc,
        "model": model_tc,  # NB: not passing the model name here, but the model itself
        "tokenizer": tokenizer_tc,  # and the tokenizer
        "assistant_token": assistant_token_tc,
    }

    if args.do_sample_tc:
        chain_kwargs["temperature"] = args.temperature_tc
        chain_kwargs["top_k"] = args.top_k_tc

    gender_target_classifier = Chain(
        **chain_kwargs
    )

    ############################
    # Build and run the graph  #
    ############################

    # build the workflow
    workflow = WorkflowHandler(
        config=vars(args),  # pass all arguments as a dictionary
        vector_store=vector_store,
        cues_detector=cues_detector,
        cues_corrector=cues_checker,
        target_classifier=gender_target_classifier,
    )

    # log start-up time (which includes building the graph)
    start_up_time = time.time() - start_time
    logger.info(f"Start-up time: {start_up_time:.2f} seconds")

    # Train music model
    music_df = pd.read_csv('music/music_groundtruth.csv', index_col="stimulus_id")
    with open('music/config_training.yaml', 'r') as f:
        music_config_training = yaml.safe_load(f)
    #with open('music/config_inference.yaml', 'r') as f:
    #    music_config_inference = yaml.safe_load(f)
    X, y_mid, y_emo, y_cls = load_embeddings_and_labels(
        music_df,
        music_config_training['mid_dict'],
        music_config_training['emo_dict'],
        music_config_training['cls_dict']
    )
    music_model, f1s_val = train_music_model(music_config_training, X, y_mid, y_emo, y_cls, return_metrics=True)
    logger.info(f"Music model validation F1: {f1s_val['cls'][args.main_task['classes_key']]:.2f}")

    # run the graph on all unseen commercials
    for idx in tqdm.tqdm(
        unseen_datapoints_df.index, 
        desc="Run the graph on all unseen datapoints"
    ):    
        # compute audio description for current datapoint
        y_mid_label, y_emo_label, y_cls_label, y_logits = id_to_labels(
            music_model, 
            music_config_training, # music_config_inference,
            idx,
            embeddings_dir=args.soundtrack_embeddings_dir,
        )

        # save audio description to file
        os.makedirs(workflow.o_folder, exist_ok=True)
        with open(os.path.join(workflow.o_folder, f"{idx}_music_pred.json"), "w") as f:
            json.dump({"mid": y_mid_label, "emo": y_emo_label, "cls": y_cls_label}, f)

        # pass music target prediction to the workflow
        workflow.auxiliary_data["music_only_predictions"][idx] = y_cls_label[args.main_task['classes_key']]

        # run the graph on the current datapoint
        #logger.info(f"Running the graph on {idx}")  
        initial_state = {
            "current_id": idx,
            "current_transcript": unseen_datapoints_df.loc[idx, args.text_column],
            "exclude_from_vector_store": ["placeholder"],  # ignore, this is used only for cross-validation
        }
        for output in workflow.workflow.stream(initial_state):
            # output is a dictionary with keys as node names and values as node outputs

            pass # everything is saved in the workflow's output folder (i.e., results/{timestamp})

            # Uncomment the following lines to print the output of each node
            """
            for key, value in output.items():
                # Node
                pprint(f"Node '{key}':")
                pprint(value, indent=2, width=80, depth=None)
            print("\n------\n")
            """

        #logger.info(f"The results for {idx} were saved in {workflow.o_folder}")
    