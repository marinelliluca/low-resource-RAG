import os
import argparse
import json
from pprint import pprint
from dynamic_parser import parse_args
from tools import load_data, convert_df_to_documents, get_top_classes_per_theme
from llm import Chain, get_huggingface_model
from vector_db import VectorDB
from workflow import WorkflowHandler
from cv_by_graph import CVbyGraph
import numpy as np
import logging
import torch
import time
from datetime import datetime
import numpy as np
import yaml
import pandas as pd
import tqdm

# Music training
from music.tools import (
    load_embeddings_and_labels, 
    train_music_model,
    id_to_labels,
    logits_to_probs,
    logits_to_text
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
            # huggingface_token.json should be {"tkn": "your_token_here"}
            logger.error(f"No huggingface token found. Please create a token and save it in {fp})")

    #########################################
    # Load data and create the vector store #
    #########################################

    groundtruth_df = load_data(args.groundtruth)

    vector_store = VectorDB(args.embedding_model_name, args.device)

    docs, ids = convert_df_to_documents(
        groundtruth_df, args.text_column, args.themes_definitions,
    )

    vector_store.create_vector_database(docs, ids)

    # count frequency of themes per class (used during target classification)
    predominant_classes = get_top_classes_per_theme(
        groundtruth_df, 
        args.main_task["classes_key"],
        args.themes_definitions
    )

    # TODO: move this to a function in tools.py
    if args.provide_theme_classes:
        with open (os.path.join("definitions", "themes_to_classes.json"), "r") as f:
            themes_to_classes = json.load(f)
        
        for theme in themes_to_classes.keys():
            if themes_to_classes[theme] is not None:
                themes_to_classes[theme] = themes_to_classes[theme].format(
                    theme=theme,
                    theme_long=theme.replace('_', ' and ')
                )
        
        for theme in predominant_classes.keys():
            if themes_to_classes[theme] is None:
                if len(predominant_classes[theme]) == 0:
                    themes_to_classes[theme] = f"The theme \"{theme}\", {theme.replace('_', ' and ')}, has no predominant classes."
                elif len(predominant_classes[theme]) == 1:
                    themes_to_classes[theme] = f"The theme \"{theme}\", {theme.replace('_', ' and ')}, appears MOSTLY in ads for \"{predominant_classes[theme][0]}\""
                elif len(predominant_classes[theme]) == 2:
                    themes_to_classes[theme] = f"The theme \"{theme}\", {theme.replace('_', ' and ')}, appears often in ads for multiple audiences, but predominantly for \"{predominant_classes[theme][0]}\" \"{predominant_classes[theme][1]}\""
                    #themes_to_classes[theme] = f"The theme \"{theme}\", {theme.replace('_', ' and ')}, has no predominant classes."
                #else:
                #    not_this_class = [x for x in groundtruth_df.target_of_toy_ad.unique() if x not in predominant_classes[theme]]
                #    themes_to_classes[theme] = f"\"{theme.replace('_', ' and ')}\" appears very RARELY in ads for {not_this_class[0]}"
    else:
        themes_to_classes = {} # must be a dictionary
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

            tasks = ["cues_detection", "cues_correction",] # "target_classification"]
            
            hook = WorkflowHandler(
                config=vars(args),
                vector_store=vector_store,
                cues_detector=None,
                cues_corrector=None,
                target_classifier=None,
                themes_to_classes={},
            )

            we_can_skip_these = []
            for task in tasks:
                for idx in groundtruth_df.index:
                    precomp_state = hook._load_precomputed_state(task, {"current_id": idx})
                    if precomp_state is None:
                        logger.info(f"State for {idx} in {task} was either not found or has different parameters in the configuration than the current run.")
                        break
                else: # A For Else appeared!
                    logger.info(f"Reusing all of the states for {task}")
                    we_can_skip_these.append(task)

            if "cues_detection" in we_can_skip_these and "cues_correction" in we_can_skip_these:
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
        themes_to_classes=themes_to_classes,
    )

    # log start-up time (which includes building the graph)
    start_up_time = time.time() - start_time
    logger.info(f"Start-up time: {start_up_time:.2f} seconds")

    # load music data
    music_df = pd.read_csv('music/music_groundtruth.csv', index_col="stimulus_id")

    with open('music/config_training.yaml', 'r') as f:
        music_config_training = yaml.safe_load(f)

    with open('music/config_inference.yaml', 'r') as f:
        music_config_inference = yaml.safe_load(f)

    # run the graph
    if args.datapoint_id is not None:
        logger.info("Running the graph on {args.datapoint_id}")   

        X, y_mid, y_emo, y_cls = load_embeddings_and_labels(
            music_df[~music_df.index.isin([args.datapoint_id])], # exclude datapoint from training
            music_config_training['mid_dict'],
            music_config_training['emo_dict'],
            music_config_training['cls_dict']
        )

        # Train music model
        music_model, f1s_val = train_music_model(music_config_training, X, y_mid, y_emo, y_cls, return_metrics=True)
        logger.info(f"Music model validation F1: {f1s_val['cls'][args.main_task['classes_key']]:.2f}")

        # precompute audio descriptions for the RAG pool
        for idx in tqdm.tqdm(
            groundtruth_df[~groundtruth_df.index.isin([args.datapoint_id])].index, 
            desc="Precomputing audio descriptions for RAG pool"
        ):
            y_mid_label, y_emo_label, y_cls_label, y_logits = id_to_labels(music_model, music_config_inference, idx)
            mid_text, emo_text, cls_text = logits_to_text(y_logits[0], y_logits[1], y_logits[2], music_config_inference)

            workflow.auxiliary_data["music_description_rag_inputs"][idx] = {
                "soundtrack_description": " ".join([mid_text, emo_text, cls_text])
            }
        
        # compute audio description for current datapoint
        y_mid_label, y_emo_label, y_cls_label, y_logits = id_to_labels(music_model, music_config_inference, args.datapoint_id)
        mid_text, emo_text, cls_text = logits_to_text(y_logits[0], y_logits[1], y_logits[2], music_config_inference)
        workflow.auxiliary_data["music_description_llm_inputs"][args.datapoint_id] = {
            "soundtrack_description": " ".join([mid_text, emo_text, cls_text])
        }

        for output in workflow.workflow.stream({"current_id": args.datapoint_id}):
            for key, value in output.items():
                # Node
                pprint(f"Node '{key}':")
                pprint(value, indent=2, width=80, depth=None)
            print("\n------\n")

        logger.info(f"The results for {args.datapoint_id} were saved in {workflow.o_folder}")
    else:
        ######
        # TODO: here we can define several CVs (with different compiled graphs) so to collect different metrics:
        # - metrics for only text 
        # - metrics for only audio (with new classifications only multitask rather than incl. regression)
        # - metrics for text and audio combined
        #     - via the llm
        #     - via the ensemble (XGBoost of the final state of the workflow + output of audio model)
        #########

        # Run cross-validation (the default in definitions/parser.json is 5-fold)
        cv_one = CVbyGraph(
            df=groundtruth_df, 
            workflow=workflow, 
            classes_key=args.main_task["classes_key"],
            music_config_training=music_config_training,
            music_config_inference=music_config_inference,
            music_df=music_df,
            n_folds=args.cv_n_folds,
            shuffle=True,
            random_state=args.cv_seed,
        )

        # Run cross-validation
        for y_true_one, y_pred_one in cv_one:
            # TODO: get all of the other predictions 
            # - final prediction, themes, cues, music stuff etc.
            # - make a pandas dataframe with all of the predictions
            # - outside of this cript train an XGBoost model on that dataframe
            #   - one INCLUDING the final task of the workflow
            #   - one EXCLUDING the final task of the workflow
            #   - (to see if the final task of the workflow is useful Vs just using the rest in XGBoost)
            pass


    
