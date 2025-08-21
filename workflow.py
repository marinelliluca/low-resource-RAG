from langgraph.graph import END, StateGraph, START
from states import GraphState, CollectedCues, ThemeState
from vector_db import VectorDB
from llm import Chain
from tools import get_prompt
import logging
import time
import json
import os
from datetime import datetime
from itertools import combinations
from collections import Counter
from typing import Union

import gc 
import torch

# helpers

def flush():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()


def decision_logic(tasks_results):
    results = [tasks_results[task]["result"] for task in tasks_results.keys()]

    results = [x for x in results if isinstance(x, str)]
    results = [x for x in results if x != "placeholder"]

    if not results:
        return "placeholder"

    counts = {x: results.count(x) for x in set(results)}
    
    max_count = max(counts.values())
    winners = [k for k, v in counts.items() if v == max_count]

    if len(winners) == 1:
        result = winners[0]
    else: # len(winners) > 1:
        # if there's a tie, return the result of the task containing the winners
        tie_task = [
            task for task in tasks_results.keys() 
            if set(tasks_results[task]["classes"]) == set(winners)
        ]
        if len(tie_task) == 1:
            result = tasks_results[tie_task[0]]["result"]
        else:
            # this should never happen, but just in case
            result = max(set(results), key=results.count)
    
    if "boys" in result.lower():
        return "Boys/men"
    if "girls" in result.lower():
        return "Girls/women"
    if "mixed" in result.lower():
        return "Mixed"
    else:
        return "placeholder"


class WorkflowHandler:
    def __init__(
        self, 
        config: dict, 
        vector_store: VectorDB, 
        cues_detector: Union[Chain, None], 
        cues_corrector: Union[Chain, None],
        target_classifier: Union[Chain, None],    
    ):
        self.config = config
        self.vector_store = vector_store
        self.cues_detector = cues_detector
        self.cues_corrector = cues_corrector
        self.target_classifier = target_classifier
        self.workflow = self._compile_workflow()
        self.logger = logging.getLogger(__name__)

        # see how it's used in cv_by_graph.py and here within _classify_target
        self.auxiliary_data = {
            "music_description_rag_inputs": {},
            "music_description_llm_inputs": {},
            "music_only_predictions": {},
        }
        
        #self.o_folder = f"results/{datetime.now().strftime('%Y-%m-%d_%H:%M')}"
        self.o_folder = os.path.join(
            self.config["output_folder"], 
            datetime.now().strftime("%Y-%m-%d_%H:%M")
        )
        
    def _initialize_datapoint(
        self,
        state: dict,
    ) -> GraphState:
        """
        Initializes the graph with an emtpy state.

        Returns:
        - A GraphState with the current_id, current_transcript and empty collected_cues
        """

        empty_cues = {
            k: ThemeState(**{"cues": []}) 
            for k in self.config["themes_definitions"].keys()
        }

        initial_state = {
            "current_id": state["current_id"],
            "current_transcript": state["current_transcript"], #self.vector_store.get_transcript_by_id(state["current_id"]),
            "collected_cues": CollectedCues(**empty_cues),
        }

        return GraphState(**initial_state)

    def _detect_cues(
        self,
        state: GraphState
    ) -> GraphState:
        """
        Detects all cues for each theme for the current datapoint.
        """

        current_task = "cues_detection"

        prompt_template, placeholder = get_prompt(current_task)

        themes_to_compute = self.config["themes_definitions"].keys()
        
        if self.config["old_run_folder"] is not None:
            precomp_state = self._load_precomputed_state(
                current_task, 
                state, 
            )
            if precomp_state is not None:
                state = precomp_state

                if self.config["recompute_this_theme"] is not None:
                    themes_to_compute = [self.config["recompute_this_theme"]]
                else:
                    themes_to_compute = [] # skip entirely the detection step

        start_time = time.time()

        for current_theme in themes_to_compute:
            pos_ex, neg_ex = self.vector_store.get_positive_negative_examples(
                datapoint_id=state["current_id"], 
                transcript=state["current_transcript"], 
                metadata_field=current_theme + "_cues", 
                top_k=self.config["pos_neg_examples_cd"], 
                exclude_from_vector_store=state["exclude_from_vector_store"]
            )
            pos_ex = "\n\n".join(pos_ex) + "\n\n"
            neg_ex = "\n\n".join(neg_ex) + "\n\n"

            output_cues = self.cues_detector.run(
                inputs={
                    "current_theme": current_theme.replace("_", " and "),
                    "current_theme_definition": self.config["themes_definitions"][current_theme],
                    "positive_examples": pos_ex,  
                    "negative_examples": neg_ex,
                    "current_transcript": state["current_transcript"],
                },
                prompt_template=prompt_template,
                placeholder=placeholder,
            )

            # keep only those that are actually in the transcript
            collected_cues = [cue.replace("_", " ") for cue in output_cues["cues"]]
            collected_cues = [
                cue for cue in collected_cues if cue.lower() in state["current_transcript"].lower()
            ]

            state["collected_cues"][current_theme]["cues"] = collected_cues

        # flush memory
        # flush() # doesn't make sense here as the cues detector is the same as the cues corrector

        # log cues detection time
        if themes_to_compute:
            self.logger.info(f"cues detection time: {time.time() - start_time:.2f} seconds")

        # dump state in a timestamped folder
        self._dump_state(state, current_task)

        return GraphState(**state)

    def _correct_cues(
        self, 
        state: GraphState
    ) -> GraphState:
        """
        Corrects all cues found in the previous step. If there are no cues to correct it passes the state to the next step.

        This bypass is done because the model tends to generate false positives more often than false negatives.

        In addition, it saves time and resources.
        """

        current_task = "cues_correction"

        prompt_template, placeholder = get_prompt(current_task)

        themes_to_compute = self.config["themes_definitions"].keys()

        if self.config["old_run_folder"] is not None:
            precomp_state = self._load_precomputed_state(
                current_task, 
                state, 
            )
            if precomp_state is not None:
                state = precomp_state

                if self.config["recompute_this_theme"] is not None:
                    themes_to_compute = [self.config["recompute_this_theme"]]
                else:
                    themes_to_compute = [] # skip entirely the correction step

        start_time = time.time()

        for current_theme in themes_to_compute:

            if state["collected_cues"][current_theme]["cues"]: # if there are cues to correct
                pos_ex, neg_ex = self.vector_store.get_positive_negative_examples(
                    datapoint_id=state["current_id"],
                    transcript=state["current_transcript"],
                    metadata_field=current_theme + "_cues",
                    top_k=self.config["pos_neg_examples_cd"],
                    exclude_from_vector_store=state["exclude_from_vector_store"],
                )
                pos_ex = "\n\n".join(pos_ex) + "\n\n"
                neg_ex = "\n\n".join(neg_ex) + "\n\n"

                output_check = self.cues_corrector.run(
                    inputs={
                        "current_theme": current_theme.replace("_", " and "),
                        "current_theme_definition": self.config["themes_definitions"][current_theme],
                        "positive_examples": pos_ex,
                        "negative_examples": neg_ex,
                        "current_transcript": state["current_transcript"],
                        "cues_old": state["collected_cues"][current_theme]["cues"],
                    },
                    prompt_template=prompt_template,
                    placeholder=placeholder,
                )

                cues_corrected = output_check["cues_corrected"]
                
                #self.logger.info(str(cues_corrected))

                # keep only those that are actually in the transcript
                cues_corrected = [cue.replace("_", " ") for cue in cues_corrected]
                cues_corrected = [
                    cue for cue in cues_corrected if cue.lower() in state["current_transcript"].lower()
                ]

                state["collected_cues"][current_theme]["cues"] = cues_corrected
            else:
                pass

        # flush memory
        if self.cues_corrector is not None or self.cues_detector is not None:
            # TODO: self.cues_corrector.chat_model # can we momentarily move this to cpu? 
            # to then move it back to gpu at the beginning of the function?
            # unfortunately, langchain's documentation is quite obscure in this regard
            # see langchain source code for ChatHuggingFace class and HuggingFacePipeline 
            # (called in this repo in llm.py at lines 197-99)
            flush()

        # log cues correction time
        if themes_to_compute:
            self.logger.info(f"cues correction time: {time.time() - start_time:.2f} seconds")

        # dump state in a timestamped folder
        self._dump_state(state, current_task)

        return GraphState(**state)
    def __subtask_wrapper(
        self,
        state,
        subset_classes, 
        reasoning_structure_dict, 
        current_themes_definitions, 
        classes_key,
        collected_themes,
        prompt_template,
        placeholder,
    ):
        """
        Helper function to perform a task based on a subset of the main classes.
        (only for target classification)
        """

        current_task = "target_classification"
        
        # we remove any mention of the classes that are not in the given subset 
        # so the model doesn't get confused
        cls_to_remove = [cls_ for cls_ in self.config["main_task"]["classes"] if cls_ not in subset_classes]

        reasoning_structure = [v for k, v in reasoning_structure_dict.items() if k in subset_classes]
        reasoning_structure = "\n".join(reasoning_structure)
        
        if "music_description_rag_inputs" in self.auxiliary_data:
            rag_auxiliary_data = self.auxiliary_data["music_description_rag_inputs"]
        else:
            rag_auxiliary_data = {}

        examples_text = self.vector_store.get_targeted_examples(
            datapoint_id=state["current_id"],
            transcript=state["current_transcript"],
            classes=subset_classes,
            themes_definitions=self.config["themes_definitions"],
            examples_per_class=self.config["examples_per_class_tc"],
            classes_key=classes_key,
            exclude_from_vector_store=state["exclude_from_vector_store"],
            auxiliary_data=rag_auxiliary_data,
        )
        examples_by_class = {cls_: [] for cls_ in subset_classes}
        for cls_ in subset_classes:
            for ex in examples_text:
                if f'"{classes_key}": "{cls_}"' in ex:
                    examples_by_class[cls_].append(ex)
        examples_text = ""
        for cls_, exs in examples_by_class.items():
            examples_text += f'## examples of "{classes_key}": "{cls_}"\n\n' + "\n\n".join(exs) + "\n\n"


        # remove any mention of the cls_to_remove
        for cls_ in cls_to_remove:
            current_themes_definitions = current_themes_definitions.replace(cls_, "placeholder")
            examples_text = examples_text.replace(cls_, "placeholder")

        llm_inputs = {
            "classes_key": classes_key,
            "number_of_classes": len(subset_classes),
            "classes": subset_classes,
            "reasoning_structure": reasoning_structure,
            "examples": examples_text,
            "current_transcript": state["current_transcript"],
            "current_themes": collected_themes,
            "current_themes_definitions": current_themes_definitions,
        }


        if state["current_id"] in self.auxiliary_data["music_description_llm_inputs"]:
            llm_inputs.update(self.auxiliary_data["music_description_llm_inputs"][state["current_id"]])

        output_target = self.target_classifier.run(
            inputs=llm_inputs,
            prompt_template=prompt_template,
            placeholder=placeholder,
        )

        return output_target
    def _classify_target(
        self, 
        state: GraphState
    ) -> GraphState:
        """
        Predicts the target class for the current datapoint.
        """

        current_task = "target_classification"

        prompt_template, placeholder, reasoning_structure_dict = get_prompt(current_task)

        classes_key = self.config["main_task"]["classes_key"]

        # get all themes that have any collected cues
        collected_themes = []
        current_themes_definitions = []
        for theme in self.config["themes_definitions"].keys():
            if len(state["collected_cues"][theme]["cues"]) > 0:
                collected_themes.append(theme)
                definition = self.config["themes_definitions"][theme]
                current_themes_definitions.append(f'\t"{theme}" --- {definition}')

        current_themes_definitions = '\n'.join(current_themes_definitions)

        precomp_state = None   
        if self.config["old_run_folder"] is not None:
            precomp_state = self._load_precomputed_state(current_task, state)
            
        if precomp_state is not None:
            state = precomp_state
        else:
            start_time = time.time()

            # define subtasks
            tasks_results = {
                "G/M":{
                    "classes": ['Girls/women', 'Mixed'], # binary classification
                    "result": None,
                },
                "G/B":{
                    "classes": ['Girls/women', 'Boys/men'],
                    "result": None,
                },
                "B/M":{
                    "classes": ['Boys/men', 'Mixed'],
                    "result": None,
                },
                "G/M/B":{
                    "classes": ['Girls/women', 'Mixed', 'Boys/men'],
                    "result": None,
                },
            }


            for task in ["G/B", "G/M/B", "G/M", "B/M"]:
                final_class = self.__subtask_wrapper(
                    state,
                    tasks_results[task]["classes"], 
                    reasoning_structure_dict, 
                    current_themes_definitions, 
                    classes_key,
                    collected_themes,
                    prompt_template,
                    placeholder,
                )
                tasks_results[task]["result"] = final_class[classes_key]

            # add auxiliary classification to the tasks_results
            if self.auxiliary_data["music_only_predictions"]:
                # NB: the music_only_predictions contain the target label, as defined in music/config_inference.yaml
                tasks_results["music"] = {
                    "result": self.auxiliary_data["music_only_predictions"][state["current_id"]],
                    "classes": [self.auxiliary_data["music_only_predictions"][state["current_id"]]],
                }

            state["target_class"] = decision_logic(tasks_results)
            
            # flush memory
            if self.cues_corrector is not None or self.cues_detector is not None:
                flush()

            # log target classification time
            self.logger.info(f"Target classification time: {time.time() - start_time:.2f} seconds")

            # dump tasks_results
            with open(os.path.join(self.o_folder, f"{state['current_id']}_tasks_results.json"), "w") as f:
                json.dump(tasks_results, f)

        # dump state in a timestamped folder 
        self._dump_state(state, "target_classification")

        return GraphState(**state)

    def _dump_state(
        self,
        state: GraphState,
        task: str
    ):
        os.makedirs(self.o_folder, exist_ok=True)
        with open(os.path.join(self.o_folder, f"{state['current_id']}_{task}.json"), "w") as f:
            json.dump(state, f)

        # dump config with the first state
        config_path = os.path.join(self.o_folder, "config.json")
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(self.config, f)

    def _load_precomputed_state(
        self, 
        task: str, 
        state: GraphState,
    ) -> dict:
        """
        Loads a precalculated state from a previous run after checking relevant config parameters
        
        This is only used in cues detection and cues correction 
        (as target classification is currently the last step)

        Use the --old_run_folder flag to specify the folder with the precomputed states

        <<USE WITH CAUTION>> 
        Only use if you haven't changed any of the prompts' files for each of the loaded tasks
        """

        state_path = os.path.join(self.config["old_run_folder"],f"{state['current_id']}_{task}.json")

        # check if the file exists
        if not os.path.exists(state_path):
            return None

        # load config from the old run
        with open(os.path.join(self.config["old_run_folder"], "config.json"), "r") as f:
            old_config = json.load(f)

        # select parameters to check in the configs
        check_only_these = {
            "all": [
                "embedding_model_name", 'text_column', 'groundtruth',
                'torch_dtype', 'framework', "cv_n_folds", "cv_seed",
            ],
            "cues_detection": [
                'themes_definitions', 'pos_neg_examples_cd',
                'model_name_cd', 'max_new_tokens_cd', 'do_sample_cd',
                'attn_implementation_cd', 'four_bit_cd', 
            ],
            "target_classification": [
                'examples_per_class_tc', 'model_name_tc', 'max_new_tokens_tc',
                'do_sample_tc', 'attn_implementation_tc', 'four_bit_tc', 
                'main_task', 
            ],
        }
        if self.config["do_sample_cd"]:
            check_only_these["cues_detection"].extend(["temperature_cd", "top_k_cd"])
        if self.config["do_sample_tc"]:
            check_only_these["target_classification"].extend(["temperature_tc", "top_k_tc"])
        if self.config["recompute_this_theme"] is not None:
            check_only_these["cues_detection"].remove("themes_definitions")
        
        check_only_these["cues_correction"] = check_only_these["cues_detection"]

        # check if the parameters are the same
        for param in check_only_these["all"]+check_only_these[task]:
            if self.config[param] != old_config[param]:
                return None

        with open(state_path, "r") as f:
            state = json.load(f)
        return state

    def _compile_workflow(self) -> StateGraph:
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("start", self._initialize_datapoint)
        workflow.add_node("cues_detection", self._detect_cues)
        #workflow.add_node("cues_correction", self._correct_cues)
        workflow.add_node("target_classification", self._classify_target)

        # Add edges  
        workflow.add_edge(START, "start")
        workflow.add_edge("start", "cues_detection")
        workflow.add_edge("cues_detection", "target_classification")
        #workflow.add_edge("cues_correction", "target_classification")
        workflow.add_edge("target_classification", END)

        return workflow.compile()


