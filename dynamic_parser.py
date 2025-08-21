import argparse
import json
from typing import Any, Dict

def gettype(name: str) -> Any:
    types = {"str": str, "int": int, "float": float, "bool": bool}
    if name in types:
        return types[name]
    raise ValueError(f"Error in definitions/parser.json, {name} is not a valid type.")

def parse_args() -> argparse.Namespace:
    
    """
    Returns:
    argparse.Namespace
        The parsed arguments.
    """
    
    with open("definitions/parser.json") as f: # parser definition
        parser_dict = json.load(f)

    # build the parser
    parser = argparse.ArgumentParser(description=parser_dict["description"])
    for argument in parser_dict["args"]:
        flags = argument["flags"]
        argument.pop("flags")
        if "type" in argument:
            type_obj = gettype(argument["type"])
            argument.pop("type")
            parser.add_argument(*flags, type=type_obj, **argument)
        else:
            parser.add_argument(*flags, **argument)

    #############################
    # TODO: add more frameworks #
    #############################

    # The Framework to use (Huggingface only supported for now)
    parser.add_argument(
        "--framework",
        type=str,
        choices=["huggingface"],
        default="huggingface",
        help="Name of the framework to use for LLMs. Huggingface only supported for now.",
    )

    args = parser.parse_args()

    # overwrite all the defaults with the base configuration file
    with open("config/base_parameters.json") as f:
        parameters = json.load(f)
    for key, value in parameters.items():
        # TODO: move all of the defaults to the base configuration file, 
        # so that we don't overwrite anything here (it's confusing)
        setattr(args, key, value)

    # load definitions of themes and main classification task
    with open(args.themes_definitions) as f:
        themes_definitions = json.load(f)

    with open(args.main_task) as f:
        main_task = json.load(f)

    # add themes definitions and main task to the args
    args.themes_definitions = themes_definitions
    args.main_task = main_task

    # delete superfluous arguments
    if not args.do_sample_cd:
        # delete the sampling arguments for theme detection
        for arg in ["temperature_cd", "top_k_cd", "top_p_cd"]:
            if hasattr(args, arg):
                delattr(args, arg)

    if not args.do_sample_tc:
        for arg in ["temperature_tc", "top_k_tc", "top_p_tc"]:
            if hasattr(args, arg):
                delattr(args, arg)

    # check that we have models names
    assert (
        args.model_name_cd is not None
    ), "If it's not in the configuration file, you need to pass a theme detection model name via command line!"
    assert (
        args.model_name_tc is not None
    ), "If it's not in the configuration file, you need to pass a theme detection model name via command line!"

    return args