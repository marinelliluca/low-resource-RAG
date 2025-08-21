from typing import List, Union
from typing_extensions import TypedDict

class ThemeState(TypedDict):
    """
    Represents the state of the collected cues for a single theme.
    """
    cues: List[str]


class CollectedCues(TypedDict):
    """
    Represents the state of the collected cues for all themes.

    The keys of this typed dictionary are the themes defined at definition_path.

    """
    def __init__(self, definition_path="definitions/themes.json"):
        with open(definition_path) as f:
            definition = json.load(f)        

        for key in definition.keys():
            setattr(self, key, ThemeState)

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    This graph is built for leave-one-out cross-validation, where we need to keep track of the current datapoint.
    In "deployment" mode we will not need 'current_id'.

    Attributes:
    - current_id: the id of the current datapoint
    - current_transcript: the transcript of the current datapoint
    - collected_cues: The cues that have been collected for each theme.
    - target_class: The target class (main task prediction) of the current datapoint.
    """

    current_id: str
    current_transcript: str
    collected_cues: CollectedCues
    target_class: str
    exclude_from_vector_store: List[str]