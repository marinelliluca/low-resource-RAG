from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from typing import List, Union, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


##################
# Vector DB Class
##################

class VectorDB:
    def __init__(
        self,
        hf_model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        device: str = "cpu",
    ):
        """
        Args:
        -----
        hf_model_name : str
            The name of the HuggingFace model to use for embeddings.
        device : str
            The device to use for embeddings (default is "cpu").
        """

        self.hf_model_name = hf_model_name
        self.device = device
        self.chromadb = None

    def create_vector_database(
        self, 
        documents: List[Document], 
        ids: Union[List[Union[int,str]]]
    ):
        """
        Fill the vector store with the provided documents.

        Arguments
        =========
        documents: List of Document
            List of documents to be embedded and stored.
        ids: List of int | str
            List of ids to associate with the documents. 
            It is not an optional argument, as unique ids are required for each document.
        """
        model_kwargs = {"device": self.device}
        embedding_function = HuggingFaceEmbeddings(
            model_name=self.hf_model_name, model_kwargs=model_kwargs
        )
        self.chromadb = Chroma(
            collection_name="custom_collection",
            embedding_function=embedding_function,
        )

        self.chromadb.add_documents(documents=documents, ids=ids)
        logger.info(f"Vector store created with {len(documents)} documents")

    def query_vector_database(
        self, query: str, top_k: int = 5, filters: Union[dict, None] = None
    ) -> List[Document]:
        """
        Arguments
        =========
        query: str
            The query string.
        top_k: int
            The number of top similar documents to retrieve.
        filters: dict | None
            The dictionary containing the filters to the query
            (see https://docs.trychroma.com/docs/querying-collections/metadata-filtering)

        Returns
        ========
        List[Document]
            the list of retrieved documents
        """
        results = self.chromadb.similarity_search(
            query=query, k=top_k, filter=filters
        )

        # invert order (most relevant last)
        #results = results[::-1]

        return results

    def get_positive_negative_examples(
        self, 
        datapoint_id: str, 
        transcript: str, 
        metadata_field: str, 
        top_k: int,
        exclude_from_vector_store: List[str] = []
    ) -> Tuple[List[str], List[str]]:
        """
        Arguments
        =========
        datapoint_id: str
            The id of the current datapoint in the vectorstore and groundtruth.
        transcript: str
            The transcript text.
        metadata_field: str
            The metadata field to filter on.
        top_k: int
            The number of top similar documents to retrieve for positive and negative examples.
            (i.e., total examples will be 2 * top_k)
        exclude_from_vector_store: List[str]
            A list of document ids to exclude from the search (e.g., for doing a CV).

        Returns
        =========
        tuple
            A tuple containing two lists of strings: positive examples and negative examples.
        """
        positive_filter = {
            "$and": [
                {metadata_field: {"$ne": -1}}
            ]
        }

        negative_filter = {
            "$and": [
                {metadata_field: -1}
            ]
        }

        if not exclude_from_vector_store:
            exclude_from_vector_store.append(datapoint_id)

        for idx in exclude_from_vector_store:
            positive_filter["$and"].append({"id": {"$ne": idx}})
            negative_filter["$and"].append({"id": {"$ne": idx}})

        positive_examples = self.query_vector_database(
            transcript, top_k=top_k, filters=positive_filter
        )
        negative_examples = self.query_vector_database(
            transcript, top_k=top_k, filters=negative_filter
        )

        example_template = '"transcript": "{transcript}"\n"cues": {cues}'

        pos_ex = []
        for ex in positive_examples:
            ts = ex.page_content
            cues = ex.metadata[metadata_field].split(" ")
            cues = [cue for cue in cues if cue != ""] # remove empty strings
            example = example_template.format(
                **{"transcript": ts, "cues": cues}
            )
            pos_ex.append(example)

        neg_ex = []
        for ex in negative_examples:
            ts = ex.page_content
            example = example_template.format(
                **{"transcript": ts, "cues": "[]"}
            )
            neg_ex.append(example)

        return pos_ex, neg_ex 

    def make_target_filter(
        self, 
        classes_key: str, 
        datapoint_id: str, 
        target: str,
        exclude_from_vector_store: List[str] = []
    ) -> dict:

        """
        Arguments
        =========
        classes_key: str
            The key used to identify the main classification task in the metadata.
        datapoint_id: str
            The ID of the current document to be excluded from the results.
        target: str
            The target value to filter documents by.

        Returns
        ==========
        dict
            A dictionary representing the MongoDB filter.
        """

        target_filter = {
            "$and": [
                {classes_key: target}
            ]
        }

        if not exclude_from_vector_store:
            exclude_from_vector_store.append(datapoint_id)

        for idx in exclude_from_vector_store:
            target_filter["$and"].append({"id": {"$ne": idx}})
        
        return target_filter

    def get_targeted_examples(
        self,
        datapoint_id: str,
        transcript: str,
        classes: List[str],
        themes_definitions: dict,
        examples_per_class: int,
        classes_key: str,
        exclude_from_vector_store: List[str] = [],
        auxiliary_data: dict = {}
    ) -> List[str]:
        """
        Arguments
        =========
        datapoint_id : str
            The current identifier to exclude from the search.
        transcript : str
            The transcript to use as the query for similarity search.
        classes : List[str]
            The list of target classes to retrieve examples for.
        themes_definitions: dict
            A dictionary containing a number of pre-extracted classes as keys and their
            cue words as values (in the main example of this repo, these are the themes and their associated cue words)
        examples_per_class : int, optional
            The number of examples to retrieve per class (default is 5).
        classes_key : str, optional
            The key used to identify the main classification task in the metadata.

        Returns
        =======
        str
            A list of strings containing the targeted examples with their transcripts, themes, and class information.
        """

        examples_docs = [
            self.query_vector_database(
                query=transcript,
                top_k=examples_per_class,
                filters=self.make_target_filter(
                    classes_key, 
                    datapoint_id, 
                    target,
                    exclude_from_vector_store
                ),
            )
            for target in classes
        ]

        examples_docs = [ex for sublist in examples_docs for ex in sublist]

        example_template = '"transcript": "{transcript}"\n"themes": {themes}\n"{classes_key}": "{class}"'

        # if auxiliary data is provided, add it to the template 
        # done according to the corresponding prompt in target classification task
        # (if needed add keys to distinguish between tasks)
        #if auxiliary_data: 
        #    example_template = '"transcript": "{transcript}"\n"soundtrack_description": "{soundtrack_description}"\n"themes": {themes}\n"{classes_key}": "{class}"'

        examples_texts = []
        for ex in examples_docs:
            ex_transcript = ex.page_content
            ex_id = ex.metadata["id"]
            ex_themes = []
            for theme in themes_definitions.keys():
                if ex.metadata[theme + "_cues"] != -1:
                    ex_themes.append(theme)
            temp_inputs = {
                    "transcript": ex_transcript,
                    "themes": ex_themes,
                    "classes_key": classes_key,
                    "class": ex.metadata[classes_key],
            }
            #temp_inputs.update(auxiliary_data[ex_id] if ex_id in auxiliary_data else {})

            ex_text = example_template.format(
                **temp_inputs
            )
            examples_texts.append(ex_text)

        return examples_texts

    def get_transcript_by_id(self, datapoint_id: str) -> str:
        """
        Arguments
        =========
        datapoint_id : str
            The ID of the document to retrieve.

        Returns
        =======
        str
            The transcript text of the document.
        """
        return self.chromadb.get(where={"id": datapoint_id})["documents"][0]