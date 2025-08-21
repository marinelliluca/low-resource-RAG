from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Union

###################
# Parsing utilities
###################

import regex as re

# find all the apostrophes in the text
pattern = re.compile(r"[A-Za-z]'[ A-Za-z]")
def find_apostrophes(text):
  return re.findall(pattern, text)

def remove_single_quotes(text):
    """
    Remove all the single quotes from text 
    (helps preventing JSON parsing errors)
    """

    # find all the apostrophes in the text and replace them with @&%€ (to avoid conflicts)
    for match in find_apostrophes(text):
        text = text.replace(match, match.replace("'", "@&%€"))
    
    # replace all single quotes with double quotes
    text = text.replace("'", '"')

    # put the apostrophes back in place
    return text.replace("@&%€", "'")

# In case of parsing errors, this prompt will be used to reformat the output of the LLM
formatting_prompt ="""
{output_format}

---

# PREVIOUS OUTPUT
{correct_this}

---

# CURRENT INSTRUCTIONS
If you are seeing this message, it means that the output of the LLM was not correctly formatted.
Please correct the last part of "# PREVIOUS OUTPUT" as specified above. 

That is, the JSON content from the "# PREVIOUS OUTPUT" should be re-formatted as follows:

<insert_three_bacticks>json
{placeholder}
<insert_three_bacticks>

NB: replace the "placeholder" with the JSON content from above, and replace <insert_three_bacticks> with three (`) characters.
"""

def correct_output(chat_model, output_format, correct_this, placeholder):
    """
    Correct the output of the LLM
    """ 
    # remove all the single quotes but leave the apostrophes
    correct_this = remove_single_quotes(correct_this)

    tmp_prompt = formatting_prompt.format(
        output_format=output_format, 
        correct_this=correct_this,
        placeholder=placeholder
    )

    tmp_prompt = remove_single_quotes(tmp_prompt)

    output = chat_model.invoke(tmp_prompt)

    return output

##########################
# Huggingface models
##########################


# get the huggingface model
def get_huggingface_model(
    model_name: str,
    device: str = "cuda",
    torch_dtype: str = "auto",
    attn_implementation: str = "sdpa",
    four_bit: bool = False,
) -> tuple:
    """
    Arguments
    =========
    model_name : str
        The name of the model to load from Hugging Face.
    device : str, optional
        The device to load the model on, default is "cuda".
    torch_dtype : str, optional
        The data type for the model's tensors, default is "auto".
    attn_implementation : str, optional
        The attention implementation to use, default is "sdpa".
    four_bit : bool, optional
        Whether to load the model in 4-bit precision, default is False.
    Returns
    =======
    tuple
        A tuple containing the loaded model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        load_in_4bit=four_bit,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assistant_token = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "skip_me"},
            {"role": "assistant", "content": "ignore_me"},
        ], 
    tokenize=False)
    assistant_token = assistant_token.split("ignore_me")[0]
    assistant_token = assistant_token.split("skip_me")[1]
    # not really the assistant token,
    # but the series of tokens that are used to separate the assistant's output from the user's input
    # this is used to reformat the output in case of parsing errors

    return model, tokenizer, assistant_token


# create langchain chat model from huggingface model
def get_langchain_huggingface_model(
    model_name: str,
    temperature: float = 0.0,
    max_new_tokens: int = 200,
    top_k: int = 50,
    do_sample: bool = True,
    device: str = "cuda",
    torch_dtype: str = "auto",
    attn_implementation: str = "sdpa",
    four_bit: bool = False,
    model: Union[AutoModelForCausalLM, None] = None,
    tokenizer: Union[AutoTokenizer, None] = None,
):
    """
    Arguments
    =========
    model_name : str
        The name of the model to load from Hugging Face.
    temperature : float, optional
        The temperature to use for sampling, default is 0.0.
    max_new_tokens : int, optional
        The maximum number of new tokens to generate, default is 200.
    top_k : int, optional
        The number of highest probability vocabulary tokens to keep for top-k-filtering, default is 50.
    do_sample : bool, optional
        Whether or not to use sampling; use greedy decoding otherwise, default is True.
    device : str, optional
        The device to load the model on, default is "cuda".
    torch_dtype : str, optional
        The data type for the model's tensors, default is "auto".
    attn_implementation : str, optional
        The attention implementation to use, default is "sdpa".
    four_bit : bool, optional
        Whether to load the model in 4-bit precision, default is False.
    model: AutoModelForCausalLM, optional
        If included, do not load the model from scratch but use an existing instance passed to this argument
    model: AutoTokenizer, optional
        If included, do not load the tokenizer from scratch but use an existing instance passed to this argument
    Returns
    =======
    ChatHuggingface
        An instance of the langchain Chat model for huggingface
    """

    if not model:
        model, tokenizer = get_huggingface_model(
            model_name, device, torch_dtype, attn_implementation, four_bit
        )
    else:
        assert (
            tokenizer is not None
        ), "if passing a pre-loaded model, you need to pass also the associated tokenizer!"

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        do_sample=do_sample,
        temperature=temperature,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    chat_model = ChatHuggingFace(llm=llm)

    return chat_model


##########################
# Langchain Chain
##########################


class Chain:
    def __init__(
        self,
        model_name: str,
        frame: str, # "huggingface" only for now
        max_new_tokens: int,
        do_sample: bool,
        device: str,
        torch_dtype: str,
        attn_implementation: str,
        four_bit: bool,
        temperature: Union[float, None] = None,
        top_k: Union[int, None] = None,
        model: Union[AutoModelForCausalLM, None] = None,
        tokenizer: Union[AutoTokenizer, None] = None,
        assistant_token: Union[str, None] = None,
    ):

        self.model_name = model_name
        self.frame = frame
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.do_sample = do_sample
        self.device = device
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.four_bit = four_bit
        self.assistant_token = assistant_token

        if frame == "huggingface":
            self.chat_model = get_langchain_huggingface_model(
                model_name=self.model_name,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                top_k=self.top_k,
                do_sample=self.do_sample,
                device=self.device,
                torch_dtype=self.torch_dtype,
                attn_implementation=self.attn_implementation,
                four_bit=self.four_bit,
                model=model,
                tokenizer=tokenizer,
            )
        else:
            raise NotImplementedError(
                "For now, only huggingface-based LLMs can be used!"
            )

    def run(
        self,
        inputs: dict,
        prompt_template: PromptTemplate,
        placeholder: dict = None,
    ) -> dict:
        """
        Arguments
        =========
        inputs: dict
            the inputs for the prompt
        prompt_template: PromptTemplate
            the prompt template to use

        Returns
        =======
        output
            Return the answer given by the LLM chain
        """

        chain = prompt_template | self.chat_model # to debug add a breakpoint here

        output = chain.invoke(inputs)

        # try to parse the output
        try:
            output_parsed = JsonOutputParser().parse(output.content)
            if isinstance(output_parsed, dict):
                if placeholder.keys() == output_parsed.keys():
                    return output_parsed
                else:
                    raise OutputParserException("Output does not match the expected format")
            else:
                raise OutputParserException("Output does not match the expected format")

        # if parsing fails, try to reformat the output
        except OutputParserException:
            # split the output in two parts: the output format and the content to correct
            keep_this = output.content.split("B) AFTER THE REASONING ")[1]
            output_format = "# OUTPUT\n" + keep_this.split("-----------------")[0]
            correct_this = keep_this.split(self.assistant_token)[-1]

            output = correct_output(self.chat_model, output_format, correct_this, placeholder)

        # try again to parse the output
        try:
            output_parsed = JsonOutputParser().parse(remove_single_quotes(output.content))
            if isinstance(output_parsed, dict):
                if placeholder.keys() == output_parsed.keys():
                    return output_parsed
                else:
                    return placeholder
            else:
                return placeholder
        except OutputParserException:
            # try again to correct
            correct_this = output.content.split(self.assistant_token)[-1]
            output = correct_output(self.chat_model, output_format, correct_this, placeholder)

        # if also the second correction fails, return a task-specific placeholder
        # (JSON object that will replace the output in case of parsing errors)
        # find it in prompts/<task_name>/placeholder.json
        try:
            output_parsed = JsonOutputParser().parse(remove_single_quotes(output.content))
            if isinstance(output_parsed, dict):
                if placeholder.keys() == output_parsed.keys():
                    return output_parsed
                else:
                    return placeholder
            else:
                return placeholder
        except OutputParserException:
            return placeholder

