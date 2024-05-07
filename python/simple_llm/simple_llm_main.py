import composo as cp
import copy
from litellm import completion
import litellm

# litellm.set_verbose = True

model_mapping = {
    "OpenAI GPT-3.5 Turbo": "gpt-3.5-turbo",
    "OpenAI GPT-4 Turbo": "gpt-4-1106-preview",
    "Google Gemini Pro": "gemini-pro",
    "Cohere Command": "command",
    "Cohere Command Light": "command-light",
    "Claude V2.1": "anthropic.claude-v2:1",
    "Meta Llama 2 13b": "meta.llama2-13b-chat-v1",
    "Meta Llama 2 70b": "meta.llama2-70b-chat-v1",
    "Mistral 8x7B": "mistral/mistral-medium",
}


@cp.Composo.link(auto_bump=True)
def simple_llm_call(
    model: cp.MultiChoiceStrParam(
        description="The specific language model to be used for generating responses.",
        choices=list(model_mapping.keys()),
    ),
    temperature: cp.FloatParam(
        description="A parameter controlling the randomness of the generated output",
        min=0.0,
        max=2.0,
    ),
    system_message: cp.StrParam(
        description="Pre-defined information or instructions given to the model as context for the current operation."
    ),
    conversation_history: cp.ConversationHistoryParam(
        description="A list of messages comprising the conversation so far, in OpenAI format"
    ),
) -> litellm.utils.CustomStreamWrapper:
    """
    A simple function to call an LLM model.
    """

    messages = copy.deepcopy(conversation_history)
    messages.insert(0, {"role": "system", "content": system_message})

    prediction = completion(
        model_mapping[model],
        messages,
        temperature=temperature,
        max_tokens=512,
        stream=True,
    )

    return prediction


stream = simple_llm_call(
    model="OpenAI GPT-3.5 Turbo",
    temperature=1,
    system_message="",
    conversation_history=[
        {
            "role": "user",
            "content": "",
        }
    ],
)
