import openai
import composo as cp

@cp.Composo.link(api_key="API_KEY_FOR_TESTING")
def simple_llm_call(
	model: cp.MultiChoiceStrParam(choices=["gpt-3.5-turbo", "gpt-4"]),
	temp: cp.FloatParam(min=0.0, max=2.0),
	system: cp.StrParam(description="System Message. Use {text}, {length} and {audience} to insert variables."),
	text: cp.StrParam(description='Text to summarise. Testing quotes in here" "'),
	length: cp.StrParam, 
	audience: cp.StrParam,
):
	replaced_system = system.format(length=length, audience=audience, text=text)
	messages = [
	{"role": "system", "content": replaced_system },
	]
	prediction = openai.ChatCompletion.create(
		model=model,
		messages=messages,
		temperature=temp,
		max_tokens=256,
	).choices[0].message["content"]
	return prediction

output = simple_llm_call(
	"gpt-3.5-turbo",
	0.7,
	"Summarise the text to {length} for {audience}. Text: {text}",
	"Generative Pre-trained Transformer 3 (GPT-3) is a large language model released by OpenAI’s research in 2020.", # This contains a funky character - good for testing!
	"5 words",
	"5 year old audience"
)