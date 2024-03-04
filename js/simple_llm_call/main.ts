import OpenAI from "openai";
import { linkComposo } from "composo";

async function simpleLLMCall(
  model: string,
  temperature: number,
  system_message: string,
  conversation_history: OpenAI.Chat.Completions.ChatCompletionMessageParam[]
): Promise<string> {
  const client = new OpenAI();

  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: "system", content: system_message },
    ...conversation_history,
  ];

  const completion = await client.chat.completions.create({
    messages: messages,
    model: model,
    temperature: temperature,
  });

  return completion.choices[0].message.content;
}

linkComposo(simpleLLMCall, [
  "gpt-3.5-turbo",
  1,
  "You are a helpful assistant.",
  [
    {
      role: "user",
      content: "What is the meaning of life?",
    },
  ],
]);
