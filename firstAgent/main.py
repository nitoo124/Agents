from dotenv import load_dotenv
import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,RunConfig
import chainlit as cl


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")


external_Client= AsyncOpenAI(
    api_key=gemini_api_key,
    base_url ="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = external_Client

)

config = RunConfig(
    model= model,
    model_provider= external_Client,
    tracing_disabled= True
)

assistant = Agent(
    name =  "Assistant",
    instructions = "you are a helpfull assistant",
    model = model
)


@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="hello, i am neetu's agent").send()


@cl.on_message
async def handle_message(message:cl.Message):
    history = cl.user_session.get("history")

    history.append({"role":"user", "content": message.content})
  

    result = Runner.run_sync(assistant, input=history, run_config= config)

    history.append({"role":"assistant", "content":result.final_output})
    cl.user_session.set("history", history)

    await cl.Message(content=result.final_output).send()


