import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, AsyncOpenAI,Runner, OpenAIChatCompletionsModel, handoff
from agents.run import RunConfig, RunContextWrapper

load_dotenv()

Gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=Gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

def on_handoff(agent: Agent, ctx: RunContextWrapper[None]):
        agent_name = agent.name
        print("--------------------------------")
        print(f"Handing off to {agent_name}...")
        print("--------------------------------")
        # Send a more visible message in the chat
        cl.Message(
            content=f"🔄 **Handing off to {agent_name}...**\n\nI'm transferring your request to our {agent_name.lower()} who will be able to better assist you.",
            author="System"
        ).send()


   

research_agent = Agent(
    name="Research Agent",
    instructions="📚 You are a Research Agent. Your job is to research topics and provide factual summaries.",
    model=model
)

code_agent = Agent(
    name="Code Agent",
    instructions="👨‍💻 You are a Code Agent. Your job is to solve coding problems, write snippets, and fix bugs.",
    model=model
)

chat_agent = Agent(
    name="Chat Agent",
    instructions="💬 You are a Chat Agent. Your job is to have friendly, informative conversations with users.",
    model=model
)
manager = Agent(
    name="Manager Agent",
    instructions="""
        You are a smart routing agent. Your job is to identify the user's request and hand it off to the correct specialist agent (Code, Chat, or Research).
        Always give a short acknowledgment and tell the user you're routing them.
        Do not answer the question yourself.
    """,
    model=model,
    handoffs=[
        handoff(chat_agent, on_handoff=lambda ctx: on_handoff(chat_agent, ctx)),
        handoff(code_agent, on_handoff=lambda ctx: on_handoff(code_agent, ctx)),
        handoff(research_agent, on_handoff=lambda ctx: on_handoff(research_agent, ctx)),
    ]
)


@cl.on_chat_start
async def start():
    cl.user_session.set("manager", manager)
    cl.user_session.set("config", config)
    cl.user_session.set("research_agent", research_agent)
    cl.user_session.set("chat_agent", chat_agent)
    cl.user_session.set("code_agent", code_agent)
    cl.user_session.set("chat_history", [])
    await cl.Message(content="Welcome to Nitoo's AI Assistant! How can I help you today?").send()


@cl.on_message
async def main(message:cl.Message):

        msg = cl.Message(content="thinking.....")
        await msg.send()

        manager : Agent= cast(Agent , cl.user_session.get("manager"))
        config : RunConfig= cast(RunConfig , cl.user_session.get("config"))

        history = cl.user_session.get("chat_history" ) or []


        history.append({"role": "user", "content": message.content})

        try:
             result = Runner.run_sync(manager, history, run_config=config)

             response = result.final_output

             msg.content = response
             await msg.update()

             history.append({"role": "developer", "content": response})

             cl.user_session.set("chat_history", history)
             print(f"History: {history}")
             
        except Exception as e:
             msg.content = f"error: {str(e)}"
             msg.update()
             print(f"error: {str(e)}")


             


    
