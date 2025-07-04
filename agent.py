
import os
import shutil
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType

load_dotenv()

from Tools.quiz import run_interactive_quiz
from Tools.summariser import summarize_topic
from Tools.retrieval import answer_question
from Tools.ingestion import load_pdf_chunks, embed_and_store


print("Upload NCERT PDF ")
pdf_path = input("Enter path to PDF: ").strip()

if not os.path.exists(pdf_path):
    print("File not found. Please make sure the path is correct.")
    exit()

print(" Processing and indexing your NCERT book...")
chunks = load_pdf_chunks(pdf_path)
embed_and_store(chunks)
print(f"{len(chunks)} chunks processed and stored!\n")


def quiz_tool_input(prompt: str):
    return run_interactive_quiz(prompt)

def summary_tool_input(prompt: str):
    return summarize_topic(prompt)

tools = [
    Tool(
        name="AnswerQuestion",
        func=answer_question,
        description="Use this to answer factual or conceptual questions from NCERT textbooks.if the topic is not found, dont provide general information just say Topic not found in NCERT content please provide the topic"
    ),
    Tool(
        name="GenerateQuiz",
        func=quiz_tool_input,
        description="Use this to create a quiz with MCQs on a topic from NCERT. if the topic is not found, dont provide general information just say Topic not found in NCERT content please provide the topic"
    ),
    Tool(
        name="SummarizeTopic",
        func=summary_tool_input,
        description="Use this to summarize a topic or chapter from NCERT in simple terms. if the topic is not found, dont provide general information just say Topic not found in NCERT content please provide the topic."
    ),
]


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

agent = initialize_agent(
    tools,
    llm,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

def run_agent():
    print("📘 NCERT AI Agent Ready!")
    print("Type your request (or 'exit'):\n")
    while True:
        user_input = input("🧑‍🎓 You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting the agent. Cleaning up vector store...")
            if os.path.exists("db"):
                shutil.rmtree("db")
                print("🧹 Vector store cleaned up!")
            break
        response = agent.run(user_input)
        print(f"\n🤖 Agent: {response}\n")
        

if __name__ == "__main__":
    run_agent()