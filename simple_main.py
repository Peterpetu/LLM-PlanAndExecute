# main.py

from simple_agent import CommandAgent
from langchain.chains.conversation.memory import ConversationBufferMemory
from tools.tool_importer import ToolImporter
from langchain.chat_models import ChatOpenAI


def main():
    # Load OpenAI API key
    with open('apikey.txt', 'r') as file:
        openai_key = file.read().replace('\n', '')

    # agent1 variable settings
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k",openai_api_key=openai_key, temperature=0, verbose=True)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent= "conversational-react-description"

    # Initialize the tool importer and import all tools
    tool_importer = ToolImporter()
    tool_importer.import_tools()
    all_tools = tool_importer.get_tools()

    # Initialize the agent
    agent1 = CommandAgent(all_tools, llm, agent, memory)

    # Run the main loop
    try:
        agent1.run()
    except KeyboardInterrupt:
        print("\nProgram has been stopped by the user. Performing cleanup...")
        # Perform any necessary cleanup here
        print("Cleanup complete. Exiting program.")

if __name__ == "__main__":
    main()