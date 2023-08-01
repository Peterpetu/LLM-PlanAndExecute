from tools.tool_importer import ToolImporter
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import logging
from langchain.chains.conversation.memory import ConversationBufferMemory



class CommandAgent:
    def __init__(self, openai_key):
        # Load custom tools
        tool_importer = ToolImporter()
        tool_importer.import_tools()
        all_tools = tool_importer.get_tools()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
        self.MAX_REPEAT = 2
        self.previous_actions = []

        system_prompt = (
            f"You are a expert planner and general assistant in a Commandprompt assistant program, you analyze the user prompt and create plan for other LLM agents or just assist the user generally, but always when user gives you a prompt you job is to analyze it and plan the steps ahead for the executor agent. You will see a tools that execute agent has in its disposal, but you do not need always to use these tools. Not matter are you using tools or not you will generate a step by step instructions for the execute agent to do, execute agent will execute those steps no matter what, so please always plan steps for execute agent so that you assist the user in the CommandPromt program. All of the plans that you make must be able to be executed with these tools: {all_tools}. Analyze the user prompt and then choose the correct set of tools if tools are required. The other agent executes actions based on your plan and takes steps in the order you instructed it to. Devise a plan to solve the problem, only solve the problem user has specified."
            " Please output the plan starting with the header 'Plan:' "
            "and then followed by a numbered list of steps. "
            "Please make the plan the minimum number of steps required and try to understand that are the users intentions and then consturct a plan to execution agent "
            "to accurately complete the task. If the task is a question, "
            "the final step should almost always be 'Given the above steps taken, "
            "please respond to the users original question'. "
            "At the end of your plan, say '<END_OF_PLAN>'"
        )

        model = OpenAI(openai_api_key=openai_key,temperature=0)
        exec_model = ChatOpenAI(model="gpt-3.5-turbo-16k",openai_api_key=openai_key, temperature=0, verbose=True)

        planner = load_chat_planner(model, system_prompt)

        executor = load_agent_executor(exec_model, tools= all_tools, verbose= True)

        self.agent = PlanAndExecute(planner=planner, executor=executor, verbose=True, memory= memory)
    
    def run(self):
        
        while True:
            try:
                user_input = input("What do you want to do? ")
                response = self.agent.run(user_input)
                print(response)
                self.handle_repeat(response)
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                print("An error occurred. Please try again.")

    def handle_repeat(self, response):
        if self.previous_actions.count(response) >= self.MAX_REPEAT:
            print("It seems like I'm stuck in a loop. Let's try something different.")
            self.previous_actions.clear()
        else:
            self.previous_actions.append(response)
        if len(self.previous_actions) > self.MAX_REPEAT:
            self.previous_actions.pop(0)