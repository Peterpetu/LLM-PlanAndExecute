# command_agent.py
from langchain.agents import initialize_agent
import logging

class CommandAgent:
    def __init__(self, tools, llm, agent, memory):
        self.agent_chain = initialize_agent(tools=tools, llm=llm, agent=agent, memory= memory, verbose=True)
        self.previous_actions = []
        self.MAX_REPEAT = 5
        
        

    def run(self):
        
        while True:
            try:
                user_input = input("What do you want to do? ")
                
                response = self.agent_chain.run(user_input)
                print(response)
                self.handle_repeat(response)
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                

    def handle_repeat(self, response):
        if self.previous_actions.count(response) >= self.MAX_REPEAT:
            print("It seems like I'm stuck in a loop. Let's try something different.")
            self.previous_actions.clear()
        else:
            self.previous_actions.append(response)
        if len(self.previous_actions) > self.MAX_REPEAT:
            self.previous_actions.pop(0)
