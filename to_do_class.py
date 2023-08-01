import os
import logging
from langchain import LLMChain, OpenAI, PromptTemplate

class ToDoTool:
    def __init__(self, name="TODO", description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!"):
        self.name = name
        self.description = description

    def run_todo(self, inputs, context=None):
        with open('apikey.txt', 'r') as file:
            openai_key = file.read().replace('\n', '')
        todo_prompt = PromptTemplate.from_template("As an expert planner, your task is to carefully analyze the user's prompt and generate a comprehensive and precise todo list that aligns with the user's intentions. This todo list will be utilized by another LLM agent, so it must be clear and concise. Your goal is to only include objectives that the user has explicitly mentioned. Please devise a todo list specifically for the following objective: {objective}.")
        todo_chain = LLMChain(llm=OpenAI(openai_api_key = openai_key, temperature=0), prompt=todo_prompt, verbose= True)
        try:
            # You need to pass some inputs to the run method of todo_chain
            objective = inputs.get('objective', '')  # Get the objective from the inputs dictionary
            result = todo_chain.run({'objective': objective})  # Pass the objective to the run method
            logging.info(f"Inputs: {inputs}")  # Log the inputs
            return result  # Return the result
        except Exception as e:
            # If an error occurs, log it and return an error message
            logging.error(f"Failed to create a TODO list: {e}")
            return f"Failed to create a TODO list. Please try again."