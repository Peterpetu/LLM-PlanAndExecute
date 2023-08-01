from langchain.agents import Tool
import os
import logging

class ChangeDirectoryTool:
    def __init__(self, name="Change Directory", description="Changes the current working directory"):
        self.name = name
        self.description = description

    def change_directory(self, new_directory):
        try:
            if not os.path.exists(new_directory):
                return f"No directory with the name {new_directory} exists."

            os.chdir(new_directory)
            return f"Current working directory changed to {new_directory} successfully."
        except Exception as e:
            logging.error(f"Failed to change working directory to {new_directory}: {e}")
            return f"Failed to change working directory to {new_directory}. Please try again."

    def to_tool(self):
        return Tool(
            name=self.name,
            func=self.change_directory,
            description=self.description
        )