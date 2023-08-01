from langchain.agents import Tool
import os
import logging

class MakeDirTool:
    def __init__(self, name="Make Directory", description="Creates a directory with the given name"):
        self.name = name
        self.description = description

    def make_dir(self, directory):
        try:
            if os.path.exists(directory):
                return f"A directory with the name {directory} already exists."
            
            os.mkdir(directory)
            return f"Directory {directory} created successfully."
        except Exception as e:
            logging.error(f"Failed to create directory {directory}: {e}")
            return f"Failed to create directory {directory}. Please try again."

    def to_tool(self):
        return Tool(
            name=self.name,
            func=self.make_dir,
            description=self.description
        )
