from langchain.agents import Tool
import os
import logging

class ListDirTool:
    def __init__(self, name="List Directory", description="Lists the contents of a directory"):
        self.name = name
        self.description = description

    def list_dir(self, directory):
        try:
            if not os.path.exists(directory):
                return f"No directory with the name {directory} exists."
            
            contents = os.listdir(directory)
            return contents
        except Exception as e:
            logging.error(f"Failed to list directory {directory}: {e}")
            return f"Failed to list directory {directory}. Please try again."

    def to_tool(self):
        return Tool(
            name=self.name,
            func=self.list_dir,
            description=self.description
        )
