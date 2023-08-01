from langchain.agents import Tool
import os
import logging

class ReadFileTool:
    def __init__(self, name="Read File", description="Reads the content of a file with the given name"):
        self.name = name
        self.description = description

    def read_file(self, filename):
        try:
            if not os.path.exists(filename):
                return f"No file with the name {filename} exists."
            with open(filename, 'r') as file:
                content = file.read()
            return content
        except Exception as e:
            logging.error(f"Failed to read file {filename}: {e}")
            return f"Failed to read file {filename}. Please try again."

    def to_tool(self):
        return Tool(
            name=self.name,
            func=self.read_file,
            description=self.description
        )
