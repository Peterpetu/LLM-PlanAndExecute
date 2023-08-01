from langchain.agents import Tool
import os
import logging


class CreateFileTool:
    def __init__(self, name="Create File", description="Creates file with the given name"):
        self.name = name
        self.description = description

    def create_file(self, filename):
        try:
            if os.path.exists(filename):
                return f"A file with the name {filename} already exists."
            
            command = f"echo. > {filename}"
            os.system(command)
            return f"File {filename} created successfully."
        except Exception as e:
        # If an error occurs, log it and return an error message
            logging.error(f"Failed to create file {filename}: {e}")
            return f"Failed to create file {filename}. Please try again."
    
    def to_tool(self):
        return Tool(
            name=self.name,
            func=self.create_file,
            description=self.description
        )
