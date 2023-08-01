from langchain.agents import Tool
import subprocess
import logging

class GitAddTool:
    def __init__(self, name="Git Add", description="Add file contents to the index"):
        self.name = name
        self.description = description

    def git_add(self, file):
        try:
            subprocess.run(["git", "add", file], check=True)
            return f"File {file} added successfully."
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to add file to Git index: {e}")
            return f"Failed to add file to Git index. Please try again."

    def to_tool(self):
        return Tool(
            name=self.name,
            func=self.git_add,
            description=self.description
        )