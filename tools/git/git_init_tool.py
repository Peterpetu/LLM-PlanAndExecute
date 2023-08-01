from langchain.agents import Tool
import os
import subprocess
import logging

class GitInitTool:
    def __init__(self, name="Git Init", description="Initializes a local Git repository"):
        self.name = name
        self.description = description

    def git_init(self, directory=None):
        if directory is None or directory.lower() == 'none':
            directory = '.'
        try:
            result = subprocess.run(["git", "init", directory], capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            logging.error(f"Failed to initialize Git repository: {e}")
            return f"Failed to initialize Git repository. Please try again."


    def to_tool(self):
        return Tool(
            name=self.name,
            func=self.git_init,
            description=self.description
        )
