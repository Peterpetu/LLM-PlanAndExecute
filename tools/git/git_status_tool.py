from langchain.agents import Tool
import subprocess
import logging

class GitStatusTool:
    def __init__(self, name="Git Status", description="Show the working tree status"):
        self.name = name
        self.description = description

    def git_status(self, _=None):
        try:
            result = subprocess.run(["git", "status"], capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            logging.error(f"Failed to get Git status: {e}")
            return f"Failed to get Git status. Please try again."

    def to_tool(self):
        return Tool(
            name=self.name,
            func=self.git_status,
            description=self.description
        )