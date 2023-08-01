from langchain.agents import Tool
import subprocess
import logging

class GitPullTool:
    def __init__(self, name="Git Pull", description="Fetch from and integrate with another repository or a local branch"):
        self.name = name
        self.description = description

    def git_pull(self, remote='origin', branch='master'):
        try:
            result = subprocess.run(["git", "pull", remote, branch], capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            logging.error(f"Failed to pull changes from Git repository: {e}")
            return f"Failed to pull changes from Git repository. Please try again."

    def to_tool(self):
        return Tool(
            name=self.name,
            func=self.git_pull,
            description=self.description
        )
