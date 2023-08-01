
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain import OpenAI, LLMChain
from tools.tool_importer import ToolImporter
from langchain.schema.output_parser import OutputParserException
from to_do_class import ToDoTool
# Load OpenAI API key
with open('apikey.txt', 'r') as file:
    openai_key = file.read().replace('\n', '')

# Define your embedding model
embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)
# Initialize the vectorstore as empty
import faiss

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})



# Load custom tools
tool_importer = ToolImporter()
tool_importer.import_tools()
custom_tools = tool_importer.get_tools()



### Define the TaskAnalyzeChain, TaskPrioritizationChain, and ExecutionChain classes
class TaskAnalyzeChain(LLMChain):
    """Chain to generates tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = (
            " These are the tasks remaining in the task list created from the user prompt: {incomplete_tasks}."
            " The last completed task has the result: {result}."
            " This result was based on the last task description: {task_description}."
            " The execution agent has these tools in its disposal: {custom_tools}, only make tasks that can be done with existing tools."
            " Based on the result, remove the completed tasks from tasks list, do not add any new tasks."
            " Return the remaining tasks in the same format as they where presented to you. This is how they where presented to you:{incomplete_tasks}."
            " If that was empty it means that all the tasks designed from user prompt were done and you do not need to plan any new tasks"
        )
        
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=[
                "result",
                "task_description",
                "incomplete_tasks",
                "custom_tools",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:

        """Get the response parser."""
        task_prioritization_template = (
        " You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing"
        " Here are the tools ExecutionAgent has in its disposal: {custom_tools}, if there are any tasks in the tasks that cannot be executed with these tools, please remove extra tasks"
        " These are the remaining tasks: {incomplete_tasks}. If there are more tasks in the tasks please remove the extra tasks."
        " the following tasks: {task_names}."
        " Do not include any tasks that have been completed."
        " Return the result as a numbered list, like:"
        " #. First task"
        " #. Second task"
        " Start the task list with number {next_task_id}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["task_names", "next_task_id", "custom_tools", "incomplete_tasks"])
        return cls(prompt=prompt, llm=llm, verbose=verbose)

        
def get_next_task(
    task_creation_chain: LLMChain,
    result: Dict,
    task_description: str,
    task_list: List[str],
    custom_tools: List[str]
) -> List[Dict]:
    """Get the next task."""
    # Remove the completed task from the list
    if task_description in task_list:
        task_list.remove(task_description)
    
    incomplete_tasks = ", ".join(task_list)
    custom_tools_str = ", ".join([tool.name for tool in custom_tools])
    
    response = task_creation_chain.run(
        result=result,
        task_description=task_description,
        incomplete_tasks=incomplete_tasks,
        custom_tools=custom_tools_str,
    )
    new_tasks = response.split("\n")
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]


def prioritize_tasks(
    task_prioritization_chain: LLMChain,
    this_task_id: int,
    task: Dict,
    task_list: List[Dict],
    custom_tools: List[str],
    task_description: str,
    pre_incomplete_task_list: List[str]
) -> List[Dict]:
    """Prioritize tasks."""
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    if task in pre_incomplete_task_list:
        pre_incomplete_task_list.remove(task_description)

    # Convert custom tools dict keys to a string
    custom_tools_str = ", ".join([tool.name for tool in custom_tools])
    pre_incomplete_task_list = [task for task in pre_incomplete_task_list if task != "."]

    # Create incomplete_tasks list
    incomplete_tasks_str = ", ".join(pre_incomplete_task_list)


    response = task_prioritization_chain.run(
        task_names=task_names, 
        next_task_id=next_task_id, 
        custom_tools=custom_tools_str, 
        incomplete_tasks=incomplete_tasks_str
    )

    new_tasks = response.split("\n")
    prioritized_task_list = []
    for task_string in new_tasks:
        if not task_string.strip():
            continue
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            prioritized_task_list.append({"task_id": task_id, "task_name": task_name})

    return prioritized_task_list

def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:
    """Get the top k tasks based on the query."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        return []
    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
    return [str(item.metadata["task"]) for item in sorted_results]


def execute_task(
    vectorstore, execution_chain: LLMChain, task: str, k:int = 5) -> str:
    """Execute a task."""
    task_name = task["task_name"]
    context = _get_top_tasks(vectorstore, task_name, k)
    context = "\n".join(context)
    response = execution_chain.run(context=context, task=task)
    return response


# Define the BabyAGI class
class BabyAGI(Chain, BaseModel):
    """Controller model for the BabyAGI agent."""

    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskAnalyzeChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: AgentExecutor = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=True)
    max_iterations: Optional[int] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

    def print_task_result(self, result: str):
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs["objective"]
        

        todo_tool = ToDoTool()
        todo_list = todo_tool.run_todo({"objective": objective})
        

        # Split the todo_list into tasks and add them to the task deque
        tasks = [task for task in todo_list.split('\n') if task.strip()]
        for i, task_name in enumerate(tasks, start=1):
            task_name = task_name.split('. ', 1)[-1]
            self.add_task({"task_id": i, "task_name": task_name})

        num_iters = 0
        while True:
            if self.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)

                # Step 2: Execute the task
                result = execute_task(
                    self.vectorstore, self.execution_chain, task
                )
                this_task_id = int(task["task_id"])
                self.print_task_result(result)

                # Step 3: Store the result in VectorStore
                result_id = f"result_{task['task_id']}"
                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = get_next_task(
                    self.task_creation_chain,
                    result,
                    task["task_name"],
                    [t["task_name"] for t in self.task_list],
                    custom_tools  # Pass the custom tools
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)

                self.task_list = deque(
                    prioritize_tasks(
                        self.task_prioritization_chain,
                        this_task_id,
                        task["task_name"],
                        list(self.task_list),
                        custom_tools,
                        task_description = task["task_name"],
                        pre_incomplete_task_list = [t["task_name"] for t in self.task_list])
                )
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print(
                    "\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m"
                )
                break
        return {}
    


    @classmethod
    def from_llm(
        cls, llm: BaseLLM, vectorstore: VectorStore, verbose: bool = False, **kwargs
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskAnalyzeChain.from_llm(llm, verbose=verbose)
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )
        # Define your custom prompt
        prefix = """You are an AI in a system where other AI agents input a task list for you: {context}. Take into account these previously completed tasks: {task}, reflect what ools you have in your disposal, do not plan any actions that you cannot execute with this toolset."""
        suffix = """Question: {task}{agent_scratchpad}"""
        prompt = ZeroShotAgent.create_prompt(custom_tools,prefix=prefix,suffix=suffix,input_variables=["task","context", "agent_scratchpad"])
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in custom_tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=custom_tools, verbose=True)
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=agent_executor,
            vectorstore=vectorstore,
            **kwargs,
        )
    

# Define the main function
def main():
    """Main function to run the BabyAGI agent."""
    # Load the LLM
    llm = OpenAI(openai_api_key=openai_key, temperature=0)
    OBJECTIVE = "Read main.py file and after that Please make a file example.py"

    # Initialize the BabyAGI agent
    baby_agi = BabyAGI.from_llm(llm, vectorstore, verbose=True)

    # Run the agent
    try:
        baby_agi({"objective": OBJECTIVE})
    except OutputParserException as e:
        print("Error parsing output:", e.llm_output)
        raise e

if __name__ == "__main__":
    main()