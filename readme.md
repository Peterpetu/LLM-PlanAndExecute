## LLM-Agent Plan-Use-Tools-Execute

LLM-Plan-Use-Tools-Execute is a Python repository designed for testing Large Language Models (LLMs) with custom tools. The selection and utilization of tools within LLMs are vital for creating multipurpose general agents capable of handling various tasks. Planning abilities are essential for executing multi-step tasks with dependencies. The integration of tool usage and multi-step task handling can enhance LLM agents/assistants, making them more effective as decision-making embedded assistants in applications.

### Agents

#### Simple-Agent
Simple-Agent is an agent capable of executing multi-step tasks but lacks planning capabilities.

#### PlanAndExecute-Agent
PlanAndExecute-Agent is a dualistic assistant design that consists of two experts: one for planning and another for executing the planned tasks.

#### BabyAGI-Agent
BabyAGI-Agent is an LLM-based system with four different experts responsible for task generation, task analysis, task prioritization, and execution. This agent operates in continuous cycles and was designed for ongoing iterative processes.

### Tools
The tools are categorized, and all agents have access to the same set. In this simple testing scenario, there are tools for executing Windows command prompt commands, enabling the agent to interact more closely with its environment. There are also Git commands to enhance the agent's functionality as a Windows command prompt assistant.

A notable tool is the Weaviate vector database query tool, which serves as a "router" LLM. The main agent can utilize this tool as a knowledge database to retrieve accurate data on a chosen topic.

### Usage

You can run the main files for different agents:
```python
python BabyAGI_main.py
python PlanAndExecute_main.py
python Simple_main.py

```
Remember to set-up OpenAI api key to:
```python
apikey.txt
```
### Setting up Weaviate vector database:

```python
docker-compose up
```
```python
setup_weaviate.py
```
```python
data_from_csv_to_weaviate.py
```

### License

[MIT](https://choosealicense.com/licenses/mit/)