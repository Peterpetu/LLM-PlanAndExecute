import importlib
import os
import logging

class ToolImporter:
    def __init__(self):
        self.tool_dirs = self.discover_tool_dirs()
        self.all_tools = []
        self.log_setup()

    def log_setup(self):
        logging.basicConfig(filename='tool_import.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    def discover_tool_dirs(self):
        return [name for name in os.listdir('tools') if os.path.isdir(os.path.join('tools', name))]

    def import_tools(self):
        for tool_dir in self.tool_dirs:
            tool_files = [f for f in os.listdir(os.path.join('tools', tool_dir)) if f.endswith('.py') and f != '__init__.py']
            for tool_file in tool_files:
                module_name = tool_file[:-3]
                try:
                    module = importlib.import_module(f'tools.{tool_dir}.{module_name}')
                    class_name = ''.join(word.title() for word in module_name.split('_'))  # Get the class name from the module name
                    tool_class = getattr(module, class_name)  # Get the tool class from the module
                    tool = tool_class().to_tool()  # Instantiate the tool class and convert it to a Tool dataclass
                    self.all_tools.append(tool)
                    logging.info(f'Successfully imported tool {module_name} from {tool_dir}/{tool_file}')
                except Exception as e:
                    logging.error(f'Failed to import tool from {tool_dir}/{tool_file}: {e}')

    def get_tools(self):
        return self.all_tools
