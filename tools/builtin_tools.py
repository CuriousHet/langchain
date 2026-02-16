from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import ShellTool

#built-in tool
search_tool = DuckDuckGoSearchRun()
results = search_tool.invoke('top news in india today')
print(results)

shell_tool = ShellTool()
results = shell_tool.invoke('')
print(results)