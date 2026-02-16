from langchain_core.tools import tool, StructuredTool, BaseTool
from pydantic import BaseModel, Field
from typing import Type

# Method-1 : Using tool decorator
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integer numbers"""
    return a*b

res = multiply.invoke({"a":13,"b":3})
print(res)

print(multiply.name)
print(multiply.description)
print(multiply.args)


# Method-2 : Using StructuredTool

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")

def multiply_func(a: int, b: int) -> int:
    return a*b

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiplicationp",
    description="multiply 2 numbers",
    args_schema=MultiplyInput
)   

res = multiply_tool.invoke({"a":3, "b":6})
print(res)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)


# Method-3 : BaseTool

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"

    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b
    
multiply_tool = MultiplyTool()
result = multiply_tool.invoke({'a':3, 'b':3})

print(result)
print(multiply_tool.name)
print(multiply_tool.description)

print(multiply_tool.args)