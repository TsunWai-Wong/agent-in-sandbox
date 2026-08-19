from agent.tool_registry import ToolRegistry
from agent.skills import SkillRegistry
from agent.llm_service import LLMService
from agent.agent import Agent
from agent.prompts import Prompt
from agent.conversation import Conversation
from agent.docker_sandbox import DockerSandbox

from monitoring import setup_tracing
setup_tracing()

tools = ToolRegistry()

skills = SkillRegistry("./")
sandbox = DockerSandbox()
tools.register("execute", sandbox.execute)

agent = Agent(tools, LLMService(), Prompt.get_agent_instruction(), skills)

conversation = Conversation(agent)

print(f"Workspace: {sandbox.workspace}")
print(conversation.ask("Try to create a file at /etc/test.txt and tell me what happens."))
# print(conversation.ask("Write a Python file with a function add(a,b), write a pytest test for it, then run pytest and report the result."))