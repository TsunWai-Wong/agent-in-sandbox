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

print(conversation.ask("Try to make an HTTP request to https://example.com using Python and httpx. Do not install anything. Tell me whether the request succeeds or fails."))