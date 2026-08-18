from agent.tool_registry import ToolRegistry
from agent.skills import SkillRegistry
from agent.llm_service import LLMService
from agent.agent import Agent
from agent.prompts import Prompt
from agent.conversation import Conversation

from monitoring import setup_tracing
setup_tracing()

tools = ToolRegistry()
skills = SkillRegistry("./")

agent = Agent(tools, LLMService(), Prompt.get_agent_instruction(), skills)

conversation = Conversation(agent)

print(conversation.ask("which model are you?"))