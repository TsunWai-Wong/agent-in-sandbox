from agent.tool_registry import ToolRegistry
from agent.skills import SkillRegistry
from agent.llm_service import LLMService
from agent.agent import Agent
from agent.prompts import Prompt
from agent.conversation import Conversation
from agent.docker_sandbox import DockerSandbox
from agent.brower import Browser

from monitoring import setup_tracing
setup_tracing()

tools = ToolRegistry()

skills = SkillRegistry("./")
sandbox = DockerSandbox()
tools.register("execute", sandbox.execute)

browser = Browser()
tools.register("navigate", browser.navigate)
tools.register("click", browser.click)
tools.register("fill", browser.fill)
tools.register("select", browser.select)
tools.register("press", browser.press)
tools.register("scroll", browser.scroll)
tools.register("back", browser.back)

agent = Agent(tools, LLMService(), Prompt.get_brower_agent_instruction(), skills)

conversation = Conversation(agent, token_budget=11116384)

print(f"Sandbox Workspace: {sandbox.workspace}")
print(conversation.ask("Try to the site of https://news.ycombinator.com/ and go to the fifth latest article and summarize its content"))
# print(conversation.ask("Try to create a file at /etc/test.txt and tell me what happens."))
# print(conversation.ask("Write a Python file with a function add(a,b), write a pytest test for it, then run pytest and report the result."))