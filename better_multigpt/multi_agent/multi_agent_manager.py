import os

from langchain.agents import Tool
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from better_multigpt.multi_agent.multi_agent import LMQLConversationalAgent

from better_multigpt.multi_agent.multi_agent_output_parser import MultiAgentOutputParser
from better_multigpt.multi_agent.multi_agent_prompt_template import MultiAgentPromptTemplate


class MultiAgentManager:
    _instance = None

    @classmethod
    def get_manager(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if self._instance is not None:
            raise Exception("You cannot create another instance of this class. Use the get_instance() method.")

        self.agents = []
        self.template = ""
        self.output_parser = MultiAgentOutputParser()
        # put this into config
        self.llm = ChatOpenAI(temperature=0, model='gpt-4')
        self.tools = []

    def create_agent(self, name, task, goals, traits):
        provisional_prompt = f"Your name is {name}, your task is {task}, your goals are {goals}, your traits are {str(traits).strip('{}')}\n"

        os.environ["SERPAPI_API_KEY"] = "<add your key here>"
        search = SerpAPIWrapper()
        self.tools = [
            Tool(
                name="Search",
                func=search.run,
                description="useful for when you need to answer questions about current events"
            )
        ]

        prompt = MultiAgentPromptTemplate(
            template=provisional_prompt + self.template,
            tools=self.tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "agent_scratchpad"]
        )

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        tool_names = [tool.name for tool in self.tools]

        agent = LMQLConversationalAgent(llm_chain=llm_chain, output_parser=self.output_parser,
                                        stop=["\nObservation:"], allowed_tools=tool_names)
        self.agents.append(agent)
        return agent

    def delete_agent(self, agent):
        self.agents.remove(agent)
