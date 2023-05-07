from langchain.agents import AgentExecutor
from langchain.chains import SequentialChain, TransformChain
from better_multigpt import prompt_engineering
from better_multigpt import utils
from better_multigpt.multi_agent.multi_agent_manager import MultiAgentManager


def main() -> None:
    parse_experts_chain = TransformChain(input_variables=["RESULT"], output_variables=["expert_tuples"],
                                         transform=utils.transform_parse_experts)

    add_trait_profiles_chain = TransformChain(input_variables=["expert_tuples"],
                                              output_variables=["expert_tuples_w_traits"],
                                              transform=utils.transform_add_trait_profiles)

    transform_into_agents_chain = TransformChain(input_variables=["expert_tuples_w_traits"],
                                                 output_variables=["agents"],
                                                 transform=utils.transform_into_agents)

    task_to_agents_chain = SequentialChain(
        chains=[prompt_engineering.generate_experts, parse_experts_chain, add_trait_profiles_chain,
                transform_into_agents_chain],
        input_variables=["task", "min_experts", "max_experts"],
        output_variables=["agents"], verbose=True)
    result = task_to_agents_chain(dict(task='Build a spaceship to Mars.', min_experts=2, max_experts=5))
    manager = MultiAgentManager.get_manager()

    agent_executor = AgentExecutor.from_agent_and_tools(agent=result['agents'][0], tools=manager.tools, verbose=True)
    while True:
        agent_executor.run("Please determine the next best action")


if __name__ == "__main__":
    main()
