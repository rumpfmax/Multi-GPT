"""Setup the AI and its goals"""
import glob
import os

from colorama import Fore
from langchain.chains import TransformChain, SequentialChain

from autogpt import utils
from multigpt import langchain_utils
from autogpt.spinner import Spinner
from multigpt import lmql_utils
from multigpt.agent_traits import AgentTraits
from multigpt.expert import Expert
from autogpt.logs import logger


def prompt_user(cfg, multi_agent_manager):
    logger.typewriter_log(
        "Welcome to MultiGPT!", Fore.BLUE,
        "I am the orchestrator of your AI assistants.", speak_text=True
    )
    experts = []
    saved_agents_directory = os.path.join(os.path.dirname(__file__), "saved_agents")
    if os.path.exists(saved_agents_directory):
        file_pattern = os.path.join(saved_agents_directory, '*.yaml')
        yaml_files = glob.glob(file_pattern)
        if yaml_files:
            agent_names = []
            for yaml_file in yaml_files:
                expert = Expert.load(yaml_file)
                agent_names.append(expert.ai_name)
                experts.append(expert)
            logger.typewriter_log(
                "Found existing agents!", Fore.BLUE,
                f"List of agents: {agent_names}", speak_text=True
            )
            loading = utils.clean_input("Do you want me to load these agents [Y/n]: ")
            if loading.upper() == "Y":

                logger.typewriter_log(
                    f"LOADING SUCCESSFUL!", Fore.YELLOW
                )

                for expert in experts:
                    logger.typewriter_log(
                        f"{expert.ai_name}", Fore.BLUE,
                        f"{expert.ai_role}", speak_text=True
                    )
                    goals_str = ""
                    for i, goal in enumerate(expert.ai_goals):
                        goals_str += f"{i + 1}. {goal}\n"
                    logger.typewriter_log(
                        f"Goals:", Fore.GREEN, goals_str
                    )
                    logger.typewriter_log(
                        "\nTrait profile:", Fore.RED,
                        str(expert.ai_traits), speak_text=True
                    )
                additional_agents = utils.clean_input(
                    "Do you want to create additional agents with a new task that join the discussion? [Y/n]: ")
                if additional_agents.upper() == "Y":
                    pass
                else:
                    for expert in experts:
                        multi_agent_manager.create_agent(expert)
                    return
            elif loading.upper() == "N":
                experts = []
            else:
                exit(1)

    logger.typewriter_log(
        "Define the task you want to accomplish and I will gather a group of expertGPTs to help you.", Fore.BLUE,
        "Be specific. Prefer 'Achieve world domination by creating a raccoon army!' to 'Achieve world domination!'",
        speak_text=True,
    )

    task = utils.clean_input("Task: ")
    if task == "":
        task = "Achieve world domination!"

    # This chain is just temporary until lmql chains work again
    generate_experts_chain = TransformChain(input_variables=["task", "min_experts", "max_experts", "llm_model"],
                                            output_variables=["RESULT"],
                                            transform=langchain_utils.transform_generate_experts_temporary_fix)

    parse_experts_chain = TransformChain(input_variables=["RESULT"], output_variables=["expert_tuples"],
                                         transform=langchain_utils.transform_parse_experts)

    add_trait_profiles_chain = TransformChain(input_variables=["expert_tuples"],
                                              output_variables=["expert_tuples_w_traits"],
                                              transform=langchain_utils.transform_add_trait_profiles)

    transform_into_agents_chain = TransformChain(input_variables=["expert_tuples_w_traits"],
                                                 output_variables=["agents"],
                                                 transform=langchain_utils.transform_into_agents)

    task_to_agents_chain = SequentialChain(
        chains=[generate_experts_chain, parse_experts_chain, add_trait_profiles_chain,
                transform_into_agents_chain],
        input_variables=["task", "min_experts", "max_experts", "llm_model"],
        output_variables=["agents"], verbose=True)

    logger.typewriter_log(f"Using Browser:", Fore.GREEN, cfg.selenium_web_browser)
    result = task_to_agents_chain(
        dict(task=task, min_experts=cfg.min_experts, max_experts=cfg.max_experts, llm_model=cfg.smart_llm_model))
    experts += result['agents']
    for expert in experts:
        multi_agent_manager.create_agent(expert)
