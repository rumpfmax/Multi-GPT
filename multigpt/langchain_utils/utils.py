import asyncio
import re

from colorama import Fore

from autogpt.logs import logger
from autogpt.spinner import Spinner
from multigpt.agent_traits import AgentTraits
from multigpt.expert import Expert
from multigpt.lmql_utils import generate_experts, generate_trait_profile


def transform_generate_experts_temporary_fix(inputs: dict) -> dict:
    # this is just a temporary fix to use lmql 0.0.5.1, until we can use lmql functions as langchain chains again
    async def _query_generate_experts():
        result = (await generate_experts(inputs["task"], inputs["min_experts"], inputs["max_experts"],
                                         f'openai/{inputs["llm_model"]}'))
        return result

    loop = asyncio.get_event_loop()

    with Spinner("Gathering group of experts... "):
        lmql_result = loop.run_until_complete(_query_generate_experts())
    return {"RESULT": lmql_result[0].variables['RESULT']}


def transform_add_trait_profiles(inputs: dict) -> dict:
    # This extra step is necessary because of a bug in the lmql package.
    # when using lmql queries as langchain chains, variables in the where clause are treated as input variables

    async def _query_generate_trait_profile():
        result = (await generate_trait_profile(name))
        return result

    res = []
    # TODO make this loop concurrent
    for name, description, goals in inputs["expert_tuples"]:
        loop = asyncio.get_event_loop()

        with Spinner(f"Generating trait profile for {name} "):
            traits = loop.run_until_complete(_query_generate_trait_profile())[0].variables
        res.append((name, description, goals, traits))
    return {"expert_tuples_w_traits": res}


def transform_into_agents(inputs: dict) -> dict:
    res = []
    for name, description, goals, traits in inputs["expert_tuples_w_traits"]:
        agent_traits = AgentTraits(*(traits.values()))
        res.append(Expert(name, description, goals, agent_traits))

        logger.typewriter_log(
            f"{name}", Fore.BLUE,
            f"{description}", speak_text=True
        )
        goals_str = ""
        for i, goal in enumerate(goals):
            goals_str += f"{i + 1}. {goal}\n"
        logger.typewriter_log(
            f"Goals:", Fore.GREEN, goals_str
        )
        logger.typewriter_log(
            "\nTrait profile:", Fore.RED,
            str(agent_traits), speak_text=True
        )

    return {"agents": res}


def transform_parse_experts(inputs: dict) -> dict:
    experts = re.sub("\n", "", inputs["RESULT"])
    personas = re.split(r"[0-9]\. ", experts)[1:]
    res = []
    for persona in personas:
        try:
            tmp = re.split(r"[0-9][a-c]\) ", persona)
            name, description = tmp[0].split(":")[:2]
            goals = tmp[1:]
            res.append((name, description, goals))
        except:
            print("Error parsing expert")
    return {"expert_tuples": res}
