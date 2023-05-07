import asyncio
import re

from better_multigpt import prompt_engineering
from better_multigpt.multi_agent.multi_agent_manager import MultiAgentManager


def transform_add_trait_profiles(inputs: dict) -> dict:
    # This extra step is necessary because of a bug in the lmql package.
    # when using lmql queries as langchain chains, variables in the where clause are treated as input variables

    async def _query_generate_trait_profile():
        result = (await prompt_engineering.generate_trait_profile(name))
        return result

    res = []
    # TODO make this loop concurrent
    for name, description, goals in inputs["expert_tuples"]:
        loop = asyncio.get_event_loop()
        traits = loop.run_until_complete(_query_generate_trait_profile())[0].variables
        res.append((name, description, goals, traits))
    return {"expert_tuples_w_traits": res}


def transform_into_agents(inputs: dict) -> dict:
    res = []
    for name, description, goals, traits in inputs["expert_tuples_w_traits"]:
        multi_agent_manager = MultiAgentManager.get_manager()
        res.append(multi_agent_manager.create_agent(name, description, goals, traits))
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
