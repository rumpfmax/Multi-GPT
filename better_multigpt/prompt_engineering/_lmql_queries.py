import lmql


@lmql.query
async def generate_trait_profile(name):
    '''
    argmax(max_len=2000)
    """
        Rate {name} on a scale from 0 (extremly low degree of) to 10 (extremly high degree of) on the following five traits: Openness, Agreeableness, Conscientiousness, Emotional Stability and Assertiveness. Follow the format precisely:

        Openness: [OPENNESS]
        Agreeableness: [AGREEABLENESS]
        Conscientiousness: [CONSCIENTIOUSNESS]
        Emotional Stability: [EMOTIONAL_STABILITY]
        Assertiveness: [ASSERTIVENESS]

        Short description of personality traits of {name}:
        [DESCRIPTION]
    """
    from
        'openai/text-davinci-003'
    where
        INT(OPENNESS) and INT(AGREEABLENESS) and INT(CONSCIENTIOUSNESS) and INT(EMOTIONAL_STABILITY) and INT(ASSERTIVENESS)
    '''


@lmql.query
async def generate_experts(task, min_experts, max_experts):
    '''
    argmax(max_len=2000)
    """
        The task is: {task}.
        Please determine which historical or renowned experts would be best suited to complete the given task. Include all experts explicitly mentioned in the task. Name between {min_experts} and {max_experts} experts. List three goals for them to help the overall task. Follow the following format precisely:
        1. <Name of the person>: <Description of how they are useful>
        1a) <Goal a>
        1b) <Goal b>
        1c) <Goal c>
        [RESULT]
    """
    from
        'openai/gpt-4'
    '''
