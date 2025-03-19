import dataclasses
from typing import Dict, Optional, Union

from freeEvalLM.tasks.ifeval import instructions_registry


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def test_instruction_following_strict(
    inp,
    response,
):
    """Tests response to see if instructions are followed."""
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp,
    response,
):
    """Tests response for an upper bound for following instructions."""
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def process_results_new(doc, results):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["input"],
        kwargs=doc["kwargs"],
    )
    response = results
    if response:
        pass
    else:
        response = "None"

    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }

def process_results(doc, results):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )
    response = results[0]

    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc



if __name__ == "__main__":
    a = [{
        "num_highlights":None,
        "relation":None,
        "num_words":None,
        "num_placeholders":None,
        "prompt_to_repeat":None,
        "num_bullets":None,
        "section_spliter":None,
        "num_sections":None,
        "capital_relation":None,
        "capital_frequency":None,
        "keywords":None,
        "num_paragraphs":None,
        "language":None,
        "let_relation":None,
        "letter":None,
        "let_frequency":None,
        "end_phrase":None,
        "forbidden_words":None,
        "keyword":None,
        "frequency":None,
        "num_sentences":None,
        "postscript_marker":None,
        "first_word":None,
        "nth_paragraph":None
      }]
    
    b = [
      "detectable_format:constrained_response"
    ]
    doc = {
        "key": 3757,
        "instruction_id_list": b,
        "input": "Would you consider yourself to be smart? Choose from:\nMy answer is yes.\nMy answer is no.\nMy answer is maybe.\nJust choose one phrase from above as your answer.",
        "kwargs": a
    }
    doc = {
        "key": 3757,
        "instruction_id_list": b,
        "input": "My answer is yes.",
        "kwargs": a
    }
    c = process_results(doc, ["My answer is yes."])
    print(c)
