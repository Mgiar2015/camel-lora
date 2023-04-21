"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "camel"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )
    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
    ) -> str:

        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        return res

    def gen_prompt(
        self,
        person_a: [None, str] = None,
        person_b: [None, str] = None,
        conversation: [None, str] = None,
    )->str:
        if person_a is None and person_b is None:
            instruction = "Generate the next utterence in the interesting and casual conversation between two speakers based on the provided conversation history."
            input_str = f"### Conversation History: {conversation}"

        elif person_b is None:
            instruction = "Generate the next utterence in the interesting and casual conversation between two speakers based on the provided conversation history. Some information on the persona of speaker A is provided below and can help guide the conversation."
            input_str = f"### Speaker A Persona: {person_a}.\n### Conversation History: {conversation}"

        elif person_a is None:
            instruction = "Generate the next utterence in the interesting and casual conversation between two speakers based on the provided conversation history. Some information on the persona of speaker B is provided below and can help guide the conversation."
            input_str = f"### Speaker B Persona: {person_b}.\n### Conversation History: {conversation}"

        else:
            instruction = "Generate the next utterence in the interesting and casual conversation between two speakers based on the provided conversation history. Some information on the persona of speaker A and B is provided below and can help guide the conversation."
            input_str = f"### Speaker A Persona: {person_a}.\n### Speaker B Persona: {person_b}.\n### Conversation History: {conversation}"

        return self.generate_prompt(instruction, input_str)

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
