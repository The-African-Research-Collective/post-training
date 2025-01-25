import json
from typing import Any, Dict


def json_parse_model_output(output: str) -> Dict[str, Any]:
    """
    This function parses the output of a model and returns a dictionary.
    it works by finding the first opening bracket and removing everything before it.
    Then it finds the last closing bracket and removes everything after it.
    Finally, it returns the JSON object.
    """

    # Find the first opening bracket and remove everything before it
    start = output.find("[")
    start_curly = output.find("{")

    if (start != -1 and start_curly < start) or (start == -1 and start_curly != -1):
        start = start_curly
        end = output.rfind("}")
    else:
        end = output.rfind("]", start)

    output = output[start:]
    output = output[: end - start + 1]

    return json.loads(output)


CHAT_TEMPLATES = []
