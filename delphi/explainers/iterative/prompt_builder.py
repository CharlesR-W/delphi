from delphi.explainers.default.prompts import SYSTEM_ITERATIVE


def iterative_build_prompt(
    explanation: str,
    normal_examples: str,
    false_positives: str,
    false_negatives: str,
) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_ITERATIVE}]

    user_start = f"Current explanation: {explanation}\n\n"
    # print("Current explanation: ", explanation)
    user_start += f"Normal examples:\n{normal_examples}\n\n"
    # print("Normal examples: ", normal_examples)
    user_start += f"False negatives:\n{false_negatives}\n"
    # print("False negatives: ", false_negatives)
    user_start += f"False positives:\n{false_positives}\n\n"
    # print("False positives: ", false_positives)

    messages.append(
        {
            "role": "user",
            "content": user_start,
        }
    )
    print(
        f"[IterativePromptBuilder] Built prompt with last message: {messages[-1]['content']}"
    )

    return messages
