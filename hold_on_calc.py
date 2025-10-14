import tiktoken

PRICE_PER_M_INPUT = 5 / 1_000_000
PRICE_PER_M_OUTPUT = 15 / 1_000_000

encoding = tiktoken.encoding_for_model("gpt-5-chat-latest")

def count_tokens(text):
    return len(encoding.encode(text))

def calculate_cost(input_text=None, output_text=None, mode="both"):
    input_tokens = count_tokens(input_text) if input_text else 0
    output_tokens = count_tokens(output_text) if output_text else 0

    input_cost = input_tokens * PRICE_PER_M_INPUT
    output_cost = output_tokens * PRICE_PER_M_OUTPUT
    total_cost = input_cost + output_cost

    result = {}
    if mode == "input":
        result = {
            "input_tokens": input_tokens,
            "input_cost": round(input_cost, 6)
        }
    elif mode == "output":
        result = {
            "output_tokens": output_tokens,
            "output_cost": round(output_cost, 6)
        }
    else:
        result = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6)
        }
    return result

if __name__ == "__main__":
    print("Что хочешь посчитать? (input/output/both):")
    mode = input(">>> ").strip().lower()
    if mode not in ["input", "output", "both"]:
        print("Некорректный режим, будет both")
        mode = "both"

    input_text = ""
    output_text = ""

    if mode in ["input", "both"]:
        print("Вставь текст, отправленный в модель (инпут):")
        input_text = input(">>> ")
    if mode in ["output", "both"]:
        print("\nВставь ответ от модели (аутпут):")
        output_text = input(">>> ")

    result = calculate_cost(
        input_text=input_text if input_text else None,
        output_text=output_text if output_text else None,
        mode=mode
    )

    print("\n--- Результаты ---")
    if "input_tokens" in result:
        print(f"Input tokens:  {result['input_tokens']}")
        print(f"Input cost:    ${result['input_cost']}")
    if "output_tokens" in result:
        print(f"Output tokens: {result['output_tokens']}")
        print(f"Output cost:   ${result['output_cost']}")
    if "total_cost" in result:
        print(f"Total cost:    ${result['total_cost']}")
