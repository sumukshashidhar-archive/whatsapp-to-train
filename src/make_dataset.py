import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
import json
from datasets import Dataset
import tiktoken

load_dotenv()

to_emulate = os.getenv("TO_EMULATE")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
encoding = tiktoken.get_encoding("cl100k_base")
context_len = 2048

dataset = []

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens


def main(filename: str, max_context_window=32) -> None:
    for window in range(max_context_window, 0, -1):
        message_list = []
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                message = json.loads(line)
                parsed_message = message['message']
                parsed_sender = message['sender']
                if len(message_list) < window:
                    if parsed_sender == to_emulate:
                        message_list.append(
                            {
                                "role" : "assistant",
                                "content" : f"{parsed_message}"
                            }
                        )
                    else:
                        message_list.append(
                            {
                                "role" : "user",
                                "content" : f"{parsed_message}"
                            }
                        )
                else:
                    # means we have reached the max_context_window, we can remove the first two elements
                    message_list.pop(0)
                    if parsed_sender == to_emulate:
                        message_list.append(
                            {
                                "role" : "assistant",
                                "content" : f"{parsed_message}"
                            }
                        )
                        # this is where we have a dataset row
                        dataset.append(
                            tokenizer.apply_chat_template(message_list, return_tensors='pt' ,tokenize=False)
                        )
                    else:
                        message_list.append(
                            {
                                "role" : "user",
                                "content" : f"{parsed_message}"
                            }
                        )
                
        return dataset


if __name__ == "__main__":
    dataset = main(filename="./data/cleaned_chat.jsonl")
    correct_dataset = []
    for i in range(len(dataset)):
        if num_tokens_from_string(dataset[i]) > context_len:
            continue
        else:
            correct_dataset.append(dataset[i])
    df = Dataset.from_dict({"text" : correct_dataset})
    # save this to disk
    print(len(df))
    print(len(dataset))
    print(df[0])
    df.save_to_disk("./data/chat_dataset")
