"""
Cleans the data export.
"""

import re
from tqdm.auto import tqdm
import json


def main(filename: str) -> None:
    """
    Cleaning function, that reads the data and cleans it.

    Args:
        filename (str): The filename to read the data from.

    Returns:
        None
    """
    message_pattern = re.compile(
        r"\[(\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}:\d{2}\s[APM]{2})\]\s(.*?):\s(.*)"
    )
    messages = []
    last_sender = None

    with open(filename, "r", encoding="utf-8") as file:
        for line in tqdm(file):
            # Skip lines with "image omitted" or HTTPS links
            if "image omitted" in line or "video omitted" in line or "https://" in line:
                continue

            match = message_pattern.match(line)
            if match:
                timestamp, sender, message = match.groups()
                # Check if current message's sender is the same as the last one
                if sender == last_sender:
                    # Append message to the last message in the list with a newline
                    messages[-1]["message"] += "\n" + message
                else:
                    # New sender, create a new message entry
                    messages.append(
                        {"sender": sender, "timestamp": timestamp, "message": message}
                    )
                    last_sender = sender
            else:
                # Continuation of the last message
                if messages:
                    messages[-1]["message"] += " " + line.strip()

    # Convert messages to JSON Lines format
    jsonl_output = "\n".join(json.dumps(message) for message in messages)

    # Write the cleaned data to a new file
    with open("./data/cleaned_chat.jsonl", "w", encoding="utf-8") as file:
        file.write(jsonl_output)


if __name__ == "__main__":
    main(filename="./data/_chat.txt")
