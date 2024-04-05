# whatsapp-to-train
Turn your whatsapp conversations into training data for a large language model

## Usage

### Data Collection

1. Export your whatsapp conversation from the app, and put it in the `data` directory, with the filename `_chat.txt` (this is the default export).
2. Run `python3 src/clean.py` to clean the data and save it to `data/cleaned.txt`.
3. Then, to make your huggingface dataset, run `python3 src/make_dataset.py`. This will save the dataset to `data/chat_dataset`.


### Training

