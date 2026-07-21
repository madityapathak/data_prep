from datasets import load_dataset


class DollyProcessor:
    USER_TOKEN = "<|USER|>"
    ASSISTANT_TOKEN = "<|ASSISTANT|>"
    EOS_TOKEN = "<|EOS|>"
    BOS_TOKEN = "<|BOS|>"
    EOT_TOKEN = "<|EOT|>"

    def __init__(self, test_size=0.05, seed=42):
        dataset = load_dataset("databricks/databricks-dolly-15k")

        split = dataset["train"].train_test_split(
            test_size=test_size,
            seed=seed,
        )

        self.train_dataset = split["train"]
        self.validation_dataset = split["test"]

    @staticmethod
    def format_conversation(sample):
        instruction = sample["instruction"].strip()
        context = sample["context"].strip()
        response = sample["response"].strip()

        if context:
            user_message = f"{instruction}\n\n{context}"
        else:
            user_message = instruction

        return [
            f"{DollyProcessor.USER_TOKEN}{user_message}{DollyProcessor.EOS_TOKEN}",
            f"{DollyProcessor.ASSISTANT_TOKEN}{response}{DollyProcessor.EOS_TOKEN}",
        ]

    @staticmethod
    def add_special_tokens(conversations):
        tokenized_conversations = []

        for conversation in conversations:
            tokenized_conversations.append(
                [DollyProcessor.BOS_TOKEN]
                + conversation
                + [DollyProcessor.EOT_TOKEN]
            )

        return tokenized_conversations

    @staticmethod
    def merge_conversation_tokens(conversations):
        merged_conversations = []

        for conversation in conversations:
            merged_conversations.append("".join(conversation))

        return merged_conversations

    def build_dataset(self, dataset):
        formatted_conversations = []

        for sample in dataset:
            formatted_conversations.append(
                self.format_conversation(sample)
            )

        # formatted_conversations = self.add_special_tokens(
        #     formatted_conversations
        # )

        formatted_conversations = self.merge_conversation_tokens(
            formatted_conversations
        )

        return formatted_conversations

    def get_train_data(self):
        return self.build_dataset(self.train_dataset)

    def get_validation_data(self):
        return self.build_dataset(self.validation_dataset)



# processor = DollyProcessor()

# dolly_train_data = processor.get_train_data()
# dolly_validation_data = processor.get_validation_data()

# print(dolly_train_data[:2])
# print(len(dolly_train_data))

# print(dolly_validation_data[:2])
# print(len(dolly_validation_data))

