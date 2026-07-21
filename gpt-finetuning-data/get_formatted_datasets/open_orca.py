from datasets import load_dataset


class OpenOrcaProcessor:
    USER_TOKEN = "<|USER|>"
    ASSISTANT_TOKEN = "<|ASSISTANT|>"
    EOS_TOKEN = "<|EOS|>"
    BOS_TOKEN = "<|BOS|>"
    EOT_TOKEN = "<|EOT|>"

    def __init__(self, train_size="train[:500000]", test_size=0.01, seed=42):
        dataset = load_dataset(
            "Open-Orca/OpenOrca",
            split=train_size,
        )

        split = dataset.train_test_split(
            test_size=test_size,
            seed=seed,
        )

        self.train_dataset = split["train"]
        self.validation_dataset = split["test"]

    @staticmethod
    def format_conversation(sample):
        question = sample["question"].strip()
        response = sample["response"].strip()

        return [
            f"{OpenOrcaProcessor.USER_TOKEN}{question}{OpenOrcaProcessor.EOS_TOKEN}",
            f"{OpenOrcaProcessor.ASSISTANT_TOKEN}{response}{OpenOrcaProcessor.EOS_TOKEN}",
        ]

    @staticmethod
    def add_special_tokens(conversations):
        tokenized_conversations = []

        for conversation in conversations:
            tokenized_conversations.append(
                [OpenOrcaProcessor.BOS_TOKEN]
                + conversation
                + [OpenOrcaProcessor.EOT_TOKEN]
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


# processor = OpenOrcaProcessor()

# openorca_train_data = processor.get_train_data()
# openorca_validation_data = processor.get_validation_data()

# print(openorca_train_data[:2])
# print(len(openorca_train_data))

# print(openorca_validation_data[:2])
# print(len(openorca_validation_data))


