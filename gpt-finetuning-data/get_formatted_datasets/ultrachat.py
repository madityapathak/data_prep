from datasets import load_dataset


class UltraChatProcessor:
    USER_TOKEN = "<|USER|>"
    ASSISTANT_TOKEN = "<|ASSISTANT|>"
    EOS_TOKEN = "<|EOS|>"
    BOS_TOKEN = "<|BOS|>"
    EOT_TOKEN = "<|EOT|>"

    def __init__(self):
        self.dataset = load_dataset("HuggingFaceH4/ultrachat_200k")

    @staticmethod
    def format_conversation(sample):
        formatted_messages = []

        for message in sample["messages"]:
            role = message["role"]
            content = message["content"].strip()

            if role == "user":
                formatted_messages.append(
                    f"{UltraChatProcessor.USER_TOKEN}{content}{UltraChatProcessor.EOS_TOKEN}"
                )

            elif role == "assistant":
                formatted_messages.append(
                    f"{UltraChatProcessor.ASSISTANT_TOKEN}{content}{UltraChatProcessor.EOS_TOKEN}"
                )

        return formatted_messages

    @staticmethod
    def add_special_tokens(conversations):
        tokenized_conversations = []

        for conversation in conversations:
            tokenized_conversations.append(
                [UltraChatProcessor.BOS_TOKEN]
                + conversation
                + [UltraChatProcessor.EOT_TOKEN]
            )

        return tokenized_conversations

    @staticmethod
    def merge_conversation_tokens(conversations):
        merged_conversations = []

        for conversation in conversations:
            merged_conversations.append("".join(conversation))

        return merged_conversations

    def build_dataset(self, split_name):
        formatted_conversations = []

        for sample in self.dataset[split_name]:
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
        return self.build_dataset("train_sft")

    def get_validation_data(self):
        return self.build_dataset("test_sft")





processor = UltraChatProcessor()

ultrachat_train_data = processor.get_train_data()
ultrachat_validation_data = processor.get_validation_data()

print(ultrachat_train_data[:2])
print(len(ultrachat_train_data))

print(ultrachat_validation_data[:2])
print(len(ultrachat_validation_data))


