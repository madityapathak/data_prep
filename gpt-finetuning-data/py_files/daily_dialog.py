from datasets import load_dataset


class DailyDialogProcessor:
    USER_TOKEN          = "<|USER|>"
    ASSISTANT_TOKEN     = "<|ASSISTANT|>"
    EOS_TOKEN           = "<|EOS|>"
    BOS_TOKEN           = "<|BOS|>"
    EOT_TOKEN           = "<|EOT|>"

    def __init__(self):
        self.dataset = load_dataset("roskoN/dailydialog")

    def _format_conversation(self, conversation):
        formatted_messages = []

        for message_index, utterance in enumerate(conversation["utterances"]):
            utterance = utterance.strip()

            if message_index % 2 == 0:
                speaker_token = self.USER_TOKEN
            else:
                speaker_token = self.ASSISTANT_TOKEN

            formatted_messages.append(
                f"{speaker_token}{utterance}{self.EOS_TOKEN}"
            )

        return formatted_messages

    @staticmethod
    def add_special_tokens(conversations):
        tokenized_conversations = []

        for conversation in conversations:
            tokenized_conversations.append(
                [DailyDialogProcessor.BOS_TOKEN]
                + conversation
                + [DailyDialogProcessor.EOT_TOKEN]
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

        for conversation in self.dataset[split_name]:
            formatted_conversations.append(
                self._format_conversation(conversation)
            )

        formatted_conversations = self.add_special_tokens(
            formatted_conversations
        )

        formatted_conversations = self.merge_conversation_tokens(
            formatted_conversations
        )

        return formatted_conversations

    def get_train_data(self):
        return self.build_dataset("train")

    def get_validation_data(self):
        return self.build_dataset("validation")




processor = DailyDialogProcessor()

train_data = processor.get_train_data()
validation_data = processor.get_validation_data()

print(train_data[:2])
print(len(train_data))



