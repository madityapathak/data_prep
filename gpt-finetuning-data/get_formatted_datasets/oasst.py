from datasets import load_dataset
from utils import en_filter, dfs


class OASSTProcessor:
    USER_TOKEN = "<|USER|>"
    ASSISTANT_TOKEN = "<|ASSISTANT|>"
    EOS_TOKEN = "<|EOS|>"
    BOS_TOKEN = "<|BOS|>"
    EOT_TOKEN = "<|EOT|>"

    def __init__(self):
        dataset = load_dataset("OpenAssistant/oasst1")

        self.train_dataset = dataset["train"].filter(en_filter)
        self.validation_dataset = dataset["validation"].filter(en_filter)

    @staticmethod
    def build_conversations(messages):
        parent_to_children = {}
        message_lookup = {}
        root_message_ids = []
        conversations = []

        for message in messages:
            message_id = message["message_id"]
            parent_message_id = message["parent_id"]

            if parent_message_id is None:
                root_message_ids.append(message_id)
            else:
                parent_to_children.setdefault(
                    parent_message_id, []
                ).append(message_id)

            message_lookup[message_id] = {
                "message": message["text"],
                "role": message["role"],
            }

        for root_message_id in root_message_ids:
            dfs(
                node=root_message_id,
                path=[],
                message_role_dict=message_lookup,
                children=parent_to_children,
                conversations=conversations,
            )

        return conversations, message_lookup

    @staticmethod
    def validate_conversations(conversations, message_lookup):
        valid = True

        for conversation_index, conversation in enumerate(conversations):

            if len(conversation) % 2 != 0:
                print(
                    f"Conversation {conversation_index} has odd length: {len(conversation)}"
                )
                valid = False

            expected_role = "prompter"

            for message_id in conversation:
                actual_role = message_lookup[message_id]["role"]

                if actual_role != expected_role:
                    print(
                        f"Conversation {conversation_index}: "
                        f"Expected '{expected_role}' but found '{actual_role}'."
                    )
                    valid = False
                    break

                expected_role = (
                    "assistant"
                    if expected_role == "prompter"
                    else "prompter"
                )

        if valid:
            print("✓ All conversations are valid.")

        return valid

    @staticmethod
    def format_conversations(conversations, message_lookup):
        formatted_conversations = []

        for conversation in conversations:
            formatted_messages = []

            for message_id in conversation:
                message = message_lookup[message_id]["message"].strip()
                role = message_lookup[message_id]["role"]

                if role == "prompter":
                    formatted_messages.append(
                        f"{OASSTProcessor.USER_TOKEN}{message}{OASSTProcessor.EOS_TOKEN}"
                    )
                else:
                    formatted_messages.append(
                        f"{OASSTProcessor.ASSISTANT_TOKEN}{message}{OASSTProcessor.EOS_TOKEN}"
                    )

            formatted_conversations.append(formatted_messages)

        return formatted_conversations

    @staticmethod
    def add_special_tokens(conversations):
        tokenized_conversations = []

        for conversation in conversations:
            tokenized_conversations.append(
                [OASSTProcessor.BOS_TOKEN]
                + conversation
                + [OASSTProcessor.EOT_TOKEN]
            )

        return tokenized_conversations

    @staticmethod
    def merge_conversation_tokens(conversations):
        merged_conversations = []

        for conversation in conversations:
            merged_conversations.append("".join(conversation))

        return merged_conversations

    def build_dataset(self, dataset):
        conversations, message_lookup = self.build_conversations(dataset)

        self.validate_conversations(
            conversations,
            message_lookup,
        )

        formatted_conversations = self.format_conversations(
            conversations,
            message_lookup,
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




# processor = OASSTProcessor()

# oasst_train_data = processor.get_train_data()
# oasst_validation_data = processor.get_validation_data()

# print(oasst_train_data[:2])
# print(len(oasst_train_data))

# print(oasst_validation_data[:3])
# print(len(oasst_validation_data))

