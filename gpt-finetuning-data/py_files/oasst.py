from datasets import load_dataset
from utils import TARGETS, en_filter, dfs



oasst_dataset                   = load_dataset("OpenAssistant/oasst1")
training_messages               = oasst_dataset["train"]
validation_messages             = oasst_dataset["validation"]
english_training_messages       = training_messages.filter(en_filter)
english_validation_messages     = validation_messages.filter(en_filter)



class OASSTProcessor:

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
                parent_to_children.setdefault(parent_message_id, []).append(message_id)

        for message in messages:
            message_lookup[message["message_id"]] = {
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
        """
        Checks that every conversation:
        1. Has an even number of messages.
        2. Starts with a prompter.
        3. Alternates prompter -> assistant -> prompter -> assistant.
        """

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
        formatted_data = []

        for conversation in conversations:
            messages = []

            for message_id in conversation:
                message = message_lookup[message_id]["message"].strip()
                role = message_lookup[message_id]["role"]

                if role == "prompter":
                    messages.append(f"<|USER|>{message}<|EOS|>")

                elif role == "assistant":
                    messages.append(f"<|ASSISTANT|>{message}<|EOS|>")

            formatted_data.append(messages)

        return formatted_data

    @staticmethod
    def add_special_tokens(conversations):
        tokenized_conversations = []

        for conversation in conversations:
            tokenized_conversations.append(
                ["<|BOS|>"] +
                conversation +
                ["<|EOT|>"]
            )

        return tokenized_conversations
    

    @staticmethod
    def merge_conversation_tokens(conversations):
        merged_conversations = []

        for conversation in conversations:
            merged_conversations.append("".join(conversation))

        return merged_conversations




training_conversations, training_message_lookup = (
    OASSTProcessor.build_conversations(
        english_training_messages
    )
)

validation_conversations, validation_message_lookup = (
    OASSTProcessor.build_conversations(
        english_validation_messages
    )
)

OASSTProcessor.validate_conversations(
    training_conversations,
    training_message_lookup,
)

OASSTProcessor.validate_conversations(
    validation_conversations,
    validation_message_lookup,
)

oasst_train_data = OASSTProcessor.format_conversations(
    training_conversations,
    training_message_lookup,
)

oasst_val_data = OASSTProcessor.format_conversations(
    validation_conversations,
    validation_message_lookup,
)

oasst_train_data = OASSTProcessor.add_special_tokens(oasst_train_data)
oasst_val_data = OASSTProcessor.add_special_tokens(oasst_val_data)


oasst_train_data = OASSTProcessor.merge_conversation_tokens(oasst_train_data)
oasst_val_data = OASSTProcessor.merge_conversation_tokens(oasst_val_data)

print(oasst_train_data[5])


