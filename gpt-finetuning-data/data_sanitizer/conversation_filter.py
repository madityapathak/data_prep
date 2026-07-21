import re
from data_sanitizer.utils import num_tokens

# This filter checks and keeps the first pair of input if needed
class ConversationFilter:
    MAX_CONTEXT = 1004

    PAIR_PATTERN = re.compile(
        r"(<\|USER\|>.*?<\|EOT\|>\s*<\|ASSISTANT\|>.*?<\|EOT\|>)",
        re.DOTALL,
    )

    ASSISTANT_PATTERN = re.compile(r"<\|ASSISTANT\|>")

    def filter_conversations(self, conversations):
        filtered_conversations = []
        left_data = []

        for conv in conversations:

            # these below three lines keep the whole conversation which are under the context length
            if num_tokens(conv) <= self.MAX_CONTEXT:
                filtered_conversations.append(conv)
                continue

            # these below lines tell not to change those conversation which have more that two turns keep them as they are so taht sliding window can be used if needed
            if len(self.ASSISTANT_PATTERN.findall(conv)) > 2:
                left_data.append(conv)
                continue

            # the below lines removes the entries that are of single conversation turn and has token more than context length
            if (
                len(self.ASSISTANT_PATTERN.findall(conv)) < 2
                and num_tokens(conv) > self.MAX_CONTEXT
            ):
                continue

            # below lines chck if conversation has two user assistant turns and checks if first pattern has number of tokens less than context length then keeps them
            pairs = self.PAIR_PATTERN.findall(conv)
            if not pairs:
                continue

            first_pair = "<|BOS|>" + pairs[0]

            # Keep only if the first pair fits
            if num_tokens(first_pair) <= self.MAX_CONTEXT:
                filtered_conversations.append(first_pair)

        return filtered_conversations, left_data
    


# conversation_filter = ConversationFilter()
# filtered_conversations, left_data = conversation_filter.filter_conversations(conversations)

