import re


class ConversationWindowProcessor:
    PAIR_PATTERN = re.compile(
        r"<\|user\|>.*?<\|eot\|>\s*<\|assistant\|>.*?<\|eot\|>",
        re.DOTALL,
    )

    def __init__(self, max_length=1010):
        self.max_length = max_length

    def split_user_assistant_turns(self, conversation):
        """
        Returns a list of complete user-assistant turn pairs.

        Example:
            Input:
                U1 A1 U2 A2 U3 A3

            Output:
                [
                    'U1 A1',
                    'U2 A2',
                    'U3 A3'
                ]
        """
        return self.PAIR_PATTERN.findall(conversation)

    def sliding_window(self, input_list):
        windows = []
        n = len(input_list)

        start = 0

        while start < n:
            conv = ""
            end = start

            while end < n:
                candidate = (
                    input_list[end]
                    if not conv
                    else conv + " " + input_list[end]
                )

                if num_tokens(candidate) <= self.max_length:
                    conv = candidate
                    end += 1
                else:
                    break

            if windows and windows[-1].endswith(conv):
                break

            windows.append(conv)

            # We've consumed the rest of the list.
            # Don't create another window consisting only of already-seen items.
            if end == n:
                break

            # Couldn't fit more than one element -> nothing meaningful to overlap.
            if end - start <= 1:
                break

            # Overlap by exactly one element.
            start = end - 1

        return windows

    def process(self, conversations):
        """
        Args:
            conversations (list[str])

        Returns:
            list[str]
        """
        processed_data = []

        for conversation in conversations:
            turns = self.split_user_assistant_turns(conversation)
            windows = self.sliding_window(turns)
            processed_data.extend(windows)

        return processed_data
    



# processor = ConversationWindowProcessor(max_length=1010)

# processed_data = processor.process(left_data_to_fix)

