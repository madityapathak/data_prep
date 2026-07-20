import re


class DatasetCleaner:
    """
    Utility class for cleaning training datasets.

    Current operations:
    - Remove conversations containing HTML/web-development tags.
    - Remove the contents of unwanted tags (e.g. <think>, <analysis>).
    - Remove long contiguous non-whitespace sequences.
    """

    REMOVE_TAGS = {
        "joke",
        "think",
        "analysis",
        "reflection",
        "scratchpad",
    }

    HTML_TAGS = {
        "html", "head", "body", "div", "span", "script", "style",
        "table", "tbody", "tr", "ul", "li", "form", "label",
        "button", "header", "footer", "nav", "main", "section",
        "aside", "template", "video", "h1", "h2", "h3", "h4",
        "h5", "h6", "p", "strong", "em", "sup",
        "a-assets", "a-scene", "mat-form-field", "my-element",
    }

    def __init__(self):
        self.remove_pattern = re.compile(
            rf"<({'|'.join(map(re.escape, self.REMOVE_TAGS))})\b[^>]*>.*?</\1>",
            flags=re.DOTALL | re.IGNORECASE,
        )

        self.html_pattern = re.compile(
            rf"<({'|'.join(map(re.escape, self.HTML_TAGS))})\b",
            flags=re.IGNORECASE,
        )

    def clean_html_and_tags(self, strings: list[str]) -> list[str]:
        """
        Remove strings containing HTML tags and remove unwanted tagged content.
        """
        cleaned = []

        for text in strings:
            if self.html_pattern.search(text):
                continue

            text = self.remove_pattern.sub("", text)
            cleaned.append(text)

        return cleaned

    def remove_long_word_sequences(
        self,
        strings: list[str],
        min_length: int = 41,
    ) -> list[str]:
        """
        Remove contiguous non-whitespace sequences whose length is
        greater than or equal to min_length.
        """
        pattern = re.compile(rf"\S{{{min_length},}}")
        return [pattern.sub("", text) for text in strings]
    
    def clean(
        self,
        strings: list[str],
        min_length: int = 41,
    ) -> list[str]:
        """
        Run the complete cleaning pipeline.

        Steps:
        1. Remove conversations containing HTML tags.
        2. Remove unwanted tagged content.
        3. Remove long contiguous non-whitespace sequences.

        Args:
            strings: List of input strings.
            min_length: Minimum length of a contiguous non-whitespace
                sequence to remove.

        Returns:
            List of cleaned strings.
        """
        strings = self.clean_html_and_tags(strings)
        strings = self.remove_long_word_sequences(strings, min_length=min_length)
        return strings



# data = cleaner.clean(data, min_length=50)

