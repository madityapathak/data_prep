"""
Processor for static conversation data.

This module loads pre-written human conversations from a text file
and formats them according to the standard conversation format used
in the fine-tuning pipeline.
"""

import os
from typing import List, Tuple


class StaticConversationsProcessor:
    """
    Processor for loading and formatting static conversation data.
    
    Static conversations are stored in a text file where:
    - Each conversation is separated by a blank line
    - Lines starting with # are comments and ignored
    - Each turn is formatted as: SPEAKER|message
    - SPEAKER can be either USER or ASSISTANT
    """
    
    USER_TOKEN = "<|USER|>"
    ASSISTANT_TOKEN = "<|ASSISTANT|>"
    EOS_TOKEN = "<|EOS|>"
    BOS_TOKEN = "<|BOS|>"
    EOT_TOKEN = "<|EOT|>"

    def __init__(self, data_file: str = None):
        """
        Initialize the processor.
        
        Args:
            data_file: Path to the static conversations text file.
                      Defaults to 'static_conversations.txt' in project root.
        """
        if data_file is None:
            # Default to the static_conversations.txt file in the project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            data_file = os.path.join(project_root, "static_conversations.txt")
        
        self.data_file = data_file
        self.conversations = self._load_conversations()
    
    def _load_conversations(self) -> List[List[Tuple[str, str]]]:
        """
        Load conversations from the text file.
        
        Returns:
            List of conversations, where each conversation is a list of
            (speaker, message) tuples.
        """
        if not os.path.exists(self.data_file):
            print(f"Warning: Static conversations file not found: {self.data_file}")
            return []
        
        conversations = []
        current_conversation = []
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        # If we have a conversation in progress, save it
                        if current_conversation:
                            conversations.append(current_conversation)
                            current_conversation = []
                        continue
                    
                    # Parse the line
                    if '|' in line:
                        parts = line.split('|', 1)
                        if len(parts) == 2:
                            speaker, message = parts
                            speaker = speaker.strip().upper()
                            message = message.strip()
                            
                            # Validate speaker
                            if speaker in ['USER', 'ASSISTANT']:
                                current_conversation.append((speaker, message))
                            else:
                                print(f"Warning: Unknown speaker '{speaker}' in line: {line[:50]}...")
                
                # Don't forget the last conversation if file doesn't end with blank line
                if current_conversation:
                    conversations.append(current_conversation)
        
        except Exception as e:
            print(f"Error loading static conversations: {e}")
            return []
        
        return conversations
    
    def _format_conversation(self, conversation: List[Tuple[str, str]]) -> List[str]:
        """
        Format a conversation into the standard token format.
        
        Args:
            conversation: List of (speaker, message) tuples
            
        Returns:
            List of formatted message strings with speaker tokens and EOS tokens
        """
        formatted_messages = []
        
        for speaker, message in conversation:
            if speaker == 'USER':
                speaker_token = self.USER_TOKEN
            else:  # ASSISTANT
                speaker_token = self.ASSISTANT_TOKEN
            
            formatted_messages.append(
                f"{speaker_token}{message}{self.EOS_TOKEN}"
            )
        
        return formatted_messages
    
    @staticmethod
    def merge_conversation_tokens(conversations: List[List[str]]) -> List[str]:
        """
        Merge token lists into single strings for each conversation.
        
        Args:
            conversations: List of conversations, each being a list of formatted messages
            
        Returns:
            List of merged conversation strings
        """
        merged_conversations = []
        
        for conversation in conversations:
            merged_conversations.append("".join(conversation))
        
        return merged_conversations
    
    def build_dataset(self, train_split: float = 0.9) -> Tuple[List[str], List[str]]:
        """
        Build train and validation datasets from static conversations.
        
        Args:
            train_split: Fraction of data to use for training (default: 0.9)
            
        Returns:
            Tuple of (train_data, validation_data)
        """
        if not self.conversations:
            print("Warning: No conversations loaded from static file")
            return [], []
        
        # Format all conversations
        formatted_conversations = []
        for conversation in self.conversations:
            formatted_conversations.append(
                self._format_conversation(conversation)
            )
        
        # Merge tokens into strings
        formatted_conversations = self.merge_conversation_tokens(
            formatted_conversations
        )
        
        # Split into train and validation
        split_idx = int(len(formatted_conversations) * train_split)
        train_data = formatted_conversations[:split_idx]
        validation_data = formatted_conversations[split_idx:]
        
        return train_data, validation_data
    
    def get_train_data(self) -> List[str]:
        """
        Get training data.
        
        Returns:
            List of formatted training conversations
        """
        train_data, _ = self.build_dataset()
        return train_data
    
    def get_validation_data(self) -> List[str]:
        """
        Get validation data.
        
        Returns:
            List of formatted validation conversations
        """
        _, validation_data = self.build_dataset()
        return validation_data
    
    def get_stats(self) -> dict:
        """
        Get statistics about the loaded conversations.
        
        Returns:
            Dictionary with statistics
        """
        total_conversations = len(self.conversations)
        total_turns = sum(len(conv) for conv in self.conversations)
        
        train_data, validation_data = self.build_dataset()
        
        return {
            'total_conversations': total_conversations,
            'total_turns': total_turns,
            'train_conversations': len(train_data),
            'validation_conversations': len(validation_data),
            'avg_turns_per_conversation': total_turns / total_conversations if total_conversations > 0 else 0
        }


# Example usage and testing
if __name__ == "__main__":
    processor = StaticConversationsProcessor()
    
    # Get statistics
    stats = processor.get_stats()
    print("Static Conversations Statistics:")
    print(f"  Total conversations: {stats['total_conversations']}")
    print(f"  Total turns: {stats['total_turns']}")
    print(f"  Average turns per conversation: {stats['avg_turns_per_conversation']:.2f}")
    print(f"  Train conversations: {stats['train_conversations']}")
    print(f"  Validation conversations: {stats['validation_conversations']}")
    
    # Get sample data
    train_data = processor.get_train_data()
    validation_data = processor.get_validation_data()
    
    if train_data:
        print("\nSample conversation (first 200 chars):")
        print(train_data[0][:200] + "...")
