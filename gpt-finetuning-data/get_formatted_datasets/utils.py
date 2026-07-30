TARGETS = {
    "ultrachat":   70_000_000,
    "openorca":    65_000_000,
    "oasst":       40_000_000,
    "dailydialog": 15_000_000,
    "dolly":       10_000_000,
}

SEED = 65

en_filter = lambda x: x["lang"] == "en"


def dfs(node, path, message_role_dict, children, conversations):
    path.append(node)

    # Save whenever we reach an assistant
    if message_role_dict[node]["role"] == "assistant":
        conversations.append(path.copy())

    # Traverse children
    for child in children.get(node, []):
        dfs(child, path, message_role_dict, children, conversations)

    path.pop()

