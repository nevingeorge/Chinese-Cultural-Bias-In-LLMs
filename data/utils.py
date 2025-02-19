def print_nested_dict(d, indent=0):
    """Recursively print a nested dictionary with proper indentation."""
    space = " " * indent
    if isinstance(d, dict):
        for key, value in d.items():
            print(f"{space}{key}:")
            print_nested_dict(value, indent + 4)
    elif isinstance(d, list):
        for i, item in enumerate(d):
            print(f"{space}- [{i}]")
            print_nested_dict(item, indent + 4)
    else:
        print(f"{space}{d}")