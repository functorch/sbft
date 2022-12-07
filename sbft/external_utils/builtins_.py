import typing as t

from omegaconf import DictConfig


def nested_dict_contains_dot_key(
    dictionary: dict[str, t.Any] | DictConfig, dot_key: str
) -> bool:
    """Return True if the dictionary contains dot_key in the exact nested position.

    Args:
    -----
        dictionary: The dictionary to check.
        dot_key: The dot key to check. For example, "a.b.c" will check if the dictionary
            at the key "a" contains a dict as value for the key "b" which contains the
            key "c".

    Returns:
    --------
        True only if the dictionary contains the dot key in the exact nested position

    Examples:
    ---------------
    >>> assert nested_dict_contains_dot_key(dict(a=0, b=dict(c=0, d=0)), "b.d")
    >>> assert not nested_dict_contains_dot_key(dict(a=0, b=dict(c=0, d=0)), "d")
    """
    if "." not in dot_key:
        return dot_key in dictionary
    key_to_check, _, rem_dot_key = dot_key.partition(".")
    if key_to_check in dictionary:
        return nested_dict_contains_dot_key(dictionary[key_to_check], rem_dot_key)
    else:
        return False
