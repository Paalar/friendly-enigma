def green_text(text):
    TGREEN = "\033[32m"
    ENDC = "\033[0m"
    return f"{TGREEN} {text} {ENDC}"


def red_text(text):
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    return f"{FAIL} {text} {ENDC}"


def print_git_diff(pred, expected):
    for index, tensor in enumerate(pred):
        expected_val = round(expected[index].item())
        val = round(tensor.item())

        if val == expected_val:
            print(green_text(val) + " | " + green_text(expected_val))
        else:
            print(red_text(val) + " | " + green_text(expected_val))


def create_cond_print(condition):
    return lambda *args: condition and print(*args)
