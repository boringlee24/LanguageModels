import os
import json
import argparse


def swap(s, c1, c2):
    t = list(s)
    for c in s:
        if c == c1:
            t 

def load_config(filename):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


def save_config(config, filename):
    with open(filename, "w") as f:
        json.dump(config, f)


def add_command_line_args(config, *clargs):
    for args in clargs:
        if hasattr(args, "__dict__"):
            args = vars(args)
        for k in args:
            key = k.upper()
            value = args[k]
            if (key in config) and bool(config[key]):
                print("Overwriting config parameter {} with command line argument {}".format(key, value))
            config[key] = value


def handle_unknown_args(argument_strings):
    formatted = []
    for s in argument_strings:
        split = s.split("=")
        formatted.extend(split)
    tuples = []
    for s in formatted:
        prefix = s[:2]
        suffix = s[2:]
        if prefix == "--":
            tuples.append([suffix, ""])
        else:
            current = tuples[-1][1]
            if len(current):
                tuples[-1][1] = " ".join([current, s])
            else:
                tuples[-1][1] = s
    args = {}
    for t in tuples:
        k = t[0].replace("-", "_")
        args[k] = True if t[1] == "" else t[1]
    return args


def test_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config filename for HF training parameters")
    arglist = [
        "--config",
        "default.json",
        "--an-arg",
        "pluto",
        "--other-arg=venus",
        "--a-flag",
        "--mars",
        "deimos",
        "phobos",
        "--giants",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
        "--learning-rate",
        "2e-5",
    ]
    args, unknown = parser.parse_known_args(arglist)
    config = load_config(args.config)
    unknown = handle_unknown_args(unknown)
    add_command_line_args(config, args, unknown)
    print(config)
    save_config(config, "test_out.json")


if __name__ == "__main__":
    test_config()
