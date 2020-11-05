from runners.STL_runner import STLRunner
from runners.MTL_runner import MTLRunner
from argparse import ArgumentParser

parser = ArgumentParser(description="A multitask learner")
parser.add_argument("model_type", choices=["mtl", "stl"], help="")


def main():
    args = parser.parse_args()
    learner = MTLRunner() if args.model_type == "mtl" else STLRunner()
    learner.run()


if __name__ == "__main__":
    main()
