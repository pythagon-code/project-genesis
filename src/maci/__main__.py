from .core.cerebrum import Cerebrum
from .nn.composition import Composition


def main() -> None:
    lst = []

    for i in range(1):
        lst.append(Composition(10, {}))

    #cerebrum = Cerebrum()
    #cerebrum.run()


if __name__ == "__main__":
    main()