

from quantum import EntType, MeasureAccu, Operation


def test():
    op = Operation(EntType.DEPHASED, MeasureAccu((1, 1, 1)))

    f = 0.8
    f1 = op.swap(f, f)
    f2 = op.purify(f1, f1)
    f3 = op.swap(f2, f)

    print(f1, f2, f3)


if __name__ == '__main__':
    test()
