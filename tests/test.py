
from src.physical.quantum import GWP, GWH_D

if __name__ == "__main__":
    
    f_e = 0.85
    
    N = 8
    f = f_e
    for i in range(N // 2):
        f, _ = GWP.purify(f, f)
        # f, _ = GWH_D.purify(f, f)
        print(f)

    # for i in range(9-1):
    #     f, _ = GWP.swap(f, f)
    # print(f)
