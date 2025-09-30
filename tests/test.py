
from src.physical.quantum import GWP

if __name__ == "__main__":
    
    f_e = 0.8
    
    N = 100
    f = f_e
    for i in range(N // 2):
        f, _ = GWP.purify(f, f)
        print(f)
