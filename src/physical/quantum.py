

# quantum related settings and functions
# including entanglement state type, operations, measurement accuracy, etc.


from enum import Enum
from collections.abc import Iterable
from copy import deepcopy


class EntType(Enum):
    # input state type of quantum operations
    DEPHASED = 1
    WERNER = 2


class OpType(Enum):
    # quantum operations
    SWAP = 1
    PURIFY = 2


class HW:
    # Hardware setting
    def __init__(self, params: 'Iterable[float]') -> None:
        """
        params: accuracy of:  one-qubit, two-qubit, and Bell state measurement
        """
        assert len(params) == 4
        for param in params:
            assert 0 <= param <= 1
        self.params = deepcopy(params)
        self.accu_1qubit = self.params[0]
        self.accu_2qubit = self.params[1]
        self.accu_BSM = self.params[2]
        self.prob_swap = self.params[3]

        self.p = self.params[0] * self.params[1]
        self.eta = self.params[2]


        if self.params[0] == self.params[1] == self.params[2] == self.params[3] == 1:
            self.noisy = False
        else:
            self.noisy = True


# perfect hardware
HWP = HW((1, 1, 1, 1))
# noisy hardware, high accuracy
HWH = HW((0.999, 0.999, 0.999, 0.99))
# noisy hardware, medium accuracy
HWM = HW((0.99, 0.99, 0.99, 0.9))
# noisy hardware, low accuracy
HWL = HW((0.9, 0.9, 0.9, 0.5))


class Operation:
    def __init__(self, 
            ent_type: EntType = EntType.DEPHASED,
            hdw: HW = HWP,
        ) -> None:
        self.ent_type = ent_type
        self.hw = deepcopy(hdw)

        # self.noisy = ms_accu.noisy
        # self.p = ms_accu.p
        # self.eta = ms_accu.eta

        if self.ent_type == EntType.DEPHASED:
            assert self.hw.noisy == False, \
                "Noisy measurement not supported in dephased system"

    def swap(self, f1, f2) -> 'tuple[float, float]':
        if self.ent_type == EntType.DEPHASED:
            f = self._swap_dephased(f1, f2)
        elif self.ent_type == EntType.WERNER:
            f = self._swap_werner(f1, f2)
        else:
            raise ValueError('ent_type must be DEPHASED or WERNER')
        
        return f, self.hw.prob_swap
    
    def seq_swap(self, fs):
        f = fs[0]
        for i in range(1, len(fs)):
            f, p = self.swap(f, fs[i])
        return f, p

    def swap_grad(self, f1, f2, partial):
        if self.ent_type == EntType.DEPHASED:
            return self._swap_dephased_grad(f1, f2, partial)
        elif self.ent_type == EntType.WERNER:
            return self._swap_werner_grad(f1, f2, partial)
        else:
            raise ValueError('ent_type must be DEPHASED or WERNER')

    def purify(self, f1, f2) -> 'tuple[float, float]':
        if self.ent_type == EntType.DEPHASED:
            return self._purify_dephased(f1, f2)
        elif self.ent_type == EntType.WERNER:
            return self._purify_werner(f1, f2)
        else:
            raise ValueError('ent_type must be DEPHASED or WERNER')
    
    def seq_purify(self, fs):
        f = fs[0]
        for i in range(1, len(fs)):
            f, p = self.purify(f, fs[i])
        return f, p

    def purify_grad(self, f1, f2, partial):
        if self.ent_type == EntType.DEPHASED:
            f = self._purify_dephased_grad(f1, f2, partial)
        elif self.ent_type == EntType.WERNER:
            f = self._purify_werner_grad(f1, f2, partial)
        else:
            raise ValueError('ent_type must be DEPHASED or WERNER')
        
        return f

    def _swap_dephased(self, f1, f2) -> float:
        f = f1*f2 + (1-f1)*(1-f2)
        return f
    
    def _swap_werner(self, f1, f2) -> float:
        p = self.hw.p
        eta = self.hw.eta
        
        f = 1/4 + (1/36) * p * (4*eta**2-1) * (4*f1 - 1) * (4*f2 - 1)
        return f

    def _swap_dephased_grad(f1, f2, partial) -> float:
        if partial == 1:
            grad = f2 - (1-f2)
        elif partial == 2:
            grad = f1 - (1-f1)
        else:
            raise ValueError('partial must be 1 or 2')
        return grad

    def _swap_werner_grad(self, f1, f2, partial) -> float:
        p = self.hw.p
        eta = self.hw.eta

        if partial == 1:
            grad = (1/9) * p * (4*eta**2-1) * (4*f2 - 1)
        elif partial == 2:
            grad = (1/9) * p * (4*eta**2-1) * (4*f1 - 1)
        else:
            raise ValueError('p must be 1 or 2')
        return grad

    def _purify_dephased(self, f1, f2) -> 'tuple[float, float]':
        prob = f1 * f2 + (1 - f1) * (1 - f2)
        f = (f1 * f2) / prob
        return f, prob

    def _purify_werner(self, f1, f2) -> 'tuple[float, float]':
        e1, e2 = (1-f1)/3, (1-f2)/3
        p = self.hw.p
        eta = self.hw.eta

        nume = (eta**2 + (1-eta)**2)*(f1*f2 + e1*e2) + 2*eta*(1-eta)*(f1*e2 + f2*e1) + (1-p**2)/(8*p**2)
        deno = (eta**2 + (1-eta)**2)*(f1*f2 + f1*e2 + f2*e1 + 5*e1*e2) + 2*eta*(1-eta)*(2*f1*e2 + 2*f2*e1 + 4*e1*e2) + (1-p**2)/(2*p**2)
        
        f = nume / deno
        prob = deno
        return f, prob

    def _purify_dephased_grad(f1, f2, partial) -> float:
        deno = ((f1 * f2 + (1 - f1) * (1 - f2)))**2
        if partial == 1:
            nume = f2 * (f1 * f2 + (1 - f1) * (1 - f2)) - f1*f2 * (f2 - (1 - f2))
        elif partial == 2:
            nume = f1 * (f1 * f2 + (1 - f1) * (1 - f2)) - f1*f2 * (f1 - (1 - f1))
        else:
            raise ValueError('partial must be 1 or 2')
        
        grad = nume / deno
        return grad
    
    def _purify_werner_grad(self, f1, f2, partial) -> float:
        e1, e2 = (1-f1)/3, (1-f2)/3
        p = self.hw.p
        eta = self.hw.eta

        eta_m = (eta**2 + (1-eta)**2)
        p_m = (1-p**2)/(p**2)
        
        nume_purify = eta_m*(f1*f2 + e1*e2) + 2*eta*(1-eta)*(f1*e2 + f2*e1) + p_m/8
        deno_purify = eta_m*(f1*f2 + f1*e2 + f2*e1 + 5*e1*e2) + 2*eta*(1-eta)*(2*f1*e2 + 2*f2*e1 + 4*e1*e2) + p_m/2
        deno = deno_purify ** 2
        if partial == 1:
            p_nume_purify_1 = eta_m*(f2 - (1/3)*e2) + 2*eta*(1-eta)*(e2-1/3*f2)
            p_deno_purify_1 = eta_m*(f2 + e2 -1/3*f2 - (5/3)*e2) + 2*eta*(1-eta)*(2*e2 - 2/3*f2 - 4/3*e2)
            nume = p_nume_purify_1 * deno_purify - nume_purify * p_deno_purify_1
        elif partial == 2:
            p_nume_purify_2 = eta_m*(f1 - (1/3)*e1) + 2*eta*(1-eta)*(e1-1/3*f1)
            p_deno_purify_2 = eta_m*(f1 + e1 -1/3*f1 - (5/3)*e1) + 2*eta*(1-eta)*(2*e1 - 2/3*f1 - 4/3*e1)
            nume = p_nume_purify_2 * deno_purify - nume_purify * p_deno_purify_2
        else:
            raise ValueError('p must be 1 or 2')

        grad = nume / deno
        return grad


# Dephased operation, perfect
DOPP = Operation(EntType.DEPHASED, HWP)
# Werner operation, perfect
WOPP = Operation(EntType.WERNER, HWP)
# Werner operation, noisy, high accuracy
WOPH = Operation(EntType.WERNER, HWH)
# Werner operation, noisy, medium accuracy
WOPM = Operation(EntType.WERNER, HWM)
# Werner operation, noisy, low accuracy
WOPL = Operation(EntType.WERNER, HWL)




if __name__ == '__main__':
    wsys = Operation(EntType.WERNER, HW((0.99, 0.99, 0.99, 0.9)))
    wsys_noiseless = Operation(EntType.WERNER, HW((1, 1, 1, 1)))
    dsys = Operation(EntType.DEPHASED, HW((1, 1, 1, 1)))
    op = wsys
    
    f1, f2, f3, f4 = 0.9, 0.9, 0.9, 0.9

    f12, p12 = op.swap(f1, f2)
    n12 = 2/p12
    f34, p34 = op.swap(f3, f4)
    n34 = 2/p34

    f, p = op.swap(f12, f34)
    n = (n12 + n34) / p
    print(f, n)

    f12, p12 = op.swap(f1, f2)
    n12 = 2/p12
    f123, p123 = op.swap(f12, f3)
    n123 = (n12 + 1) / p123
    f, p = op.swap(f123, f4)
    n = (n123 + 1) / p
    print(f, n)


    ns = 0
    p = op.hw.prob_swap
    simu = 100
    import numpy as np
    ns = np.zeros((simu, simu))
    for m in range(0, simu):
        for n in range(0, simu):
            ns[m, n] = (1-p)**m*p * (1-p)**n*p * ((m+1)*2 + (n+1)*2)
    print(sum(ns.flatten()) / p)

    en = (2/p + 2/p) / p
    print(en)

