

# quantum related settings and functions
# including entanglement state type, operations, measurement accuracy, etc.


from enum import Enum
from collections.abc import Iterable
from copy import deepcopy
from typing import NewType


FidType = NewType('FidType', float)
ProbType = NewType('ProbType', float)
BudgetType = NewType('BudgetType', int)
ExpCostType = NewType('ExpBudgetType', float)
OpResultType = NewType('OpResultType', tuple[FidType, ProbType])


class EntType(Enum):
    # input state type of quantum operations
    BINARY = 1
    WERNER = 2


class OpType(Enum):
    # quantum operations
    SWAP = 1
    PURIFY = 2


class HWParam:
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


        if self.params[0] == self.params[1] == self.params[2] == 1:
            self.noisy = False
        else:
            self.noisy = True


# perfect hardware
HWP = HWParam((1, 1, 1, 1))
# perfect hardware with linear optics swapping, low success rate
HWP_LOSL = HWParam((1, 1, 1, 0.5))
# perfect hardware with linear optics swapping, medium success rate
HWP_LOSM = HWParam((1, 1, 1, 0.625))
# perfect hardware with linear optics swapping, high success rate
HWP_LOSH = HWParam((1, 1, 1, 0.75))
HWP_DS = HWParam((1, 1, 1, 0.9999))
# noisy hardware, high accuracy
HWH = HWParam((0.99999, 0.99999, 0.9999, 0.75))
# noisy hardware, medium accuracy
HWM = HWParam((0.9999, 0.9999, 0.999, 0.625))
# noisy hardware, low accuracy
HWL = HWParam((0.999, 0.999, 0.99, 0.5))

# noisy hardware, high accuracy, deterministic swappin
HWH_D = HWParam((0.99999, 0.99999, 0.9999, 0.9999))
# noisy hardware, medium accuracy, deterministic swapping
HWM_D = HWParam((0.9999, 0.9999, 0.999, 0.999))
# noisy hardware, low accuracy, deterministic swapping
HWL_D = HWParam((0.999, 0.999, 0.99, 0.99))

HWD = HWParam((0.99999, 0.99999, 0.9999, 0.9999))

class Gate:
    def __init__(self, 
            ent_type: EntType = EntType.BINARY,
            hdw: HWParam = HWP,
        ) -> None:
        self.ent_type = ent_type
        self.hw = deepcopy(hdw)

        # self.noisy = ms_accu.noisy
        # self.p = ms_accu.p
        # self.eta = ms_accu.eta

        if self.ent_type == EntType.BINARY:
            assert self.hw.noisy == False, \
                "Noisy measurement not supported in dephased system"

    def swap(self, f1, f2) -> 'OpResultType':
        if self.ent_type == EntType.BINARY:
            f = self._swap_dephased(f1, f2)
        elif self.ent_type == EntType.WERNER:
            f = self._swap_werner(f1, f2)
        else:
            raise ValueError('ent_type must be DEPHASED or WERNER')
        
        return f, self.hw.prob_swap
    
    def seq_swap(self, fids):
        f = fids[0]
        prob = 1
        for i in range(1, len(fids)):
            f, p = self.swap(f, fids[i])
            prob *= p
        return f, prob
    
    def seq_swap_grad(self, fids, partial):
        prod = 1
        for i in range(len(fids)):
            if i == partial:
                pass
            else:
                if self.ent_type == EntType.BINARY:
                    prod *= fids[i] - 1/2
                elif self.ent_type == EntType.WERNER:
                    prod *= fids[i] - 1/4
                else:
                    raise ValueError('ent_type must be DEPHASED or WERNER')
                
        return prod

    def swap_grad(self, f1, f2, partial,) -> 'tuple[FidType, ExpCostType, ExpCostType]':
        if self.ent_type == EntType.BINARY:
            grad_f = self._swap_dephased_grad(f1, f2, partial)
        elif self.ent_type == EntType.WERNER:
            grad_f = self._swap_werner_grad(f1, f2, partial)
        else:
            raise ValueError('ent_type must be DEPHASED or WERNER')
        grad_cn = 1/self.hw.prob_swap
        return grad_f, grad_cn, 0

    def balanced_swap(self, fids: 'list[float]') -> 'OpResultType':
        assert len(fids) > 0
        next_fids = []
        prob = 1
        while len(fids) > 1:
            while len(fids) > 1:
                f1, f2 = fids.pop(0), fids.pop(0)
                f, p = self.swap(f1, f2)
                next_fids.append(f)
                prob *= p

            fids = next_fids
            next_fids = []

        return fids[0], prob

    def purify(self, f1, f2) -> 'OpResultType':
        if self.ent_type == EntType.BINARY:
            return self._purify_dephased(f1, f2)
        elif self.ent_type == EntType.WERNER:
            return self._purify_werner(f1, f2)
        else:
            raise ValueError('ent_type must be DEPHASED or WERNER')
    
    def seq_purify(self, fids) -> 'OpResultType':
        f = fids[0]
        prob = 1
        for i in range(1, len(fids)):
            f, p = self.purify(f, fids[i])
            prob *= p
        return f, prob
    
    def balanced_purify(self, fids: 'list[float]') -> 'OpResultType':
        assert len(fids) > 0
        next_fids = []
        prob = 1
        while len(fids) > 1:
            while len(fids) > 1:
                fid1, fid2 = fids.pop(0), fids.pop(0)
                f, p = self.purify(fid1, fid2)
                next_fids.append(f)
                prob *= p

            if len(fids) == 1:
                next_fids.append(fids.pop(0))

            fids = next_fids
            next_fids = []

        return fids[0], prob



    def purify_grad(self, f1, f2, n1, n2, partial,) -> 'tuple[FidType, ExpCostType, ExpCostType]':
        if self.ent_type == EntType.BINARY:
            grad_f, grad_cn, grad_cf = self._purify_dephased_grad(f1, f2, n1, n2, partial)
        elif self.ent_type == EntType.WERNER:
            grad_f, grad_cn, grad_cf = self._purify_werner_grad(f1, f2, n1, n2, partial)
        else:
            raise ValueError('ent_type must be DEPHASED or WERNER')
        
        return grad_f, grad_cn, grad_cf

    def _swap_dephased(self, f1, f2) -> FidType:
        f = f1*f2 + (1-f1)*(1-f2)
        return f
    
    def _swap_werner(self, f1, f2) -> FidType:
        p = self.hw.p
        eta = self.hw.eta
        
        f = 1/4 + (1/36) * p * (4*eta**2-1) * (4*f1 - 1) * (4*f2 - 1)
        return f

    def _swap_dephased_grad(self, f1, f2, partial) -> FidType:
        if partial == 1:
            grad_f = f2 - (1-f2)
        elif partial == 2:
            grad_f = f1 - (1-f1)
        else:
            raise ValueError('partial must be 1 or 2')
        return grad_f

    def _swap_werner_grad(self, f1, f2, partial) -> FidType:
        p = self.hw.p
        eta = self.hw.eta

        if partial == 1:
            grad = (1/9) * p * (4*eta**2-1) * (4*f2 - 1)
        elif partial == 2:
            grad = (1/9) * p * (4*eta**2-1) * (4*f1 - 1)
        else:
            raise ValueError('p must be 1 or 2')
        return grad

    def _purify_dephased(self, f1, f2) -> 'OpResultType':
        prob = f1 * f2 + (1 - f1) * (1 - f2)
        f = (f1 * f2) / prob
        return f, prob

    def _purify_werner(self, f1, f2) -> 'OpResultType':
        e1, e2 = (1-f1)/3, (1-f2)/3
        p = self.hw.p
        eta = self.hw.eta

        nume = (eta**2 + (1-eta)**2)*(f1*f2 + e1*e2) + 2*eta*(1-eta)*(f1*e2 + f2*e1) + (1-p**2)/(8*p**2)
        deno = (eta**2 + (1-eta)**2)*(f1*f2 + f1*e2 + f2*e1 + 5*e1*e2) + 2*eta*(1-eta)*(2*f1*e2 + 2*f2*e1 + 4*e1*e2) + (1-p**2)/(2*p**2)
        
        f = nume / deno
        prob = deno * p**2
        return f, prob

    def _purify_dephased_grad(self, f1, f2, n1, n2, partial) \
            -> 'tuple[FidType, ExpCostType, ExpCostType]':
        deno = ((f1 * f2 + (1 - f1) * (1 - f2)))**2
        if partial == 1:
            nume_f = f2 * (f1 * f2 + (1 - f1) * (1 - f2)) - f1*f2 * (f2 - (1 - f2))
            grad_cf = (n1+n2)*(-1/deno)*(2*f2 - 1)
        elif partial == 2:
            nume_f = f1 * (f1 * f2 + (1 - f1) * (1 - f2)) - f1*f2 * (f1 - (1 - f1))
            grad_cf = (n1+n2)*(-1/deno)*(2*f1 - 1)
        else:
            raise ValueError('partial must be 1 or 2')
        
        grad_cn = 1/(f1 * f2 + (1 - f1) * (1 - f2))
        grad_f = nume_f / deno
        return grad_f, grad_cn, grad_cf
    
    def _purify_werner_grad(self, f1, f2, n1, n2, partial) \
            -> 'tuple[FidType, ExpCostType, ExpCostType]':
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
            grad_cf = (n1+n2)*(-1/deno)*(p**2*p_deno_purify_1)
        elif partial == 2:
            p_nume_purify_2 = eta_m*(f1 - (1/3)*e1) + 2*eta*(1-eta)*(e1-1/3*f1)
            p_deno_purify_2 = eta_m*(f1 + e1 -1/3*f1 - (5/3)*e1) + 2*eta*(1-eta)*(2*e1 - 2/3*f1 - 4/3*e1)
            nume = p_nume_purify_2 * deno_purify - nume_purify * p_deno_purify_2
            grad_cf = (n1+n2)*(-1/deno)*(p**2*p_deno_purify_2)
        else:
            raise ValueError('p must be 1 or 2')

        grad_f = nume / deno
        grad_cn = 1/deno_purify
        return grad_f, grad_cn, grad_cf


# Dephased operation, perfect
GDP = Gate(EntType.BINARY, HWP)
# Dephased operation, perfect, with linear optics swapping, low success rate
GDP_LOSL = Gate(EntType.BINARY, HWP_LOSL)
# Dephased operation, perfect, with linear optics swapping, medium success rate
GDP_LOSM = Gate(EntType.BINARY, HWP_LOSM)
# Dephased operation, perfect, with linear optics swapping, high success rate
GDP_LOSH = Gate(EntType.BINARY, HWP_LOSH)

GDH_D = Gate(EntType.BINARY, HWP_DS)

# Werner operation, perfect
GWP = Gate(EntType.WERNER, HWP)
# Werner operation, noisy, high accuracy
GWH = Gate(EntType.WERNER, HWH)
# Werner operation, noisy, medium accuracy
GWM = Gate(EntType.WERNER, HWM)
# Werner operation, noisy, low accuracy
GWL = Gate(EntType.WERNER, HWL)
# Werner operation, noisy, high accuracy, with deterministic swapping
GWD = Gate(EntType.WERNER, HWD)

# Werner operation, noisy, high accuracy, deterministic swapping
GWH_D = Gate(EntType.WERNER, HWH_D)
# Werner operation, noisy, medium accuracy, deterministic swapping
GWM_D = Gate(EntType.WERNER, HWM_D)
# Werner operation, noisy, low accuracy, deterministic swapping
GWL_D = Gate(EntType.WERNER, HWL_D)




if __name__ == '__main__':
    gate = Gate(EntType.WERNER, HWParam((0.99, 0.99, 0.99, 0.99)))
    f1, f2 = 0.9, 0.9
    f, p = gate.purify(f1, f2)
    print(f, p)

