

# quantum related settings and functions
# including entanglement state type, operations, measurement accuracy, etc.


from enum import Enum
from collections.abc import Iterable


class EntType(Enum):
    # input state type of quantum operations
    DEPHASED = 1
    WERNER = 2

class MeasureAccu:
    # measurement accuracy parameters
    def __init__(self, params: 'Iterable[float]') -> None:
        """
        params: accuracy of:  one-qubit, two-qubit, and Bell state measurement
        """
        assert len(params) == 3
        for param in params:
            assert 0 <= param <= 1
        self.params = params

        self.p = params[0]*params[1]
        self.eta = params[2]

        if self.params[0] == 1 and self.params[1] == 1 and self.params[2] == 1:
            self.noisy = False
        else:
            self.noisy = True

class Operation:
    def __init__(self, ent_type: EntType, ms_accu: MeasureAccu) -> None:
        self.ent_type = ent_type
        self.ms_accu = ms_accu

        # self.noisy = ms_accu.noisy
        # self.p = ms_accu.p
        # self.eta = ms_accu.eta

        if self.ent_type == EntType.DEPHASED:
            assert self.noisy == False, \
                "Noisy measurement not supported in dephased system"

    def _swap_dephased(self, f1, f2):
        f = f1*f2 + (1-f1)*(1-f2)
        return f
    
    def _swap_werner(self, f1, f2):
        p = self.ms_accu.p
        eta = self.ms_accu.eta
        
        f = 1/4 + (1/36) * p * (4*eta**2-1) * (4*f1 - 1) * (4*f2 - 1)
        return f

    def swap(self, f1, f2):
        if self.ent_type == EntType.DEPHASED:
            return self._swap_dephased(f1, f2)
        elif self.ent_type == EntType.WERNER:
            return self._swap_werner(f1, f2, error_mode=True)
        else:
            raise ValueError('ent_type must be DEPHASED or WERNER')
    
    def _swap_dephased_grad(f1, f2, partial=1):
        if partial == 1:
            grad = f2 - (1-f2)
        elif partial == 2:
            grad = f1 - (1-f1)
        else:
            raise ValueError('partial must be 1 or 2')
        return grad

    def _swap_werner_grad(self, f1, f2, partial=1):
        p = self.ms_accu.p
        eta = self.ms_accu.eta

        if partial == 1:
            grad = (1/9) * p * (4*eta**2-1) * (4*f2 - 1)
        elif partial == 2:
            grad = (1/9) * p * (4*eta**2-1) * (4*f1 - 1)
        else:
            raise ValueError('p must be 1 or 2')
        return grad

    def swap_grad(self, f1, f2, partial):
        if self.ent_type == EntType.DEPHASED:
            return self._swap_dephased_grad(f1, f2, partial)
        elif self.ent_type == EntType.WERNER:
            return self._swap_werner_grad(f1, f2, partial)
        else:
            raise ValueError('ent_type must be DEPHASED or WERNER')

    def _purify_dephased(self, f1, f2):
        f = (f1 * f2) / (f1 * f2 + (1 - f1) * (1 - f2))
        return f

    def _purify_werner(self, f1, f2):
        e1, e2 = (1-f1)/3, (1-f2)/3
        p = self.ms_accu.p
        eta = self.ms_accu.eta

        nume = (eta**2 + (1-eta)**2)*(f1*f2 + e1*e2) + 2*eta*(1-eta)*(f1*e2 + f2*e1) + (1-p**2)/(8*p**2)
        deno = (eta**2 + (1-eta)**2)*(f1*f2 + f1*e2 + f2*e1 + 5*e1*e2) + 2*eta*(1-eta)*(2*f1*e2 + 2*f2*e1 + 4*e1*e2) + (1-p**2)/(2*p**2)
        
        f = nume / deno
        return f

    def purify(self, f1, f2):
        if self.ent_type == EntType.DEPHASED:
            return self._purify_dephased(f1, f2)
        elif self.ent_type == EntType.WERNER:
            return self._purify_werner(f1, f2)
        else:
            raise ValueError('ent_type must be DEPHASED or WERNER')
    
    def _purify_dephased_grad(f1, f2, partial=1):
        deno = ((f1 * f2 + (1 - f1) * (1 - f2)))**2
        if partial == 1:
            nume = f2 * (f1 * f2 + (1 - f1) * (1 - f2)) - f1*f2 * (f2 - (1 - f2))
        elif partial == 2:
            nume = f1 * (f1 * f2 + (1 - f1) * (1 - f2)) - f1*f2 * (f1 - (1 - f1))
        else:
            raise ValueError('partial must be 1 or 2')
        
        grad = nume / deno
        return grad
    
    def _purify_werner_grad(self, f1, f2, partial=1):
        e1, e2 = (1-f1)/3, (1-f2)/3
        p = self.ms_accu.p
        eta = self.ms_accu.eta

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

    def purify_grad(self, f1, f2, partial=1):
        if self.ent_type == EntType.DEPHASED:
            return self._purify_dephased_grad(f1, f2, partial)
        elif self.ent_type == EntType.WERNER:
            return self._purify_werner_grad(f1, f2, partial)
        else:
            raise ValueError('ent_type must be DEPHASED or WERNER')


if __name__ == '__main__':
    op = Operation(EntType.WERNER, MeasureAccu((0.999, 0.999, 0.999)))
    print(op.purify(0.5, 0.5))



