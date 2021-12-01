import numpy as np
import gym
from gym import error, spaces, utils
from scipy import linalg as SciLA
from numpy import linalg as LA
from gym.utils import seeding
import random

import operator
from functools import reduce
from itertools import product

class cutEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        high = np.ones(90)
        low = np.zeros(90)
        self.state=np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]])
        self.t=0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        # self.l=[0,1,2,3,4,5,6,7,8,9,10]
        pass

    def step(self, action):
        done=False
        t=self.t
        # l=self.l
        # l[t]=action
        state=self.state
        R=0
        # l=self.l
        if action!=0:
            a=state[int(t/15)][t%15]
            b=state[int(t/15)][(t+int(action))%15]
            state[int(t/15)][(t+int(action))%15]=a
            state[int(t/15)][t % 15]=b
        t += 1
        if t==90:
            done=True

            N = 6
            beta = 6
            n_trotter = 6


            I = np.eye(2)
            X = np.array([[0., 1.], [1., 0.]], dtype=complex)
            Y = 1j * np.array([[0., -1.], [1., 0.]])
            Z = np.array([[1., 0.], [0., -1.]], dtype=complex)

            def kron_list(oprt_list):
                ret = np.eye(1)
                for oprt in oprt_list:
                    ret = np.kron(ret, oprt)
                return ret

            def MaxCut_Hamiltonian():

                np.random.seed(2)
                params = np.random.uniform(-1, 1, 15)

                pos_list = []
                for _0 in range(5):
                    for _1 in range(_0 + 1, 6):
                        pos_list.append([_0, _1])

                h_list = [[params[_] * Z, Z] for _ in range(15)]

                H = 0
                for ind0 in range(15):
                    I_list = [I for _ in range(6)]
                    for ind1 in range(2):
                        I_list[pos_list[ind0][ind1]] = h_list[ind0][ind1]
                    H = H + kron_list(I_list)

                return h_list, pos_list, H

            def initial_state(N):
                ret = np.array([1 / np.sqrt(2 ** N) for _ in range(2 ** N)])
                return ret

            def qite_step(psi, delta_t, h, h_pos):

                I_list = [I for _ in range(6)]
                for _ in range(2):
                    I_list[h_pos[_]] = h[_]

                h_glob = kron_list(I_list)
                c = 1 - 2 * delta_t * np.conj(psi).dot(h_glob).dot(psi)

                pauli_list = []
                for p0 in [I, X, Y, Z]:
                    for p1 in [I, X, Y, Z]:
                        I_list = [I for _ in range(6)]
                        I_list[h_pos[0]] = p0
                        I_list[h_pos[1]] = p1
                        pauli_list.append(kron_list(I_list))

                S = np.array([[np.conj(psi).dot(p1).dot(p0).dot(psi)
                               for p0 in pauli_list] for p1 in pauli_list]).real
                b_p = np.array([1 / delta_t * (1 / np.sqrt(c) - 1) * np.conj(psi).dot(p1).dot(psi)
                                - 1 / np.sqrt(c) * np.conj(psi).dot(p1).dot(h_glob).dot(psi) for p1 in pauli_list])
                b = (1j * np.conj(b_p) - 1j * b_p).real

                a_list = LA.lstsq(S + np.transpose(S), -b, rcond=-1)[0]

                Hamiltonian_A = 0
                for _ in range(len(a_list)):
                    Hamiltonian_A = Hamiltonian_A + a_list[_] * pauli_list[_]
                psi_out = SciLA.expm(- 1j * delta_t * Hamiltonian_A).dot(psi)

                return psi_out

            def QITE(qite_layout):
                psi = initial_state(N)
                for ITR in range(n_trotter):
                    for h_index in range(n_term):
                        pos = qite_layout[ITR][h_index]
                        psi = qite_step(psi, beta / n_trotter, h_list[pos], pos_list[pos])
                ret = np.conj(psi).dot(H).dot(psi).real
                return ret
            h_list, pos_list, H = MaxCut_Hamiltonian()
            n_term = len(h_list)

        if t==90:
            qite_layout = state.tolist()



            R=-QITE(qite_layout)/3.5757973907199148
            if R<=1:
                R=-1
            else:
                R=-1/np.log10(R-1)
            done=True

        self.state=state
        self.t = t
        state=state/14.0
        return np.array(reduce(operator.add, state.tolist())),float(R),done,{}


    def reset(self):

        self.state=np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]])
        self.t=0
        state=self.state/14.0
        return np.array(reduce(operator.add, state.tolist()))

    def render(self, mode='human', close=False):


        pass
