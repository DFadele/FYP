from dataclasses import dataclass
import numpy as np
import do_mpc


print('Numpy Version:',np.__version__)
print('Do MPC Version:',do_mpc.__version__)

model = do_mpc.model.Model(model_type='continuous')
X = model.set_variable(var_type='_x',var_name='X',shape=(9,1))
U = model.set_variable(var_type='_u',var_name='U',shape=(4,1))


@dataclass
class DroneModel:
    #constants
    g = 9.81
    Xu = 0.5
    Xq = 0.01
    Yv = 0.065
    Xp = 0.032
    Zw = -0.987
    Lv = 0.098
    Lp = 0.321
    Mu = 0.0767
    Mq = 0.765
    Nr = 0.986
    Z_fz = 0.586
    L_mx = 0.456
    M_my = -0.765
    N_mz = -0.081

    A = np.array([
        [Xu,0,0,0,Xq,0,0,-g,0],
        [0,Yv,0,Xp,0,0,g,0,0],
        [0,0,Zw,0,0,0,0,0,0],
        [0,Lv,0,Lp,0,0,0,0,0],
        [Mu,0,0,0,Mq,0,0,0,0],
        [0,0,0,0,0,Nr,0,0,0],
        [0,0,0,1,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,1,0,0,0]
    ])

    B = np.array([
        [0,0,0,0],
        [0,0,0,0],
        [Z_fz,0,0,0],
        [0,L_mx,0,0],
        [0,0,M_my,0],
        [0,0,0,N_mz],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0]
    ])

    # states
    u: float = 0.0
    v: float = 0.0
    w: float = 0.0
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0




    def get_states(self):
        return np.array([[self.u],[self.v],[self.w],[self.p],[self.q],[self.r]])
    

    def open_loop(self):
        pass



    
