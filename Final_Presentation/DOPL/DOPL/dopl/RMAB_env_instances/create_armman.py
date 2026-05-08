import numpy as np
np.random.seed(0)


class armman:
    def __init__(self,p1,p2,p3,p4,p5,p6):
        self.n_actions = 2
        self.n_states = 3
        self.p1=p1
        self.p2=p2
        self.p3=p3
        self.p4=p4
        self.p5=p5
        self.p6=p6


    def reward(self):
        R = np.zeros((self.n_states, self.n_actions))
        R[0,:] = 1
        R[1,:] = 0.5
        R[2,:] = 0
        return R
    
    def Probability(self):
        P = np.zeros((self.n_states,self.n_states,self.n_actions))

        P[0,:,0] = [self.p1,0.95-self.p1,0.05]
        P[0,:,1] = [self.p2,0.95-self.p2,0.05]

        P[1,:,0] = [0.05,0.95-self.p3,self.p3]
        P[1,:,1] = [self.p4,0.95-self.p4,0.05]

        P[2,:,0] = [0.05,0.95-self.p5,self.p5]
        P[2,:,1] = [0.05,0.95-self.p6,self.p6]

        return P

def save_matrices(P, R, arm_type):
    np.save(f"armman_arm_type_{arm_type}_transitions.npy", P)
    np.save(f"armman_arm_type_{arm_type}_rewards.npy", R)
    print(f"Matrices for arm_type_{arm_type} saved successfully.")


def typeA():

    p1 = np.random.uniform(0.05,0.95)
    p2 = np.random.uniform(0.05,0.95)
    p3 = np.random.uniform(0.45,0.95)
    p4 = np.random.uniform(0.45,0.95)
    p5 = np.random.uniform(0.35,0.85)
    p6 = np.random.uniform(0.35,0.85)

    return p1,p2,p3,p4,p5,p6

def typeB():

    p1 = np.random.uniform(0.05,0.95)
    p2 = np.random.uniform(0.05,0.95)
    p3 = np.random.uniform(0.35,0.85)
    p4 = np.random.uniform(0.15,0.65)
    p5 = np.random.uniform(0.35,0.85)
    p6 = np.random.uniform(0.35,0.85)

    return p1,p2,p3,p4,p5,p6


def typeC():

    p1 = np.random.uniform(0.05,0.95)
    p2 = np.random.uniform(0.05,0.95)
    p3 = np.random.uniform(0.35,0.85)
    p4 = np.random.uniform(0.05,0.50)
    p5 = np.random.uniform(0.35,0.85)
    p6 = np.random.uniform(0.35,0.85)

    return p1,p2,p3,p4,p5,p6

if __name__=="__main__":
    
    for i in range(1,11):
        if(i<=1):
            p1,p2,p3,p4,p5,p6 = typeA()
            A = armman(p1,p2,p3,p4,p5,p6)
            save_matrices(A.Probability(),A.reward(),i)
        elif(i<=3):
            p1,p2,p3,p4,p5,p6 = typeC()
            A = armman(p1,p2,p3,p4,p5,p6)
            save_matrices(A.Probability(),A.reward(),i)
        else:
            p1,p2,p3,p4,p5,p6 = typeB()
            A = armman(p1,p2,p3,p4,p5,p6)
            save_matrices(A.Probability(),A.reward(),i)



            









