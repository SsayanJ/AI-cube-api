import numpy as np


def flatten(t):
    return [item for sublist in t for item in sublist]


def textfile_to_list(filepath):
    with open(filepath) as f:
        lines = f.read().splitlines()
    return lines


def get_algs(filename):
    with open(filename) as f:
        lines = f.readlines()

    moves = []
    for l in lines[1:]:
        moves.append(l.strip().split(';')[-1])

    return moves


PLL_scrambles = 'data\\PLL_scrambles_unique.txt'
OLL_scrambles = 'data\\OLL_scrambles_unique.txt'
OLL_algs = 'data\\OLL algs.csv'
PLL_algs = 'data\\PLL algs.csv'


oll_moves = get_algs(OLL_algs)
pll_moves = get_algs(PLL_algs)

pll_scrambles = textfile_to_list(PLL_scrambles)
oll_scrambles = textfile_to_list(OLL_scrambles)

cube_colors = {0: "W", 1: "G", 2: "R", 3: 'B', 4: "O", 5: 'Y'}
move_list = ["R", "L", "U", "D", "F", "B", "Rp", "Lp", "Up", "Dp", "Fp", "Bp"]

SCRAMBLE_SIZE = 15
MAX_EXPLO = 50


class SpeedCube():
    def __init__(self):
        self.status = np.array([np.zeros(9)*3, np.ones(9), np.ones(9)*2, np.ones(9)*3, np.ones(9)*4,
                                np.ones(9)*5], dtype=np.int32)
        self.solved_cross = self.get_yellow_edges()
        self.solved_cross_new = self.get_yellow_edges()
        self.move_list = move_list
        self._episode_ended = False
        self._state = 0
        self._reward = 0
        self._solved_episodes = 0
        self._failed_episodes = 0

    def reset(self):
        self._state = 0
        self._reward = 0
        self._episode_ended = False
        self.status = np.array([np.zeros(9)*3, np.ones(9), np.ones(9)*2, np.ones(9)*3,
                                np.ones(9)*4, np.ones(9)*5], dtype=np.int32)
        scramble_s = np.random.randint(1, SCRAMBLE_SIZE)
        random_actions = np.random.randint(
            len(self.move_list), size=scramble_s)
        self._initial_scramble = [self.move_list[i] for i in random_actions]
        self.scramble(self._initial_scramble)
        return self.get_yellow_edges()

    def step(self, action):
        action = self.move_list[action]
        if action != "":
            getattr(self, action)()
        yellow_edges = self.get_yellow_edges()
        # nb_solved_edges = 0
        # for i in range(8, 12):
        #     if SOLVED_YELLOW_CROSS[i] == yellow_edges[i]:
        #         nb_solved_edges += 1
        # if nb_solved_edges == 4:
        #     step_reward = 300
        #     self._episode_ended = True
        # elif nb_solved_edges == 0:
        #     step_reward = -1
        # else:
        #     step_reward = nb_solved_edges

        if np.all(self.solved_cross_new == yellow_edges):
            step_reward = 10
            self._episode_ended = True
            print('cross solved:', self._state+1)
        else:
            step_reward = -1

        self._reward += step_reward
        self._state += 1

        if self._episode_ended or self._state > MAX_EXPLO:
            if self._episode_ended:
                print('episode ended:', self._state)
                self._solved_episodes += 1
            else:
                self._failed_episodes += 1
            self._episode_ended = True
        return yellow_edges, float(step_reward), self._episode_ended, 'no'

    def R(self):
        # permute corners
        self.status[2][0], self.status[2][2], self.status[2][8], self.status[2][
            6] = self.status[2][6], self.status[2][0], self.status[2][2], self.status[2][8]
        # permute edges
        self.status[2][1], self.status[2][5], self.status[2][7], self.status[2][
            3] = self.status[2][3], self.status[2][1], self.status[2][5], self.status[2][7]
        for i in range(3):
            self.status[0][i*3+2], self.status[1][i*3+2], self.status[5][i*3+2], \
                self.status[3][6-3 * i] = self.status[1][i*3+2], \
                self.status[5][i*3+2], self.status[3][6 -
                                                      3*i], self.status[0][i*3+2]

    def L(self):
      # permute corners
        self.status[4][0], self.status[4][2], self.status[4][8], self.status[4][
            6] = self.status[4][6], self.status[4][0], self.status[4][2], self.status[4][8]
        # permute edges
        self.status[4][1], self.status[4][5], self.status[4][7], self.status[4][
            3] = self.status[4][3], self.status[4][1], self.status[4][5], self.status[4][7]
        for i in range(3):
            self.status[0][i*3], self.status[1][i*3], self.status[5][i*3], \
                self.status[3][8-i * 3] = self.status[3][8-i*3], \
                self.status[0][i*3], self.status[1][i*3], self.status[5][i*3]

    def U(self):
        # permute corners
        self.status[0][0], self.status[0][2], self.status[0][8], self.status[0][
            6] = self.status[0][6], self.status[0][0], self.status[0][2], self.status[0][8]
        # permute edges
        self.status[0][1], self.status[0][5], self.status[0][7], self.status[0][
            3] = self.status[0][3], self.status[0][1], self.status[0][5], self.status[0][7]

        for i in range(3):
            self.status[1][i], self.status[4][i], self.status[3][i], self.status[2][
                i] = self.status[2][i], self.status[1][i], self.status[4][i], self.status[3][i]

    def D(self):
      # permute corners
        self.status[5][0], self.status[5][2], self.status[5][8], self.status[5][
            6] = self.status[5][6], self.status[5][0], self.status[5][2], self.status[5][8]
        # permute edges
        self.status[5][1], self.status[5][5], self.status[5][7], self.status[5][
            3] = self.status[5][3], self.status[5][1], self.status[5][5], self.status[5][7]
        for i in range(6, 9):
            self.status[1][i], self.status[2][i], self.status[3][i], self.status[4][
                i] = self.status[4][i], self.status[1][i], self.status[2][i], self.status[3][i]

    def F(self):
      # permute corners
        self.status[1][0], self.status[1][2], self.status[1][8], self.status[1][
            6] = self.status[1][6], self.status[1][0], self.status[1][2], self.status[1][8]
        # permute edges
        self.status[1][1], self.status[1][5], self.status[1][7], self.status[1][
            3] = self.status[1][3], self.status[1][1], self.status[1][5], self.status[1][7]
        for i in range(3):
            self.status[0][6+i], self.status[2][i*3], self.status[5][2-i], \
                self.status[4][8-i * 3] = self.status[4][8-i*3], \
                self.status[0][6+i], self.status[2][i*3], self.status[5][2-i]

    def B(self):
      # permute corners
        self.status[3][0], self.status[3][2], self.status[3][8], self.status[3][
            6] = self.status[3][6], self.status[3][0], self.status[3][2], self.status[3][8]
        # permute edges
        self.status[3][1], self.status[3][5], self.status[3][7], self.status[3][
            3] = self.status[3][3], self.status[3][1], self.status[3][5], self.status[3][7]
        for i in range(3):
            self.status[0][2-i], self.status[4][3*i], self.status[5][6+i], \
                self.status[2][8-3 * i] = self.status[2][8-3*i], \
                self.status[0][2-i], self.status[4][3*i], self.status[5][6+i]

    def R2(self):
        self.R()
        self.R()

    def L2(self):
        self.L()
        self.L()

    def F2(self):
        self.F()
        self.F()

    def B2(self):
        self.B()
        self.B()

    def U2(self):
        self.U()
        self.U()

    def D2(self):
        self.D()
        self.D()

    def Rp(self):
        self.R()
        self.R()
        self.R()

    def Lp(self):
        self.L()
        self.L()
        self.L()

    def Fp(self):
        self.F()
        self.F()
        self.F()

    def Bp(self):
        self.B()
        self.B()
        self.B()

    def Up(self):
        self.U()
        self.U()
        self.U()

    def Dp(self):
        self.D()
        self.D()
        self.D()

    def z2(self):
        # probably correct
        temp = self.status[0].copy()
        self.status[0] = self.status[5][::-1]
        self.status[5] = temp[::-1]
        # probably incorrect - to be checked
        idx = [6, 7, 8, 3, 4, 5, 0, 1, 2]
        temp = self.status[2].copy()
        self.status[2] = self.status[4][idx]
        self.status[4] = temp[idx]
        temp = self.status[1].copy()
        self.status[1] = temp[::-1]
        temp = self.status[3].copy()
        self.status[3] = temp[::-1]

    # takes a scramble as a list of moves or a string
    def scramble(self, scramble):
        if isinstance(scramble, list):
            moves = [s.replace("'", "p") for s in scramble]
        else:
            moves = scramble.replace("'", "p").split(' ')
        for i in moves:
            if i != "":
                getattr(self, i)()

    def show(self):
        face = self.status[3]
        for i in range(3):
            print("      ", cube_colors[face[8-i*3]],
                  cube_colors[face[7-i*3]], cube_colors[face[6-i*3]], "      ")
        print("")
        f1, f2, f3 = self.status[4], self.status[0], self.status[2]
        for j in range(3):
            print(cube_colors[f1[6+j]], cube_colors[f1[3+j]], cube_colors[f1[j]], '',
                  cube_colors[f2[j*3]], cube_colors[f2[j*3+1]
                                                    ], cube_colors[f2[j*3+2]], '',
                  cube_colors[f3[2-j]], cube_colors[f3[5-j]], cube_colors[f3[8-j]])
        face = self.status[1]
        print("")
        for i in range(3):
            print("      ", cube_colors[face[i*3]],
                  cube_colors[face[i*3+1]], cube_colors[face[i*3+2]], "      ")
        print("")
        face = self.status[5]
        for i in range(3):
            print("      ", cube_colors[face[i*3]],
                  cube_colors[face[i*3+1]], cube_colors[face[i*3+2]], "      ")

    # Get list of edges in a list, each edge is composed of 2 faces:
    # DF, DR, DB, DL, FR, BR, BL, FL, UF, UR, UB, UL

    def get_edges(self):
        self.edges = [  # Top (white) layer edges
            [self.status[0][7], self.status[1][1]],  # UF
            [self.status[0][5], self.status[2][1]],  # UR
            [self.status[0][1], self.status[3][1]],  # UB
            [self.status[0][3], self.status[4][1]],  # UL
            # E layer edges
            [self.status[1][3], self.status[4][5]],  # FR
            [self.status[3][5], self.status[4][3]],  # BR
            [self.status[3][3], self.status[2][5]],  # BL
            [self.status[1][5], self.status[2][3]],  # FL
            # Bottom layer (yellow) edges
            [self.status[5][1], self.status[1][7]],  # DF
            [self.status[5][3], self.status[4][7]],  # DR
            [self.status[5][7], self.status[3][7]],  # DB
            [self.status[5][5], self.status[2][7]]  # DL
        ]
        return self.edges

    def get_yellow_edges(self):
        self.yellow_edges_new = [e if 5 in e else [-1, -1]
                                 for e in self.get_edges()]
        return np.array(self.yellow_edges_new).flatten()

    # Get list of corners in a list, each corner is composed of 3 faces:
    # DFR, DBR, DBL, DFL, UFR, UBR, UBL, UFL
    def get_corners(self):
        self.corners = [[self.status[1][8], self.status[2][6], self.status[5][2]],  # DFR
                        [self.status[2][8], self.status[3][6], self.status[5][8]],
                        [self.status[3][8], self.status[4][6], self.status[5][6]],
                        [self.status[4][8], self.status[1][6], self.status[5][0]],
                        [self.status[1][2], self.status[2][0], self.status[0][8]],
                        [self.status[2][2], self.status[3][0], self.status[0][2]],
                        [self.status[3][2], self.status[4][0], self.status[0][0]],
                        [self.status[4][2], self.status[1][0], self.status[0][6]]
                        ]
        return self.corners

    def get_last_layer(self):
        edges = self.get_edges()
        corners = self.get_corners()
        ll_state = edges[:4]
        ll_state.extend(corners[4:])
        return np.array(flatten(ll_state))

    def get_oll_state(self):
        return np.array([x if x == 0 else -1 for x in self.get_last_layer()])

    def get_pll_state(self):
        pll_state = np.concatenate([self.status[i+1][:3] for i in range(4)])
        return np.array(pll_state).flatten()


class PLL_cube(SpeedCube):
    def __init__(self):
        super().__init__()
        self.solved_pll = self.get_pll_state()
        self.move_list = pll_moves

    def reset(self):
        self._state = 0
        self._reward = 0
        self._episode_ended = False
        self.status = np.array([np.zeros(9)*3, np.ones(9), np.ones(9)*2, np.ones(9)*3,
                                np.ones(9)*4, np.ones(9)*5], dtype=np.int32)
        auf = np.random.randint(0, 3)
        scramble_pll = np.random.choice(pll_scrambles)
        for _ in range(auf):
            scramble_pll += " U"
        self.scramble(scramble_pll)
        return self.get_pll_state()

    def step(self, action):
        action = self.move_list[action]
        if action != "":
            self.scramble(action)
        pll_state = self.get_pll_state()
        if np.all(self.solved_pll == pll_state):
            step_reward = 20
            self._episode_ended = True
            print('PLL solved:', self._state+1)
        elif action in ["U", "U'"]:
            step_reward = -1
        else:
            step_reward = -5

        self._reward += step_reward
        self._state += 1

        if self._episode_ended or self._state > MAX_EXPLO:
            if self._episode_ended:
                print('episode ended:', self._state)
                self._solved_episodes += 1
            else:
                self._failed_episodes += 1

            self._episode_ended = True
        return pll_state, float(step_reward), self._episode_ended, ''


class OLL_cube(SpeedCube):
    def __init__(self):
        super().__init__()
        self.solved_oll = self.get_oll_state()
        self.move_list = oll_moves

    def reset(self):
        self._state = 0
        self._reward = 0
        self._episode_ended = False
        self.status = np.array([np.zeros(9)*3, np.ones(9), np.ones(9)*2, np.ones(9)*3,
                                np.ones(9)*4, np.ones(9)*5], dtype=np.int32)
        auf = np.random.randint(0, 3)
        scramble_oll = np.random.choice(oll_scrambles)
        for _ in range(auf):
            scramble_oll += " U"
        self.scramble(scramble_oll)
        return self.get_oll_state()

    def step(self, action):
        action = self.move_list[action]
        if action != "":
            self.scramble(action)
        oll_state = self.get_oll_state()
        if np.all(self.solved_oll == oll_state):
            step_reward = 20
            self._episode_ended = True
            print('OLL solved:', self._state+1)
        elif action in ["U", "U'"]:
            step_reward = -1
        else:
            step_reward = -5

        self._reward += step_reward
        self._state += 1

        if self._episode_ended or self._state > MAX_EXPLO:
            if self._episode_ended:
                print('episode ended:', self._state)
                self._solved_episodes += 1
            else:
                self._failed_episodes += 1

            self._episode_ended = True
        return oll_state, float(step_reward), self._episode_ended, ''


def main():
    cube = SpeedCube()
    cube.scramble("R U L D2 F R'")
    cube.show()


if __name__ == "__main__":
    main()
