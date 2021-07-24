"""
Note: you need to run the app from the root folder otherwise the models folder will not be found
- To run the app
$ uvicorn AI_cube_api:app --reload

  -Inputs: scrambles as string ("R U L D'") 
  -Outputs: solution to the step as string ("D L' U' R'") 
"""

import numpy as np
from fastapi import FastAPI
import json
from DQN.dqn_agent import DQNAgent
from cube_env import OLL_cube, PLL_cube, SpeedCube, MAX_EXPLO, F2L1_cube, F2L2_cube, F2L3_cube, F2L4_cube, Daisy_cube, Cross_from_daisy
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware


class Scramble(BaseModel):
    scramble_s: str


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Loading models
#f2l1_model = 'models/0720-F2L1_model.h5'
#f2l2_model = 'models/0720-F2L2_model.h5'
#f2l3_model = 'models/0720-F2L3_model.h5'
#f2l4_model = 'models/0720-F2L4_model.h5'
cross_model = "models/cross_model15moves-16Jul.h5"
#oll_model = 'models/OLL_model_good2k.h5'
#pll_model = 'models/PLL_model_good10k.h5'
daisy_model = 'models/daisy_model_nohack.h5'
cross_from_daisy_model = 'models/Cross_from_daisy_model_sat.h5'
oll_model = 'models/oll_model'
pll_model = 'models/pll_model'
f2l1_model = 'models/f2l1_model'
f2l2_model = 'models/f2l2_model'
f2l3_model = 'models/f2l3_model'
f2l4_model = 'models/f2l4_model'


cross_solver = load_model(cross_model)

f2l1_solver = load_model(f2l1_model)

f2l2_solver = load_model(f2l2_model)

f2l3_solver = load_model(f2l3_model)

f2l4_solver = load_model(f2l4_model)

oll_solver = load_model(oll_model)

pll_solver = load_model(pll_model)

daisy_solver = load_model(daisy_model)

cross_from_daisy_solver = load_model(cross_from_daisy_model)


def _cancel_moves(scramble):
    scramble_size = len(scramble)
    simplified = scramble.copy()
    i = 0
    j = 0
    modified = False
    while i < scramble_size - 1:
        left = scramble[i]
        right = scramble[i+1]
        if left[0] == right[0] and len(left) + len(right) == 3:
            simplified.pop(j)
            simplified.pop(j)
            i += 2
            modified = True
        else:
            i += 1
            j += 1
    # if scramble_size%2 == 1: simplified.append(scramble[-1])
    return simplified, modified


def _cancel_triple_move(scramble):
    scramble_size = len(scramble)
    simplified = scramble.copy()
    i = 0
    j = 0
    modified = False
    while i < scramble_size - 2:
        left = scramble[i]
        center = scramble[i+1]
        right = scramble[i+2]

        if left == center == right:
            simplified.pop(j)
            simplified.pop(j)
            simplified.pop(j)
            i += 3
            modified = True
            if len(left) == 1:
                simplified.insert(j, left + "'")
            elif left[1] == "'":
                simplified.insert(j, left[0])
            elif left[1] == "2":
                simplified.insert(j, left)
        else:
            i += 1
            j += 1
    # if scramble_size%2 == 1: simplified.append(scramble[-1])
    return simplified, modified


def _simplify_solution(solution: str):
    modification = True
    modif = False
    s = solution.strip().split(' ')
    while modification or modif:
        s, modification = _cancel_moves(s)
        s, modif = _cancel_triple_move(s)
    return (" ").join(s)


def predict_f2l1(scramble: str):
    f2l1_cube = F2L1_cube()
    f2l1_cube.scramble(scramble)
    obs = f2l1_cube.get_f2l1()
    i = 0
    solution = ""
    done = False
    while not done:
        action = np.argmax(f2l1_solver.predict(
            obs[np.newaxis, :].astype(np.float32)))
        solution += f2l1_cube.move_list[action] + " "
        next = f2l1_cube.step(action)
        observation_, _, done, _ = next
        if done:
            break
        obs = observation_
        i += 1
    if i >= MAX_EXPLO:
        return None
    else:
        return solution.strip()


def predict_f2l2(scramble: str):
    f2l2_cube = F2L2_cube()
    f2l2_cube.scramble(scramble)
    obs = f2l2_cube.get_f2l2()
    i = 0
    solution = ""
    done = False
    while not done:
        action = np.argmax(f2l2_solver.predict(
            obs[np.newaxis, :].astype(np.float32)))
        solution += f2l2_cube.move_list[action] + " "
        next = f2l2_cube.step(action)
        observation_, _, done, _ = next
        if done:
            break
        obs = observation_
        i += 1
    if i >= MAX_EXPLO:
        return None
    else:
        return solution.strip()


def predict_f2l3(scramble: str):
    f2l3_cube = F2L3_cube()
    f2l3_cube.scramble(scramble)
    obs = f2l3_cube.get_f2l3()
    i = 0
    solution = ""
    done = False
    while not done:
        action = np.argmax(f2l3_solver.predict(
            obs[np.newaxis, :].astype(np.float32)))
        solution += f2l3_cube.move_list[action] + " "
        next = f2l3_cube.step(action)
        observation_, _, done, _ = next
        if done:
            break
        obs = observation_
        i += 1
    if i >= MAX_EXPLO:
        return None
    else:
        return solution.strip()


def predict_f2l4(scramble: str):
    f2l4_cube = F2L4_cube()
    f2l4_cube.scramble(scramble)
    obs = f2l4_cube.get_f2l4()
    i = 0
    solution = ""
    done = False
    while not done:
        action = np.argmax(f2l4_solver.predict(
            obs[np.newaxis, :].astype(np.float32)))
        solution += f2l4_cube.move_list[action] + " "
        next = f2l4_cube.step(action)
        observation_, _, done, _ = next
        if done:
            break
        obs = observation_
        i += 1
    if i >= MAX_EXPLO:
        return None
    else:
        return solution.strip()


with open("models/model_config.json") as f:
    model_config = json.load(f)


# def _predict_cross(scramble: str):
#     cross_cube = SpeedCube()
#     cross_cube.scramble(scramble)
#     obs = cross_cube.get_yellow_edges()
#     i = 0
#     solution = ""
#     done = False
#     while not done:
#         action = np.argmax(cross_solver.predict(obs[np.newaxis, :]))
#         solution += cross_cube.move_list[action] + " "
#         next = cross_cube.step(action)
#         observation_, _, done, _ = next
#         if done:
#             break
#         obs = observation_
#         i += 1
#     if i >= MAX_EXPLO:
#         return "No cross solution found for this scramble"
#     else:
#         return solution.strip()


def _predict_daisy(scramble: str):
    daisy_cube = Daisy_cube()
    daisy_cube.scramble(scramble)
    obs = daisy_cube.get_daisy()
    i = 0
    solution = ""
    done = False
    while not done:
        action = np.argmax(daisy_solver.predict(obs[np.newaxis, :]))
        solution += daisy_cube.move_list[action] + " "
        next = daisy_cube.step(action)
        observation_, _, done, _ = next
        if done:
            break
        obs = observation_
        i += 1
    if i >= MAX_EXPLO:
        return "No daisy solution found for this scramble"
    else:
        return solution.strip()


def _predict_cross_from_daisy(scramble: str):
    cross_daisy_cube = Cross_from_daisy()
    cross_daisy_cube.scramble(scramble)
    obs = cross_daisy_cube.get_yellow_edges()
    i = 0
    solution = ""
    done = False
    while not done:
        action = np.argmax(cross_from_daisy_solver.predict(obs[np.newaxis, :]))
        solution += cross_daisy_cube.move_list[action] + " "
        next = cross_daisy_cube.step(action)
        observation_, _, done, _ = next
        if done:
            break
        obs = observation_
        i += 1
    if i >= MAX_EXPLO:
        return "No cross solution found for this scramble"
    else:
        return solution.strip()


def _predict_cross(scramble: str):
    solution = _predict_daisy(scramble)
    if solution.startswith("No"):
        print('Daisy error')
        return "No Daisy solution found"
    # print(solution)
    cross_solution = _predict_cross_from_daisy(scramble + " " + solution)
    if cross_solution.startswith("No"):
        print('Cross from daisy error')
        return "No Cross solution found"
    else:
        solution = solution + " " + cross_solution
        return solution.strip().replace('p', "'")


def _predict_f2l(scramble: str):
    solution = predict_f2l1(scramble)
    if solution == None:
        print("1 - F2L1 error")
        return "No F2L1 solution found"
    f2l2_solution = predict_f2l2(scramble + " " + solution)
    if f2l2_solution == None:
        print("2 - F2L2 error")
        return "No F2L2 solution found"
    else:
        solution = solution + " " + f2l2_solution
    f2l3_solution = predict_f2l3(scramble + " " + solution)
    if f2l3_solution == None:
        print("3 - F2L3 error")
        return "No F2L3 solution found"
    else:
        solution = solution + " " + f2l3_solution
    f2l4_solution = predict_f2l4(scramble + " " + solution)
    if f2l4_solution == None:
        print("4 - F2L4 error")
        return "No F2L4 solution found"
    else:
        solution = solution + " " + f2l4_solution

    return solution.strip()


def _predict_oll(scramble: str):
    oll_cube = OLL_cube()
    oll_cube.scramble(scramble)
    obs = oll_cube.get_oll_state()
    i = 0
    solution = ""
    done = False
    while not done:
        action = np.argmax(oll_solver.predict(
            obs[np.newaxis, :].astype(np.float32)))
        solution += oll_cube.move_list[action] + " "
        next = oll_cube.step(action)
        observation_, _, done, _ = next
        if done:
            break
        obs = observation_
        i += 1
    if i >= MAX_EXPLO:
        return "No OLL solution found for this scramble"
    else:
        return solution.strip()


def _predict_pll(scramble: str):
    pll_cube = PLL_cube()
    pll_cube.scramble(scramble)
    obs = pll_cube.get_pll_state()
    i = 0
    solution = ""
    done = False
    while not done:
        action = np.argmax(pll_solver.predict(
            obs[np.newaxis, :].astype(np.float32)))
        solution += pll_cube.move_list[action] + " "
        next = pll_cube.step(action)
        observation_, _, done, _ = next
        if done:
            break
        obs = observation_
        i += 1
    if i >= MAX_EXPLO:
        return "No PLL solution found for this scramble"
    else:
        return solution.strip()


@app.get("/")
async def root():
    return {"message": "FastAPI to solve Rubik's cube with AI"}


@app.get("/model_config")
async def return_model_config():
    return model_config


@app.post('/predict_cross')
async def predict_cross(scramble: Scramble):
    scramble_str = scramble.dict()["scramble_s"]

    return _predict_cross(scramble_str)


@app.post('/predict_f2l')
async def predict_f2l(scramble: Scramble):
    scramble_str = scramble.dict()["scramble_s"]

    return _simplify_solution(_predict_f2l(scramble_str))


@app.post('/predict_oll')
async def predict_oll(scramble: Scramble):
    scramble_str = scramble.dict()["scramble_s"]

    return _predict_oll(scramble_str)


@app.post('/predict_pll')
async def predict_pll(scramble: Scramble):
    scramble_str = scramble.dict()["scramble_s"]
    return _predict_pll(scramble_str)


@app.post('/solve_cube')
async def solve_cube(scramble: Scramble):
    scramble_str = scramble.dict()["scramble_s"]
    solution = {"original_scramble": scramble_str,
                "solution_found": True}

    cross_sol = _predict_cross(scramble_str)
    solution["step_solutions"] = {"Cross": cross_sol.split(" ")}
    if cross_sol.startswith("No"):
        solution['solution_found'] = False
        return solution

    step_scramble = scramble_str + " " + cross_sol
    f2l_sol = _predict_f2l(step_scramble)
    solution["step_solutions"]["F2L"] = f2l_sol.split(" ")

    if f2l_sol.startswith("No"):
        print("F2L error")
        solution['solution_found'] = False
        return solution

    step_scramble = step_scramble + " " + f2l_sol
    oll_sol = _predict_oll(step_scramble)
    solution["step_solutions"]["OLL"] = oll_sol.split(" ")
    if oll_sol.startswith("No"):
        print('OLL error')
        solution['solution_found'] = False
        return solution

    step_scramble = step_scramble + " " + oll_sol
    pll_sol = _predict_pll(step_scramble)
    solution["step_solutions"]["PLL"] = pll_sol.split(" ")
    if pll_sol.startswith("No"):
        print("PLL error")
        solution['solution_found'] = False
        return solution

    return solution
