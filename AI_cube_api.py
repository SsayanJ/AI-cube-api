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
from cube_env import OLL_cube, PLL_cube, SpeedCube, MAX_EXPLO, F2L1_cube, F2L2_cube, F2L3_cube, F2L4_cube
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
f2l1_model = 'models/0720-F2L1_model.h5'
f2l2_model = 'models/0720-F2L2_model.h5'
f2l3_model = 'models/0720-F2L3_model.h5'
f2l4_model = 'models/0720-F2L4_model.h5'
cross_model = "models/cross_model15moves-16Jul.h5"
#oll_model = 'models/OLL_model_good2k.h5'
pll_model = 'models/PLL_model_good10k.h5'

oll_model = 'models/oll_model'

input_cross = 24
input_oll = 20
input_pll = 12
input_f2l = 40
actions_cross = 12
actions_oll = 59
actions_pll = 23
actions_f2l1 = 18
actions_f2l2 = 14
actions_f2l3 = 10
actions_f2l4 = 6


cross_solver = load_model(cross_model)

f2l1_solver = load_model(f2l1_model)

f2l2_solver = load_model(f2l2_model)

f2l3_solver = load_model(f2l3_model)

f2l4_solver = load_model(f2l4_model)

oll_solver = load_model(oll_model)

pll_solver = load_model(pll_model)


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


def cancel_moves(scramble):
    modification = True
    s = scramble.strip().split(' ')
    while modification:
        s, modification = cancel_moves(s)
    return s.strip()


def predict_f2l1(scramble: str):
    f2l1_cube = F2L1_cube()
    f2l1_cube.scramble(scramble)
    obs = f2l1_cube.get_f2l1()
    i = 0
    solution = ""
    done = False
    while not done:
        action = np.argmax(f2l1_solver.predict(obs[np.newaxis, :]))
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
        action = np.argmax(f2l2_solver.predict(obs[np.newaxis, :]))
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
        action = np.argmax(f2l3_solver.predict(obs[np.newaxis, :]))
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
        action = np.argmax(f2l4_solver.predict(obs[np.newaxis, :]))
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


def _predict_cross(scramble: str):
    cross_cube = SpeedCube()
    cross_cube.scramble(scramble)
    obs = cross_cube.get_yellow_edges()
    i = 0
    solution = ""
    done = False
    while not done:
        action = np.argmax(cross_solver.predict(obs[np.newaxis, :]))
        solution += cross_cube.move_list[action] + " "
        next = cross_cube.step(action)
        observation_, _, done, _ = next
        if done:
            break
        obs = observation_
        i += 1
    if i >= MAX_EXPLO:
        return "No cross solution found for this scramble"
    else:
        return solution.strip()


def _predict_f2l(scramble: str):
    solution = predict_f2l1(scramble)
    if solution == None:
        return "No F2L solution found"
    f2l2_solution = predict_f2l2(scramble + " " + solution)
    if f2l2_solution == None:
        return "No F2L solution found"
    else:
        solution = solution + " " + f2l2_solution
    f2l3_solution = predict_f2l3(scramble + " " + solution)
    if f2l3_solution == None:
        return "No F2L solution found"
    else:
        solution = solution + " " + f2l3_solution
    f2l4_solution = predict_f2l4(scramble + " " + solution)
    if f2l4_solution == None:
        return "No F2L solution found"
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
        action = np.argmax(oll_solver.predict(obs[np.newaxis, :].astype(np.float32)))
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
        action = np.argmax(pll_solver.predict(obs[np.newaxis, :]))
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

    return _predict_f2l(scramble_str)


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
        solution['solution_found'] = False
        return solution

    step_scramble = step_scramble + " " + f2l_sol
    oll_sol = _predict_oll(step_scramble)
    solution["step_solutions"]["OLL"] = oll_sol.split(" ")
    if oll_sol.startswith("No"):
        solution['solution_found'] = False
        return solution

    step_scramble = step_scramble + " " + oll_sol
    pll_sol = _predict_pll(step_scramble)
    solution["step_solutions"]["PLL"] = pll_sol.split(" ")
    if pll_sol.startswith("No"):
        solution['solution_found'] = False
        return solution

    return solution
