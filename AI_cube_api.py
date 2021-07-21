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
f2l1_model = 'models/F2L1_model.h5'
f2l2_model = 'models/F2L2_model.h5'
f2l3_model = 'models/F2L3_model.h5'
f2l4_model = 'models/F2L4_model.h5'
cross_model = "models/cross_model15moves-16Jul.h5"
oll_model = 'models/OLL_model_good2k.h5'
pll_model = 'models/PLL_model_good10k.h5'

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


def predict_f2l1(scramble: str):
    f2l1_cube = F2L1_cube()
    f2l1_cube.scramble(scramble)
    obs = f2l1_cube.get_f2l1()
    i = 0
    solution = ""
    done = False
    while not done:
        action = np.argmax(f2l1_cube.predict(obs[np.newaxis, :]))
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
        return solution


def predict_f2l2(scramble: str):
    f2l2_cube = F2L2_cube()
    f2l2_cube.scramble(scramble)
    obs = f2l2_cube.get_f2l2()
    i = 0
    solution = ""
    done = False
    while not done:
        action = np.argmax(f2l2_cube.predict(obs[np.newaxis, :]))
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
        return solution


def predict_f2l3(scramble: str):
    f2l3_cube = F2L3_cube()
    f2l3_cube.scramble(scramble)
    obs = f2l3_cube.get_f2l3()
    i = 0
    solution = ""
    done = False
    while not done:
        action = np.argmax(f2l3_cube.predict(obs[np.newaxis, :]))
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
        return solution


def predict_f2l4(scramble: str):
    f2l4_cube = F2L4_cube()
    f2l4_cube.scramble(scramble)
    obs = f2l4_cube.get_f2l4()
    i = 0
    solution = ""
    done = False
    while not done:
        action = np.argmax(f2l4_cube.predict(obs[np.newaxis, :]))
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
        return solution


with open("models/model_config.json") as f:
    model_config = json.load(f)


@app.get("/")
async def root():
    return {"message": "FastAPI to solve Rubik's cube with AI"}


@app.get("/model_config")
async def return_model_config():
    return model_config


@app.post('/predict_cross')
async def predict_cross(scramble: Scramble):
    cross_cube = SpeedCube()
    scramble_str = scramble.dict()["scramble_s"]
    cross_cube.scramble(scramble_str)
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
        return solution


@app.post('/predict_f2l')
async def predict_f2l(scramble: Scramble):
    scramble_str = scramble.dict()["scramble_s"]
    solution = predict_f2l1(scramble_str)
    print(solution)
    if solution == None:
        return "No F2L solution found"
    f2l2_solution = predict_f2l2(scramble_str + " " + solution)
    if f2l2_solution == None:
        return "No F2L solution found"
    else:
        solution = solution + " " + f2l2_solution
    f2l3_solution = predict_f2l3(scramble_str + " " + solution)
    if f2l3_solution == None:
        return "No F2L solution found"
    else:
        solution = solution + " " + f2l3_solution
    f2l4_solution = predict_f2l4(scramble_str + " " + solution)
    if f2l4_solution == None:
        return "No F2L solution found"
    else:
        solution = solution + " " + f2l4_solution
    return solution


@app.post('/predict_oll')
async def predict_oll(scramble: Scramble):
    oll_cube = OLL_cube()
    scramble_str = scramble.dict()["scramble_s"]
    oll_cube.scramble(scramble_str)
    obs = oll_cube.get_oll_state()
    i = 0
    solution = ""
    done = False
    while not done:
        action = np.argmax(oll_solver.predict(obs[np.newaxis, :]))
        try:
            solution += oll_cube.move_list[action] + " "
        except:
            print(action)
            print(len(oll_cube.move_list))
        next = oll_cube.step(action)
        observation_, _, done, _ = next
        if done:
            break
        obs = observation_
        i += 1
    if i >= MAX_EXPLO:
        return "No OLL solution found for this scramble"
    else:
        return solution


@app.post('/predict_pll')
async def predict_pll(scramble: Scramble):
    pll_cube = PLL_cube()
    scramble_str = scramble.dict()["scramble_s"]
    pll_cube.scramble(scramble_str)
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
        return solution
