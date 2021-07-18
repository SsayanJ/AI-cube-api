"""
Note: you need to run the app from the root folder otherwise the models folder will not be found
- To run the app
$ uvicorn serving.model_as_service.main:app --reload
- To make a prediction from terminal
$ curl -X 'POST' 'http://127.0.0.1:8000/predict_obj' \
  -H 'accept: application/json' -H 'Content-Type: application/json' \
  -d '{ "age": 0, "sex": 0, "bmi": 0, "bp": 0, "s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0, "s6": 0 }'
"""


import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json
from DQN.dqn_agent import DQNAgent
from cube_env import OLL_cube, PLL_cube, SpeedCube, MAX_EXPLO

app = FastAPI()

# Loading models
cross_model = "models\cross_model15moves-16Jul.h5"
oll_model = 'models\OLL_model_good2k.h5'
pll_model = 'models\PLL_model_good10k.h5'

input_cross = 24
input_oll = 20
input_pll = 12
actions_cross = 12
actions_oll = 59
actions_pll = 23

cross_solver = DQNAgent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=input_cross,
                        n_actions=actions_cross, mem_size=1_000_000, epsilon_end=0.01,
                        batch_size=64, fname=cross_model)

cross_solver.load_model()

oll_solver = DQNAgent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=input_oll,
                      n_actions=actions_oll, mem_size=1_000_000, epsilon_end=0.01,
                      batch_size=64, fname=oll_model)
oll_solver.load_model()

pll_solver = DQNAgent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=input_pll,
                      n_actions=actions_pll, mem_size=1_000_000, epsilon_end=0.01,
                      batch_size=64, fname=pll_model)
pll_solver.load_model()

with open("models/model_config.json") as f:
    model_config = json.load(f)


@app.get("/")
async def root():
    return {"message": "FastAPI to solve Rubik's cube with AI"}


@app.get("/model_config")
async def return_model_config():
    return model_config


@app.post('/predict_cross')
async def predict_cross(scramble: str):
    cross_cube = SpeedCube()
    cross_cube.scramble(scramble)
    obs = cross_cube.get_yellow_edges()
    i = 0
    solution = ""
    done = False
    while not done:
        action = cross_solver.choose_action(obs)
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


@app.post('/predict_oll')
async def predict_oll(scramble: str):
    oll_cube = OLL_cube()
    oll_cube.scramble(scramble)
    obs = oll_cube.get_oll_state()
    i = 0
    solution = ""
    done = False
    while not done:
        action = oll_solver.choose_action(obs)
        print(action)
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
async def predict_pll(scramble: str):
    pll_cube = PLL_cube()
    pll_cube.scramble(scramble)
    obs = pll_cube.get_pll_state()
    i = 0
    solution = ""
    done = False
    while not done:
        action = pll_solver.choose_action(obs)
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
