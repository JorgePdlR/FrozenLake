## Project Structure

### Main
The main.py file is the entry point of the project for the agent to interact with the 
environment and find optimal policies with each algorithm.

### Algorithms
File algorithms.py implements all reinforcement learning algorithms.
It is divided in classes based on the algorithms:
  - Tabular model based
    - Policy evaluation
    - Policy improvement
    - Policy iteration
    - Value iteration
  - SARSA
    - SARSA control (make_policy)
    - linear SARSA control (make_linear_approx_policy)
  - Q-Learning
    - Q-learning control (make_policy)
    - Linear Q-learning control (make_linear_approx_policy)
  - Deep Q-Learning
    - DeepQLearning (make_policy)

### Frozen Lake
In the file frozenLake.py the environment (big & small lakes) dynamics are encoded.
Use play() to test the implementation of the environment.

### Assignment report 
File assignment_report.py is a secondary main file to run individually each algorithm
and implements all the necessary code to answer the questions for the report assignment

### Config
The file config.py is used to configure the verbose print for extra tracing and
debugging in the project, all files include this file. Any extra general configurations
for the project can be added here.

---

## Create Environment for Project 
### (Optional but highly recommended)

This is to be done only the first time.
Note: This project uses Python3.10.
1. Create virtual environment: 
- python3 -m venv lake_venv

2. Activate virtual environment: 
- for Linux:
  - source lake_venv/bin/activate
- for Windows:
  - .\lake_venv\Scripts\Activate
  
3. Install requirements using either of the 2 commands:
   - pip install -r requirements.txt
   - cat requirements.txt | xargs -n 1 pip install

---

## Run Project

1. Activate virtual environment: 
- for Linux:
  - source lake_venv/bin/activate
- for Windows:
  - .\lake_venv\Scripts\Activate
  
2. Run code: python main.py

3. If you install new libraries, to automatically add it and the version to the list of required libraries run:
- pip freeze > requirements.txt 

---


