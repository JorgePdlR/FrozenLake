## Project Structure

### Main
This is the entry point of the project for the agent to interact with the 
environment and find optimal policies.

### Control Algorithms
  - Dynamic Programming
  - SARSA Control
  - SARSA Control using Linear Approximation function
  - Q-Learning
  - Q-Learning using Linear Approximation function
  - Deep Q-Learning

### Frozen Lake
It has the environment (big & small lakes) encoded.
Use play() to test the implementation of the environment.

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


