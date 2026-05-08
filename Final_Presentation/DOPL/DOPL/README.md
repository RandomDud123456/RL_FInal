# **Direct Online Preference Learning**  

This is the official codebase for the paper:  
**[Direct Online Preference Learning for Restless Bandits with Preference Feedback](https://arxiv.org/abs/2410.05527)**

---

## **Setup Instructions**

### Install Dependencies
1. **Using pip**:  
   ```bash
   pip install -r requirements.txt
   ```

2. **Using conda**:  
   ```bash
   conda env create -f environment.yaml
   ```

### Additional Requirements for Linear Solver
To use the Linear Program Solver, a **Gurobi License** is required.  
Refer to the official academic licensing program here:  
[Academic Program and Licenses - Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/)

---

## **Setting Up the PREF-RMAB Environments**

Environments are completely characterized by their **transition kernels** (transition probability matrices) and **reward functions**. To generate these files for predefined environments, run the following commands:

```bash
cd dopl/RMAB_env_instances
python create_cpap.py
python create_armman.py
python create_app_marketing.py
```

### Creating Your Own Environment
You can create a custom environment by writing a `create_<your_env>.py` script that generates the transition kernel and reward `.npy` files. 

- Modify the `env_config.arm` parameter in the configuration file to use `<your_env>` in a run.

---

## **Running the Code**

Run the code for a specific environment using the following command:  

```bash
python run.py --config-name=<env_name>
```

### Example:
To start a run for the CPAP environment:

```bash
python run.py --config-name=cpap
```

Check out the configuration files in the `conf` folder. 

The output will be stored in a timestamped directory within the `outputs` folder.

