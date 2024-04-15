import pandas as pd
import numpy as np
import warnings
import gurobipy as gp
import random
import re
from collections import defaultdict

### INITIAL SETTINGS ###

# Disable Future Warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Random data generation pagrameters
random_seeds = [10,20,30]

#Data set codes
demand_data_set_codes = [1,2,3]

#Max data size
length_of_full_demand_data=100

#Display to screen
Display_to_screen=True

## FUNCTIONS

# Function for creating nested dictionary
def nested_dict(n, type):
    if n == 1:
        return defaultdict(lambda:type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))

def Specifications_and_Waste_Costs_Generator ():

    # Define global variables
    global demand_data_set_codes

    # Define specifications for all data sets
    Raw_Material_Specification_Row_Names=["min_width","max_width","min_thickness","max_thickness","min_reduction_rate","max_reduction_rate"]
    Process_Specication_Row_Names=["min_L1_edge_trim","max_L1_edge_trim", "min_L2_edge_trim","max_L2_edge_trim","min_L1_scrap","max_L1_scrap","min_L2_scrap","max_L2_scrap"]
    Waste_Cost_Row_Names=["L1_edge_trim","L2_edge_trim","L1_scrap","L2_scrap","Final_leftover"]
    Operational_Cost_Row_Names=["L1","L2"]

    Raw_Material_Specification=pd.DataFrame(index=Raw_Material_Specification_Row_Names,columns=demand_data_set_codes)
    Process_Specification=pd.DataFrame(index=Process_Specication_Row_Names,columns=demand_data_set_codes)
    Waste_Costs=pd.DataFrame(index=Waste_Cost_Row_Names,columns=demand_data_set_codes)
    Operational_Costs=pd.DataFrame(index=Operational_Cost_Row_Names,columns=demand_data_set_codes)

    min_width=[100,100,100]
    max_width=[1600,1600,1600]
    min_thickness=[2,3,4]
    max_thickness=[6,5,6]
    min_coil_length=[1000,1000,1000]
    max_coil_length=[2000,2000,2000]
    min_reduction_rate=[0.5,0.5,0.5]
    max_reduction_rate=[0.8,0.8,0.8]
    min_L1_edge_trim=[10,10,10]
    max_L1_edge_trim=[50,50,50]
    min_L2_edge_trim=[10,10,10]
    max_L2_edge_trim=[50,50,50]
    min_L2_scrap=[0,0,0]
    waste_cost_L1_edge_trim=[100,100,100]
    waste_cost_L2_edge_trim=[100,100,100]
    waste_cost_L1_scrap=[100,100,100]
    waste_cost_L2_scrap=[150,150,150]
    waste_cost_Final_leftover=[200,200,200]
    operational_cost_L2=[20,20,20]

    for i in demand_data_set_codes:
        Raw_Material_Specification.loc["min_width",i]=min_width[i-1]
        Raw_Material_Specification.loc["max_width",i]=max_width[i-1]
        Raw_Material_Specification.loc["min_thickness",i]=min_thickness[i-1]
        Raw_Material_Specification.loc["max_thickness",i]=max_thickness[i-1]
        Raw_Material_Specification.loc["min_coil_length",i]=min_coil_length[i-1]
        Raw_Material_Specification.loc["max_coil_length",i]=max_coil_length[i-1]
        Raw_Material_Specification.loc["min_reduction_rate",i]=min_reduction_rate[i-1]
        Raw_Material_Specification.loc["max_reduction_rate",i]=max_reduction_rate[i-1]
        Process_Specification.loc["min_L1_edge_trim",i]=min_L1_edge_trim[i-1]
        Process_Specification.loc["max_L1_edge_trim",i]=max_L1_edge_trim[i-1]
        Process_Specification.loc["min_L2_edge_trim",i]=min_L2_edge_trim[i-1]
        Process_Specification.loc["max_L2_edge_trim",i]=max_L2_edge_trim[i-1]
        Process_Specification.loc["min_L2_scrap",i]=min_L2_scrap[i-1]
        Waste_Costs.loc["L1_edge_trim",i]=waste_cost_L1_edge_trim[i-1]
        Waste_Costs.loc["L2_edge_trim",i]=waste_cost_L2_edge_trim[i-1]
        Waste_Costs.loc["L1_scrap",i]=waste_cost_L1_scrap[i-1]
        Waste_Costs.loc["L2_scrap",i]=waste_cost_L2_scrap[i-1]
        Waste_Costs.loc["Final_leftover",i]=waste_cost_Final_leftover[i-1]
        Operational_Costs.loc["L2",i]=operational_cost_L2[i-1]

    return(Raw_Material_Specification,Process_Specification,Waste_Costs,Operational_Costs)

def Random_Demand_Generator (length_of_full_demand_data):

    # Define global variables
    global demand_data_set_code

    # Define parameters for each dataset
    sensitivity_width_values = [50, 50, 50]
    sensitivity_thickness_values = [0.2, 0.2, 0.2]
    sensitivity_length_values = [50, 50, 50]
    max_number_of_coil_demanded_values = [10, 10, 10]
    min_demand_coil_thickness_values = [0.3, 0.3, 0.3]
    max_demand_coil_thickness_values = [2.0, 2.0, 2.0]
    min_demand_coil_weight_values = [5, 5, 5]
    max_demand_coil_weight_values = [15, 15, 15]
    min_demand_coil_width_values = [100, 100, 500]
    max_demand_coil_width_values = [1500,  1500, 1500]
    min_demand_coil_length_values = [500, 500, 500]
    max_demand_coil_lenght_values = [1500,  1500, 1500]

    sensitivity_width= sensitivity_width_values[demand_data_set_code - 1]
    sensitivity_thickness= sensitivity_thickness_values[demand_data_set_code - 1]
    sensitivity_length= sensitivity_length_values[demand_data_set_code - 1]
    max_number_of_coil_demanded=max_number_of_coil_demanded_values[demand_data_set_code - 1]
    min_demand_coil_thickness= min_demand_coil_thickness_values[demand_data_set_code - 1]
    max_demand_coil_thickness= max_demand_coil_thickness_values[demand_data_set_code - 1]
    min_demand_coil_weight= min_demand_coil_weight_values[demand_data_set_code - 1]
    max_demand_coil_weight= max_demand_coil_weight_values[demand_data_set_code - 1]
    min_demand_coil_width= min_demand_coil_width_values[demand_data_set_code - 1]
    max_demand_coil_width= max_demand_coil_width_values[demand_data_set_code - 1]

    random.seed(random_seeds[demand_data_set_code-1] ) # Set the random seed
    Demand_Column_Names=["D","ψ_d","τ_d","μ_d","α_d"]
    Df_demand_full=pd.DataFrame(columns=Demand_Column_Names)

    for i in range(0,length_of_full_demand_data):
        Df_demand_full.loc[i,"D"]=int(i+1)
        Df_demand_full.loc[i,"ψ_d"]=int(round(random.randint(min_demand_coil_width, max_demand_coil_width)/sensitivity_width)*sensitivity_width)
        Df_demand_full.loc[i,"τ_d"] = round(round(random.uniform(min_demand_coil_thickness, max_demand_coil_thickness) / sensitivity_thickness) * sensitivity_thickness, 1)
        Df_demand_full.loc[i,"μ_d"]=(int(round((Df_demand_full.loc[i,"ψ_d"]*random.randint(min_demand_coil_weight,max_demand_coil_weight)/7.85)/(Df_demand_full.loc[i,"τ_d"]*Df_demand_full.loc[i,"ψ_d"]/1000)/sensitivity_length)*sensitivity_length))
        Df_demand_full.loc[i,"α_d"]=Df_demand_full.loc[i,"μ_d"]*(random.randint(1,max_number_of_coil_demanded))
    return (Df_demand_full)

def Random_Demand_Compatibility_Matrix_Generation (Df_demand):
    size_of_matrix = Df_demand.shape[0]  #the number of rows in Df_demand
    compatability_dictionary = nested_dict(2, list)
    for i in range(1,size_of_matrix+1):
        for j in range(1,size_of_matrix+1):
            if i==j:
                compatability_dictionary[(i,j)]=1
            else:
                #compatability_dictionary[(i,j)]=random.choice([0, 1])
                compatability_dictionary[(i,j)]=1
    return(compatability_dictionary)

def Gurobi_Modelling (Df_demand,Df_raw_material):

    # Define global variables
    global data_set_code

    # Defining sets
    D=Df_demand["D"].to_dict()
    R=Df_raw_material["R"].to_dict()

    # Defining parameters
    M=9999999

    ψ_d = Df_demand["ψ_d"]  # Width of demand d ∈ D
    ψ_d = ψ_d.to_dict()

    τ_d = Df_demand["τ_d"]  # Thickness of demand d ∈ D
    τ_d =τ_d.to_dict()

    μ_d = Df_demand["μ_d"]  # length of coil for demand d ∈ D
    μ_d =μ_d.to_dict()

    α_d = Df_demand["α_d"]  # Total demand as total length for d ∈ D
    α_d =α_d.to_dict()

    γ_comp=Random_Demand_Compatibility_Matrix_Generation(Df_demand) # compatability dictionary as γdd′ d ∈ D and d′ ∈ D

    λmin_width = Raw_Material_Specification.loc["min_width", demand_data_set_code]
    λmax_width = Raw_Material_Specification.loc["max_width", demand_data_set_code]
    λmin_thickness = Raw_Material_Specification.loc["min_thickness", demand_data_set_code]
    λmax_thickness = Raw_Material_Specification.loc["max_thickness", demand_data_set_code]
    λmin_coil_lenght = Raw_Material_Specification.loc["min_coil_length", demand_data_set_code]
    λmax_coil_lenght = Raw_Material_Specification.loc["max_coil_length", demand_data_set_code]
    λmin_reduction_rate = Raw_Material_Specification.loc["min_reduction_rate", demand_data_set_code]
    λmax_reduction_rate = Raw_Material_Specification.loc["max_reduction_rate", demand_data_set_code]
    λmin_L1_edge_trim = Process_Specification.loc["min_L1_edge_trim", demand_data_set_code]
    λmax_L1_edge_trim = Process_Specification.loc["max_L1_edge_trim", demand_data_set_code]
    λmin_L2_edge_trim = Process_Specification.loc["min_L2_edge_trim", demand_data_set_code]
    λmax_L2_edge_trim = Process_Specification.loc["max_L2_edge_trim", demand_data_set_code]
    λmin_L2_scrap = Process_Specification.loc["min_L2_scrap", demand_data_set_code]
    #λmax_L2_scrap = Process_Specification.loc["max_L2_scrap", demand_data_set_code]
    Cs_L1_edge_trim = Waste_Costs.loc["L1_edge_trim", demand_data_set_code]
    Cs_L2_edge_trim = Waste_Costs.loc["L2_edge_trim", demand_data_set_code]
    Cs_L2_scrap = Waste_Costs.loc["L2_scrap", demand_data_set_code]
    Cs_Final_leftover = Waste_Costs.loc["Final_leftover", demand_data_set_code]
    Co_L2 = Operational_Costs.loc["L2", demand_data_set_code]

    # Model Initialization
    with gp.Env(empty=True) as env:
        # Set the OutputFlag parameter to 1 to enable messages during optimization
        env.setParam('OutputFlag', 1)
        env.start()
        with gp.Model(env=env) as model:
            model = gp.Model("Basic Model")
            # Set the LogToConsole parameter to 0 to disable messages during optimization
            if Display_to_screen:
                model.Params.LogToConsole = 1
            else:
                model.Params.LogToConsole = 0

    # Defining variables

    # Binary variable that denotes if sales order d ∈ D is allocated to raw material order r ∈ R
    X = model.addVars(R, D, vtype="B", name="X|")

    # Binary variable that denotes if operation is planned for raw material order at operation L2, r ∈ R
    I = model.addVars(R, vtype="B", lb=0, name="I|")

    # The planned waste for raw material order r ∈ R due to adjustment in specification s ∈ {level 1 edge trim, level 2 edge trim, level 2 scrap}
    E_L1_edge_trim=model.addVars(R, vtype="C", lb=λmin_L1_edge_trim, ub=λmax_L1_edge_trim,name="E_L1_edge_trim|")
    E_L2_edge_trim=model.addVars(R, vtype="C", lb=0, ub=λmax_L2_edge_trim,name="E_L2_edge_trim|")
    E_L2_scrap=model.addVars(R, vtype="C", lb=λmin_L2_scrap,name="E_L2_scrap|")

    # The surplus in length for placement of sales order d ∈ D on raw material order r ∈ R
    F=model.addVars(R, D, lb=0, vtype="C", name="F|")

    # Width of raw material order r ∈ R
    W = model.addVars(R, vtype="C", lb=λmin_width, ub=λmax_width, name="W|")

    # Thickness of raw material order r ∈ R
    T=model.addVars(R, vtype="C", lb=λmin_thickness, ub=λmax_thickness, name="T|")

    # Coil Length of raw material order r ∈ R
    ρ=model.addVars(R, vtype="C",lb=λmin_coil_lenght, ub=λmax_coil_lenght, name="CL|")

    # Total Length of raw material order r ∈ R
    L=model.addVars(R, vtype="C", name="L|")

    # The number of times the coil length of raw material to be ordered r ∈ R, d ∈ D
    NR = model.addVars(R, vtype="I", lb=0, name="N_r|")

    # Total length of demand d ∈ D allocated to raw material order r ∈ R
    Λ= model.addVars(R, D, lb=0, vtype="C", name="L_d|")

    # The number of times the coil length of demand allocated to raw material coil for r ∈ R, d ∈ D
    #ND = model.addVars(R, D, lb=0, vtype="I", name="N_d|")

    # Auxiliary binary variable that denotes if raw material order r ∈ R is place
    Ω= model.addVars(R, vtype="B", name="O|")

    # Defining Constraints

    for d in D:
        model.addConstr(sum(X[r,d] for r in R ) <= 1)
        model.addConstr(sum(Λ[r, d] * NR[r] for r in R) <= α_d[d])

    for d in D:
        for r in R:
            model.addConstr(Λ[r, d] == μ_d[d]* X[r,d])
            model.addConstr(F[r, d] <= M * X[r,d])
            model.addConstr(Λ[r, d] + F[r, d] == ρ[r]* X[r,d])
            model.addConstr(T[r] >= (τ_d[d] / (1 - λmin_reduction_rate)) * X[r, d])
            model.addConstr(T[r] <= (τ_d[d] / (1 - λmax_reduction_rate)) * X[r, d] + M* (1 - X[r, d]))

    for r in R:
        model.addConstr(sum(X[r,d] for d in D ) <= Ω[r])
        model.addConstr(sum(X[r,d]* ψ_d[d] for d in D) +E_L1_edge_trim[r] + E_L2_edge_trim[r] + E_L2_scrap[r]== W[r])
        model.addConstr(ρ[r]*NR[r]==L[r])

    for r in R:
        for d in D:
            for d_1 in D:
                if d!=d:
                    model.addConstr(X[r,d]+X[r,d_1] <= 1+γ_comp[d,d_1])

    for r in R:
        model.addConstr(E_L2_edge_trim[r] >= λmin_L2_edge_trim * I[r])
        model.addConstr(E_L2_edge_trim[r] <= W[r] *I[r])
        model.addConstr(E_L2_scrap[r] <= W[r] * I[r])

    # Defining Objectives
    Obj_1 = (sum(E_L1_edge_trim[r]*L[r]*Cs_L1_edge_trim + E_L2_edge_trim[r]*L[r]*Cs_L2_edge_trim  + E_L2_scrap[r]*L[r]*Cs_L2_scrap+I[r]* L[r]*Co_L2  for r in R) + sum(F[r,d]* L[r]*Cs_Final_leftover for r in R for d in D))
    Obj_2 = (sum(Ω[r] for r in R))
    model.setObjective(Obj_1, gp.GRB.MINIMIZE)

    # Optimizing the model
    model.optimize()

    # Exporting the model
    model.write ("Model.lp")

    # Check optimization status
    status = model.status
    result_data = {}

    if status == gp.GRB.Status.OPTIMAL:
        # Model is feasible and bounded
        print("Optimal objective value:", model.objVal)
        # Print attribute values
        print("Attributes of the optimum result:")
        # Iterate over variables
        for var in model.getVars():
            # Extract generic name and indices
            variable_name = var.varName
            generic_name = variable_name.split("|")[0]
            indices_str = variable_name.split("|")[1] if "|" in variable_name else ""

            # Split indices into two parts
            indices_parts = indices_str.split(",")
            indice_1 = indices_parts[0].strip("[]") if indices_parts else ""
            indice_2 = indices_parts[1].strip("[]") if len(indices_parts) > 1 else "NA"

            # Store generic name, indice_1, indice_2, and value in the dictionary
            result_data[(generic_name, indice_1, indice_2)] = var.x

    elif status == gp.GRB.Status.INFEASIBLE:
        # Model is infeasible
        print("Model is infeasible.")
        # Compute and print the Irreducible Inconsistent Subsystem (IIS)
        model.computeIIS()
        for constr in model.getConstrs():
            if constr.IISConstr:
                print(f"Infeasible constraint: {constr.ConstrName}")
    elif status == gp.GRB.Status.UNBOUNDED:
        # Model is unbounded
        print("Model is unbounded.")
        model.computeIIS()
        for constr in model.getConstrs():
            if constr.IISConstr:
                print(f"Infeasible constraint: {constr.ConstrName}")
    else:
        # Other cases
        print("Optimization ended with status:", status)
        model.computeIIS()
        for constr in model.getConstrs():
            if constr.IISConstr:
                print(f"Infeasible constraint: {constr.ConstrName}")

    print (result_data)

    return result_data

def Opt_Result_Visualization (Df_demand, Df_raw_material, result_data):

    for r in Df_raw_material["R"].values:
        Raw_material_text= "Coil ID  :" + str(r) + " > " + str(result_data[('T', str(r), "NA")])+"x"+str(result_data[('W', str(r), "NA")]) +"x"+str(result_data[('CL', str(r), "NA")])+ " > Length:" + str(result_data[('L', str(r), "NA")])+"("+ str(result_data[('N_r', str(r), "NA")])+")"
        print (Raw_material_text)
        for d in Df_demand["D"].values:
            Demand_allocated_text=""
            if result_data[('X', str(r), str(d))]==1:
                Demand_allocated_text="Demand ID:"+str(d)+" > "+str(Df_demand.loc[d,"τ_d"])+"x"+str(Df_demand.loc[d,"ψ_d"])+" > Length:"+str(result_data[('L_d', str(r), str(d))])
            if result_data[('F', str(r), str(d))]==1:
                Demand_allocated_text=Demand_allocated_text+"|Leftover:"+str(result_data[('F', str(r), str(d))])
            if Demand_allocated_text!="":
                print (Demand_allocated_text)
        E_L1_edge_trim_text="L1 Edge Trim:"+ str(result_data[('E_L1_edge_trim', str(r), "NA")])
        E_L2_edge_trim_text="L2 Edge Trim:"+ str(result_data[('E_L2_edge_trim', str(r), "NA")])
        E_L2_scrap_text="L2 Scrap:"+ str(result_data[('E_L2_scrap', str(r), "NA")])
        if (E_L1_edge_trim_text!=""):
            print (E_L1_edge_trim_text)
        if (E_L2_edge_trim_text!=""):
            print (E_L2_edge_trim_text)
        if (E_L2_scrap_text!=""):
            print (E_L2_scrap_text)
        print(30 * "~")
    return()

## MAIN CODE

# Specify demand data set & data size
demand_data_set_code=1
demand_data_size=20

# Create Df_Demand with random generation
Df_demand_full = Random_Demand_Generator(length_of_full_demand_data)
Df_demand=Df_demand_full.head(demand_data_size)
Df_demand.set_index(Df_demand["D"].copy(), inplace=True)

# Create Df_Df_raw_material as an empty dataframe
Raw_Material_Column_Names=["R","W","T","L"]
Df_raw_material = pd.DataFrame(columns=Raw_Material_Column_Names, index=range(demand_data_size))
Df_raw_material["R"] = range(1, demand_data_size+1) # Set the "R" column values from 1 to demand_data_size
Df_raw_material.set_index(Df_raw_material["R"].copy(), inplace=True)

# Get specifications of raw material purchasing and process specifications
Raw_Material_Specification,Process_Specification,Waste_Costs,Operational_Costs=Specifications_and_Waste_Costs_Generator()

# Call Gurobi modelling

Optimization_results=Gurobi_Modelling(Df_demand,Df_raw_material)
Opt_Result_Visualization(Df_demand, Df_raw_material, Optimization_results)

exit()
