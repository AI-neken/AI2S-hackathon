#!/usr/bin/env python

import pandas as pd
from ortools.linear_solver import pywraplp
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Configuration
DEMAND_PATH = Path('data/02_input_target.csv')
CAPACITY_PATH = Path('data/02_input_capacity.csv')
SHIPMENT_COST_PATH = Path('data/02_03_input_shipmentsCost.csv')
PRODUCTION_COST_PATH = Path('data/03_input_productionCost.csv')
OUTPUT_PRODUCTION_PATH = Path('outputs/03_output_productionPlan_1239.csv')
OUTPUT_SHIPMENTS_PATH = Path('outputs/03_output_shipments_1239.csv')
COST_WEIGHT = 1e-8  # Weight for cost in objective function

# Type aliases for clarity
CountryName = str
ProductName = str
MonthName = str
DemandKey = Tuple[CountryName, ProductName, MonthName]
LocationPair = Tuple[CountryName, CountryName]
ProductionKey = Tuple[CountryName, ProductName]
Quantity = int
Cost = float
LoadFactor = float

def load_input_data(
    demand_path: Path, 
    capacity_path: Path, 
    shipment_cost_path: Path, 
    production_cost_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all input data from CSV files."""
    demand_df = pd.read_csv(demand_path)
    capacity_df = pd.read_csv(capacity_path)
    shipment_cost_df = pd.read_csv(shipment_cost_path)
    production_cost_df = pd.read_csv(production_cost_path)
    return demand_df, capacity_df, shipment_cost_df, production_cost_df

def prepare_optimization_data(
    demand_df: pd.DataFrame, 
    capacity_df: pd.DataFrame,
    shipment_cost_df: pd.DataFrame,
    production_cost_df: pd.DataFrame
) -> Tuple[
    Dict[DemandKey, Quantity], 
    Dict[CountryName, Quantity], 
    Dict[LocationPair, Cost],
    Dict[ProductionKey, Cost],
    List[CountryName], 
    List[MonthName], 
    List[DemandKey]
]:
    """Prepare data structures for the optimization model."""
    # Create demand dictionary with composite keys
    demand_df['Key'] = list(zip(demand_df['Country'], demand_df['Product'], demand_df['Month']))
    demand: Dict[DemandKey, Quantity] = dict(zip(demand_df['Key'], demand_df['Quantity']))
    
    # Create capacity dictionary
    capacity: Dict[CountryName, Quantity] = dict(zip(capacity_df['Country'], capacity_df['Monthly Capacity']))
    
    # Create transport costs dictionary
    transport_costs: Dict[LocationPair, Cost] = {}
    for _, row in shipment_cost_df.iterrows():
        key = (row['Origin'], row['Destination'])
        transport_costs[key] = row['Unit Cost']
    
    # Create production costs dictionary
    production_costs: Dict[ProductionKey, Cost] = {}
    for _, row in production_cost_df.iterrows():
        key = (row['Country'], row['Product'])
        production_costs[key] = row['Unit Cost']
    
    # Define plants, months, and demand keys
    plants: List[CountryName] = list(capacity.keys())
    months: List[MonthName] = sorted(demand_df['Month'].unique())
    demand_keys: List[DemandKey] = list(demand.keys())
    
    return demand, capacity, transport_costs, production_costs, plants, months, demand_keys

def setup_solver_model(
    demand: Dict[DemandKey, Quantity],
    capacity: Dict[CountryName, Quantity],
    transport_costs: Dict[LocationPair, Cost],
    production_costs: Dict[ProductionKey, Cost],
    plants: List[CountryName],
    months: List[MonthName],
    demand_keys: List[DemandKey],
    cost_weight: float = 1e-8
) -> Tuple[pywraplp.Solver, Dict[Tuple[CountryName, CountryName, ProductName, MonthName], Any], Any]:
    """Set up the linear programming model with both load balancing and cost minimization."""
    # Initialize the solver
    solver = pywraplp.Solver.CreateSolver('SAT')
    
    # Define the production variables for each origin, destination, product, and month
    x: Dict[Tuple[CountryName, CountryName, ProductName, MonthName], Any] = {}
    for i in plants:
        for (j, p, m) in demand_keys:
            x[i, j, p, m] = solver.NumVar(0, solver.infinity(), f'x_{i}_{j}_{p}_{m}')
    
    # Maximum load variable to be minimized
    max_load = solver.NumVar(0, 1, 'max_load')
    
    # Add demands constraints
    for (j, p, m) in demand_keys:
        solver.Add(solver.Sum(x[i, j, p, m] for i in plants) == demand[(j, p, m)])
    
    # Add maximum capacity constraints
    for i in plants:
        for m in months:
            monthly_prod = solver.Sum(x[i, j, p, m2] for (j, p, m2) in demand_keys if m2 == m)
            solver.Add(monthly_prod <= capacity[i])
    
    # Add maximum load constraints
    for i in plants:
        for m in months:
            monthly_prod = solver.Sum(x[i, j, p, m2] for (j, p, m2) in demand_keys if m2 == m)
            solver.Add(monthly_prod <= max_load * capacity[i])
    
    # Add shipment costs to the objective function
    total_cost = solver.Sum(
        x[i, j, p, m] * transport_costs.get((i, j), 0)  # Transport cost if found, else 0
        for i in plants for (j, p, m) in demand_keys
    )
    
    # Add production costs to the objective function
    total_cost += solver.Sum(
        x[i, j, p, m] * production_costs.get((i, p), 0)  # Production cost if found, else 0
        for i in plants for (j, p, m) in demand_keys
    )
    
    # Minimize the combined objective function: weighted cost + load balancing
    solver.Minimize(cost_weight * total_cost + max_load)
    
    return solver, x, max_load

def solve_model(solver: pywraplp.Solver) -> Tuple[int, float]:
    """Run the solver and return the solution status and objective value."""
    status = solver.Solve()
    objective_value = solver.Objective().Value() if status == pywraplp.Solver.OPTIMAL else None
    return status, objective_value

def process_results(
    status: int,
    objective_value: float,
    x: Dict[Tuple[CountryName, CountryName, ProductName, MonthName], Any],
    max_load: Any,
    output_shipments_path: Path,
    output_production_path: Path
) -> None:
    """Process solver results and save to output files if optimal solution found."""
    if status == pywraplp.Solver.OPTIMAL:
        max_load_value: LoadFactor = max_load.solution_value()
        print(f"Status: Optimal")
        print(f"Maximum load used: {max_load_value:.2%}")
        print(f"Total costs: {objective_value}")
        
        # Save the results
        shipment_data = []
        
        # Fill the shipment data with the results
        for (i, j, p, m), var in x.items():
            quantity = int(var.solution_value())
            if quantity > 0:  # Only include non-zero shipments
                shipment_data.append({
                    'Origin': i,
                    'Destination': j,
                    'Product': p,
                    'Month': m,
                    'Quantity': quantity
                })
        
        # Create dataframes for output
        shipment_plan = pd.DataFrame(shipment_data)
        
        # Create production plan by aggregating shipments
        if not shipment_plan.empty:
            production_plan = shipment_plan.groupby(['Origin', 'Product', 'Month'], as_index=False).agg({'Quantity': 'sum'})
            production_plan.rename(columns={'Origin': 'Country'}, inplace=True)
            
            # Save output files
            production_plan.to_csv(output_production_path, index=False)
            shipment_plan.to_csv(output_shipments_path, index=False)
            
            print(f"Results saved to {output_production_path} and {output_shipments_path}")
        else:
            print("No shipments in the solution.")
    else:
        print('No optimal solution found.')

def main() -> None:
    """Main execution function."""
    
    # Load input data
    demand_df, capacity_df, shipment_cost_df, production_cost_df = load_input_data(
        DEMAND_PATH, CAPACITY_PATH, SHIPMENT_COST_PATH, PRODUCTION_COST_PATH
    )
    
    # Prepare data structures for optimization
    demand, capacity, transport_costs, production_costs, plants, months, demand_keys = prepare_optimization_data(
        demand_df, capacity_df, shipment_cost_df, production_cost_df
    )
    
    # Set up the solver model
    solver, x, max_load = setup_solver_model(
        demand, capacity, transport_costs, production_costs, plants, months, demand_keys, COST_WEIGHT
    )
    
    # Solve the model
    status, objective_value = solve_model(solver)
    
    # Process results and save outputs
    process_results(
        status, objective_value, x, max_load, OUTPUT_SHIPMENTS_PATH, OUTPUT_PRODUCTION_PATH
    )

if __name__ == "__main__":
    main()