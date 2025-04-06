#!/usr/bin/env python

import pandas as pd
from ortools.linear_solver import pywraplp
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Configuration
DEMAND_PATH = Path('data/02_input_target.csv')
CAPACITY_PATH = Path('data/02_input_capacity.csv')
OUTPUT_SHIPMENTS_PATH = Path('outputs/02_output_shipments_1239.csv')
OUTPUT_PRODUCTION_PATH = Path('outputs/02_output_productionPlan_1239.csv')

# Type aliases for clarity
CountryName = str
ProductName = str
MonthName = str
DemandKey = Tuple[CountryName, ProductName, MonthName]
Quantity = int
LoadFactor = float

def load_input_data(demand_path: Path, capacity_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load demand and capacity input data from CSV files."""
    demand_df = pd.read_csv(demand_path)
    capacity_df = pd.read_csv(capacity_path)
    return demand_df, capacity_df

def prepare_optimization_data(
    demand_df: pd.DataFrame, 
    capacity_df: pd.DataFrame
) -> Tuple[Dict[DemandKey, Quantity], Dict[CountryName, Quantity], List[CountryName], List[MonthName], List[DemandKey]]:
    """Prepare data structures for the optimization model."""
    # Create demand dictionary with composite keys
    demand_df['Key'] = list(zip(demand_df['Country'], demand_df['Product'], demand_df['Month']))
    demand: Dict[DemandKey, Quantity] = dict(zip(demand_df['Key'], demand_df['Quantity']))
    
    # Create capacity dictionary
    capacity: Dict[CountryName, Quantity] = dict(zip(capacity_df['Country'], capacity_df['Monthly Capacity']))
    
    # Define plants, months, and demand keys
    plants: List[CountryName] = list(capacity.keys())
    months: List[MonthName] = sorted(demand_df['Month'].unique())
    demand_keys: List[DemandKey] = list(demand.keys())
    
    return demand, capacity, plants, months, demand_keys

def setup_solver_model(
    demand: Dict[DemandKey, Quantity],
    capacity: Dict[CountryName, Quantity],
    plants: List[CountryName],
    months: List[MonthName],
    demand_keys: List[DemandKey]
) -> Tuple[pywraplp.Solver, Dict[Tuple[CountryName, CountryName, ProductName, MonthName], Any], Any]:
    """Set up the linear programming model using OR-Tools."""
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
    
    # Minimize the objective function
    solver.Minimize(max_load)
    
    return solver, x, max_load

def solve_model(solver: pywraplp.Solver) -> int:
    """Run the solver and return the solution status."""
    return solver.Solve()

def process_results(
    status: int,
    solver: pywraplp.Solver,
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
            shipment_plan.to_csv(output_shipments_path, index=False)
            production_plan.to_csv(output_production_path, index=False)
            
            print(f"Results saved to {output_shipments_path} and {output_production_path}")
        else:
            print("No shipments in the solution.")
    else:
        print('No optimal solution found.')

def main() -> None:
    """Main execution function."""
    
    # Load input data
    demand_df, capacity_df = load_input_data(DEMAND_PATH, CAPACITY_PATH)
    
    # Prepare data structures for optimization
    demand, capacity, plants, months, demand_keys = prepare_optimization_data(demand_df, capacity_df)
    
    # Set up the solver model
    solver, x, max_load = setup_solver_model(demand, capacity, plants, months, demand_keys)
    
    # Solve the model
    status = solve_model(solver)
    
    # Process results and save outputs
    process_results(status, solver, x, max_load, OUTPUT_SHIPMENTS_PATH, OUTPUT_PRODUCTION_PATH)

if __name__ == "__main__":
    main()