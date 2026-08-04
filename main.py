import yaml
import pandas as pd
import scipy.stats as st
from src.distribution_fitting import DistributionFitter
from src.simulation import RiskSimulator
from src.reliability import EventCombinator
from src.analysis import RiskAnalyzer, RiskVisualizer

def run_pipeline():
    # 1. Load the Configuration
    print("Loading configuration...")
    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # 2. Get parameters and paths from config
    # Paths for input .csv files
    market_demand_path = config['data_paths']['raw']['market_demand']
    base_capacity_path = config['data_paths']['raw']['base_capacity']
    crf_input_path = config['data_paths']['raw']['crf_input']
    single_input_path =config['data_paths']['raw'].get('single_input', 'data/raw/single_input.csv') # Fallback just in case I missed

    # Paths for output of results
    reports_output_path = config['data_paths']['output']['reports']
    figures_output_path = config['data_paths']['output']['figures']

    # Parameters for simulation.py
    iterations = config['project']['iterations']
    correlation = config['simulation']['correlation']
    fr_lower = config['simulation']['distributions']['demand_ratio']['truncation']['lower_bound']
    fr_upper = config['simulation']['distributions']['demand_ratio']['truncation']['upper_bound']
    bc_lower = config['simulation']['distributions']['base_capacity']['truncation']['lower_bound']
    candidate_dists = config['distribution_fitter']['default_distributions']

    # Parameters for analysis.py
    contract_forecast = config['simulation']['contract_forecast']
    conversion_factor = config['simulation']['conversion_factor']

    # Parameters for reliability
    num_fails = config['reliability']['num_fails']
    crf_rest = config['reliability']['crf_rest']

    # 3. Fit distibutions for market and base capacity
    print("Loading raw data for market and base capcacity...")
    df_market = pd.read_csv(market_demand_path)
    df_capacity = pd.read_csv(base_capacity_path)   

    print("Fitting distributions...")
    demand_fitter = DistributionFitter(df_market['demand_ratio'], data_name="Market Demand", candidate_dists=candidate_dists)
    capacity_fitter = DistributionFitter(df_capacity['capacity'], data_name="Base Capacity", candidate_dists=candidate_dists)

    demand_fitter.fit()
    capacity_fitter.fit()

    best_demand = demand_fitter.get_best_config()
    best_capacity = capacity_fitter.get_best_config()

    print(f"Top Demand Dist: {best_demand['distribution']} with params {best_demand['parameters']}")
    print(f"Top Capacity Dist: {best_capacity['distribution']} with params {best_capacity['parameters']}")

    demand_model = getattr(st, best_demand['distribution'])
    capacity_model = getattr(st, best_capacity['distribution'])

    fr_scipy_dist = demand_model(*best_demand['parameters'])
    bc_scipy_dist = capacity_model(*best_capacity['parameters'])

    # 4. Run Monte Carlo simulation
    print("Running Monte Carlo simulation...")
    sim = RiskSimulator(iterations=iterations, correlation=correlation)  

    fr_sample, bc_sample, _, _ = sim.run_multivariate_sample(
        fr_dist=fr_scipy_dist,
        bc_dist=bc_scipy_dist,
        fr_bounds=(fr_lower, fr_upper),
        bc_lower_bound=bc_lower
    )

    # 5. Simulate Capability-Demand Interference & Calculate Shortfalls
    # Loading single unit failures (%) and CRF assigned to each events 
    print("Loading single unit failures and CRFs per event...")
    df_singles = pd.read_csv(single_input_path)
    df_crf = pd.read_csv(crf_input_path)

    # Calculate event probability and merge with CRF
    print("Calculating event (failure) probabilities...")
    combinator = EventCombinator(df_raw=df_singles, num_fails=num_fails)
    df_crf_prob = combinator.merge_crf(df_crf, crf_rest=crf_rest)

    # Save the merged dataframe back to the input folder
    merged_crf_output_path = 'data/raw/merged_crf_prob.csv'
    df_crf_prob.to_csv(merged_crf_output_path, index=False)
    print(f"Merged events, CRFs, and probabilities saved to {merged_crf_output_path}")

    print("Simulating daily demand and shortfalls...")
    sim_data = sim.simulate_day_demand(
        fr_sample=fr_sample,
        bc_sample=bc_sample,
        df_crf_prob=df_crf_prob,
        contract_forecast=contract_forecast,
        conversion_factor=conversion_factor
    )

    # 6. Analyze risk and export reports
    print("Analyzing risk metrics...")
    df_results = RiskAnalyzer.compute_metrics(sim_data)

    df_results.to_csv(reports_output_path, index=False)
    print(f"Risk metrics successfully saved to: {reports_output_path}")
    print("\n Summary Risk Metrics:")
    print(df_results.head())

    # 7. Generate visuals
    print("Generating capacity vs. demand distributions...")
    RiskVisualizer.plot_capacity_distributions(
        sim_data=sim_data,
        title="Capacity vs. Demand Distributions",
        save_path=f"{figures_output_path}capacity_distribution.png"
    )

    print("Pipeline execution completed.")

if __name__ == "__main__":
    run_pipeline()
