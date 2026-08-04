import pandas as pd
import numpy as np
from itertools import combinations

class EventCombinator:
    """
    Calculates each failure event's likelihood then combines each with its system impact (severity). 
    Events are independent and mutually exclusive.
    """
    def __init__(self, df_raw: pd.DataFrame, num_fails: int = 2):
        self.df_raw = df_raw # Individual unit's / equipment's failure rate
        self.num_fails = num_fails # Concurrent number of unit/equipment failures
        self.event_df = None # Combinations of events and their likelihood

    def generate_events(self) -> pd.DataFrame:
        """ 
        Creates a dataframe of all combinations of unit failure events and 
        calculate each event joint probabilites. Output looks like:
        {'unit_combinations': ["[]", "['Unit1']", "['Unit2']", "['Unit1', 'Unit2']", ...],
         'event_prob': [0.65, 0.05, 0.04, 0.01, ...]} 
        """
        dict_1 = {} # {['Unit 1', 'Unit 2']: 0.02} <- units 1 and 2 fail with 2% chance

        unit_ids = self.df_raw['unit_id'].to_list() # List form ['Unit1', 'Unit2', ...]
        outage_freqs = self.df_raw['unit_outage_freq'].values # NP array form array([0.1, 0.05, ...])

        # Zero failure [] -> get event name -> calc event prob
        # One failure [(0,), (1,), (2,)] -> [Unit 1, Unit 2, Unit 3] -> [2%, 5%, 3%]
        for n in range(self.num_fails + 1):
            # As combination of unit failures are generated (e.g., (0, 1))
            # get the unit IDs in String form and calculate the joint prob.
            for idx_comb in combinations(range(len(unit_ids)), n):

                unit_names = [unit_ids[i] for i in idx_comb] # Change (0,1) -> ['Unit 1', 'Unit 2']
                key = str(unit_names) # ['Unit 1'] -> "['Unit 1']" for map's key

                # Set a boolean mask used to calculate the event probability
                mask = np.zeros(len(outage_freqs), dtype=bool) # [false false false]
                mask[list(idx_comb)] = True # idx_comb = (0, 1) -> [True, True, False]

                # np.where(use boolean mask as condition, if_true use prob failure, if_false use avail/reliab)
                prob = np.prod(np.where(mask, outage_freqs, 1 - outage_freqs)) # np.prod() to calc the even prob
                dict_1[key] = prob
                
        # Convert dict to DataFrame
        self.event_df = pd.DataFrame(
            list(dict_1.items()),
            columns=['unit_combinations', 'event_prob']
        )
        return self.event_df

    def calculate_sample_space_coverage(self) -> float:
        """ Calcualtes the total probability coverage of the combinations. """
        if self.event_df is None:
            self.generate_events()
        return self.event_df['event_prob'].sum()

    def merge_crf(self, df_crf: pd.DataFrame, crf_rest: float = 0.75) -> pd.DataFrame:
        """ 
        Merges each event's impact and its probability of occuring. Then adds the remaining 
        sample space that wasn't covered by the events (i.e., greater than num_fails) as a single 
        block of event with its own impact (crf_rest). This ensures 100% sample space coverage.
        
        @param df_crf: Event's impact to system capability (e.g., all units avail. = 100%).
        @param crf_rest: Estimate of the system impact with concurrent failures greater than num_fails.
        """
        if self.event_df is None:
            self.generate_events()

        # Format unit_combinations names to match between df_crf and event_df
        df_crf_clean = df_crf.assign(
            unit_combinations=lambda d: d.unit_combinations.str.replace(r"\(|\)|'|,", "", regex=True).str.split()
        ).astype({'unit_combinations': 'str'})

        # Make sure the unit_combinations is a String type
        prob_df_str = self.event_df.astype({'unit_combinations': 'str'})
        merged = df_crf_clean.merge(prob_df_str, on='unit_combinations', how='left')

        # Insert the last row of any unit failure combinations greater than num_fails 
        prob_fail_gt = 1.0 - self.calculate_sample_space_coverage() # Get the remaining sample space
        merged.loc[len(merged)] = {'unit_combinations': "['GT']", 'crf': crf_rest, 'event_prob': prob_fail_gt}
        return merged


