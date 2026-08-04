import pytest
import pandas as pd
from src.reliability import EventCombinator

@pytest.fixture
def dummy_unit_df():
    return pd.DataFrame({
        'unit_id': ['Unit1', 'Unit2', 'Unit3'],
        'unit_outage_freq': [0.1, 0.05, 0.2]
    })

def test_zero_failure_probability(dummy_unit_df):
    """ Test zero unit failure event probability """
    combinator = EventCombinator(dummy_unit_df, num_fails=1)
    df_events = combinator.generate_events()

    # Check that 0 faiures + individual unit failure events were generated (1 + 3 (singles) = 4)
    assert len(df_events) == 4

    zero_fail_prob = df_events.loc[df_events['unit_combinations'] == "[]", 'event_prob'].values[0]
    assert pytest.approx(zero_fail_prob,  0.0001) == 0.684

def test_one_failure_probability(dummy_unit_df):
    """ Test one unit failure events' probabilities """
    combinator = EventCombinator(dummy_unit_df, num_fails=1)
    df_events = combinator.generate_events()

    # Check that 0 faiures + individual unit failure events were generated (1 + 3 (singles) = 4)
    assert len(df_events) == 4

    one_fail_row = df_events.loc[df_events['unit_combinations'] == "['Unit1']"]
    actual_prob = one_fail_row['event_prob'].values[0]
    assert pytest.approx(actual_prob, 0.001) == 0.076

def test_two_failures_probability(dummy_unit_df):
    """ Test two units failures event probability """
    # Intitialize your EventCombinator class
    combinator = EventCombinator(dummy_unit_df, num_fails=2)

    # Generate the failure events and their probabilities
    df_events = combinator.generate_events()

    two_fails_row = df_events[df_events['unit_combinations'] == "['Unit1', 'Unit2']"]
    actual_prob = two_fails_row['event_prob'].values[0]
    assert pytest.approx(actual_prob, 0.0001) == 0.004

def test_sample_space_coverage(dummy_unit_df):
    # Initialize class EventCombinator
    combinator = EventCombinator(dummy_unit_df, num_fails=1)
    # Calculate the sample space coverage
    prob = combinator.calculate_sample_space_coverage()
    assert pytest.approx(prob, 0.0001) == 0.967

def test_single_failure_merge_crf(dummy_unit_df):
    df_crf = pd.DataFrame({
        'unit_combinations' : ["()", "('Unit1')", "('Unit2')", "('Unit3')"],
        'crf' : [1.0, 0.99, 0.98, 0.97]
    })

    # Initialize combinator of events from zero and single failures with its probability
    combinator = EventCombinator(dummy_unit_df, num_fails=1)

    # Merge CRFs with probabilities
    df_merged = combinator.merge_crf(df_crf, crf_rest=0.75)

    assert len(df_merged) == 5

    # Test one of the events
    gen_a_row = df_merged[df_merged['unit_combinations'] == "['Unit1']"]
    assert not gen_a_row.empty, "The merge failed to find 'Unit1"
    assert gen_a_row['event_prob'].values[0] == pytest.approx(0.076)
    assert gen_a_row['crf'].values[0] == 0.99

    # Test the greater than event
    gt_row = df_merged[df_merged['unit_combinations'] == "['GT']"]
    assert not gt_row.empty, "The greater than 'GT' row wasn't created"
    assert gt_row['event_prob'].values[0] == pytest.approx(0.033)
    assert gt_row['crf'].values[0] == 0.75