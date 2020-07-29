import numpy as np
import pandas as pd
import tqdm


def extract_ISIs_from_session(session_number: int, steinmetz_extra, steinmetz_meta):
    """Extracts ISIs from the given session using the data dictionaries and outputs DataFrame with metadata.

    Args:
        session_number (int): number of a session
        steinmetz_extra (steinmetz data): steinmetz extra dataset with spike times
        steinmetz_meta (steinmetz data): steinmetz dataset with metadata such as brain_area, contrasts and responses

    Returns:
        pandas.DataFrame: dataframe containing ISIs for each spike activation and corresponding metadata
    """    
    df = pd.DataFrame(columns=['ISIs', 'neuron_number', 'session_number', 'trial_number',
                               'mouse_name', 'brain_area', 'contrast_right', 'contrast_left', 
                               'response', 'feedback_type'])
    # extract relevant dict for the given session
    meta = steinmetz_meta[session_number]
    # get number of neurons and number of trials by looking at the shape of 'ss' array
    number_of_neurons, number_of_trials = steinmetz_extra[session_number]['ss'].shape
    # we know the mouse name now since each session corresponded to one mouse
    mouse_name = meta['mouse_name']  # per session
    # iterate through neurons
    # note that iteration through neurons and trials commutes
    # tqdm.trange just gives a nice progress bar to look at how our neurons are being processed
    # looks cool, you should check it out if you install `tqdm`
    for neuron_number in tqdm.trange(number_of_neurons, desc='neurons processed', leave=None, ncols=60):
        # create an array containing dictionaries for all trials to append to the dataframe
        trials_dicts = []
        # we know brain area which is given by the neuron number
        brain_area = meta['brain_area'][neuron_number]
        # iterate through trials
        for trial_number in range(number_of_trials):
            # get spike times for the given neuron number at that trial
            spikes = steinmetz_extra[session_number]['ss'][neuron_number][trial_number]
            
            # check if atleast two the neuron spikes atleast twice
            if (len(spikes) > 1):
                isi = np.diff(spikes)
            else:
                # skip trial for this neuron if no spikes detected
                # the code below will not be run and the for loop will skip to the next trial
                continue

            contrast_right = meta['contrast_right'][trial_number]
            contrast_left = meta['contrast_left'][trial_number]
            response = meta['response'][trial_number]
            feedback_type = meta['feedback_type'][trial_number]
            trial_dict = \
                {'ISIs': isi,
                 'neuron_number': neuron_number,
                 'session_number': session_number,
                 'trial_number': trial_number,
                 'mouse_name': mouse_name,
                 'brain_area': brain_area,
                 'contrast_right': contrast_right,
                 'contrast_left': contrast_left,
                 'response': response,
                 'feedback_type': feedback_type}
            # append dictionary that has info about the trial to the big array
            # big array contains info about all trials for a given neuron
            trials_dicts += [trial_dict]
        # after we finish iterating through all trials 
        df = df.append(trials_dicts)
        # this might be the problem???
    return df
