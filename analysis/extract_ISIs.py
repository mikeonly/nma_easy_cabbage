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
    for neuron_number in tqdm.trange(number_of_neurons, desc='neurons processed', leave=None, ncols=80):
        # create an array containing dictionaries for all trials to append to the dataframe
        trials_dicts = []
        # we know brain area which is given by the neuron number
        brain_area = meta['brain_area'][neuron_number]
        # iterate through trials
        for trial_number in range(number_of_trials):
            # get spike times for the given neuron number at that trial
            spikes = steinmetz_extra[session_number]['ss'][neuron_number][trial_number]
            
            # check if the neuron spikes at least twice
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


def extract_ISIs_from_session_version2(session_number: int, steinmetz_extra, steinmetz_meta):
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
    # extract relevant dict for the given session with brain region info
    meta = steinmetz_meta[session_number]
    # extract spike timing data
    ss = steinmetz_extra[session_number]['ss']  # array with shape (number of neurons, number of trials)
    # get number of neurons and number of trials by looking at the shape of 'ss' array
    number_of_neurons, number_of_trials = ss.shape
    isis = []  # all isis
    neuron_labels = []
    trial_labels = []
    brain_area_labels = []
    # iterate through neurons
    # note that iteration through neurons and trials commutes
    # tqdm.trange just gives a nice progress bar to look at how our neurons are being processed
    # looks cool, you should check it out if you install `tqdm`
    for neuron_number in tqdm.trange(number_of_neurons, desc='neurons processed', leave=None, ncols=80):
        # locate in which brain area is the neuron
        brain_area = meta['brain_area'][neuron_number]
        # array where we save all isis for a neuron
        # basically a list of all trials where neuron spiked more than once
        for trial_number in range(number_of_trials):
            spikes = ss[neuron_number, trial_number]
            if len(spikes) > 1:
                trial_isis = np.diff(spikes)
            else:
                # skip the trial if neuron spiked less than once
                continue
            isis.append(trial_isis)
            number_of_isis_in_trial = len(trial_isis)
            # array containing trial number labels
            trial_labels.append([trial_number] * number_of_isis_in_trial)
            # we add neuron number labels such that they have the same length as trial labels   
            neuron_labels.append([neuron_number] * number_of_isis_in_trial)
            # we add brain area labels such that they have the same length as trial labels
            brain_area_labels.append([brain_area] * number_of_isis_in_trial)

    # when we are done with all neurons, we should have huge isis arrays with all ISIs and 
    # we should have corresponding arrays with neuron labels, trial labels, and brain areas
    isis = np.concatenate(isis)[:]
    neuron_labels = np.concatenate(neuron_labels)[:]
    trial_labels = np.concatenate(trial_labels)[:]
    brain_area_labels = np.concatenate(brain_area_labels)[:]
    
    # finally construct the dataframe with all the processed data
    df = pd.DataFrame(data={'isi': isis,
                            'neuron': neuron_labels,
                            'trial': trial_labels,
                            'brain_area': brain_area_labels,})
    return df
