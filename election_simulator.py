import tkinter as tk
from tkinter import ttk, messagebox, BooleanVar
import numpy as np
import pandas as pd
import threading
import random
from queue import Queue
import sys
import os
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

# Determine if the application is a frozen/executable bundle
if getattr(sys, 'frozen', False):
    # If the app is running as a bundle, the PyInstaller bootloader
    # sets the app path into sys._MEIPASS.
    bundle_dir = sys._MEIPASS
else:
    # If running in a normal Python environment
    bundle_dir = os.path.dirname(os.path.abspath(__file__))

# Constants for candidate types
CANDIDATE_TYPES = ["All Extreme", "All Moderate", "Even Mix"]

# Updated Voting Probabilities
VOTING_PROBABILITIES = {
    'Red': {
        'Democrat': {
            'Extreme': 0.01,
            'Moderate': 0.05
        },
        'Republican': {
            'Extreme': 0.43,
            'Moderate': 0.43
        },
        'Third Party': 0.08
    },
    'Blue': {
        'Democrat': {
            'Extreme': 0.43,
            'Moderate': 0.43
        },
        'Republican': {
            'Extreme': 0.01,
            'Moderate': 0.05
        },
        'Third Party': 0.08
    }
}

# Define Candidate class
class Candidate:
    def __init__(self, party, type_, index_within_group):
        self.party = party
        self.type = type_
        self.index_within_group = index_within_group

# Function to generate candidates based on type
def generate_candidates(num, party, selection):
    candidates = []
    if selection == "All Extreme":
        types = ["Extreme"] * num
    elif selection == "All Moderate":
        types = ["Moderate"] * num
    elif selection == "Even Mix":
        types = ["Extreme" if i % 2 == 0 else "Moderate" for i in range(num)]
    
    # Group candidates by type
    type_indices = {'Extreme': 1, 'Moderate': 1}
    for t in types:
        idx = type_indices[t]
        candidates.append(Candidate(party, t, idx))
        type_indices[t] += 1
    return candidates

def linear_decay(n, k):
    return (2 * (n - k + 1)) / (n * (n + 1))

def quadratic_decay(n, k):
    normalization_factor = sum((n - j + 1) ** 2 for j in range(1, n + 1))
    return (n - k + 1) ** 2 / normalization_factor

# Function to select a vote based on voter type and candidates
def select_vote_vectorized(voter_types, candidate_probs):
    # voter_types: array of 'Red' or 'Blue'
    # candidate_probs: dict with keys as candidate indices and values as probabilities
    # Returns an array of vote counts per candidate
    # This function is not used in the optimized version
    pass

# Simulation Function (Optimized)
def run_simulation_optimized(num_simulations, population_size, red_percentage, 
                             num_republicans, rep_selection, num_democrats, dem_selection, 
                             num_third_parties, progress_queue, stop_flag):
    # Generate candidates
    candidates = []
    candidates += generate_candidates(num_republicans, 'Republican', rep_selection)
    candidates += generate_candidates(num_democrats, 'Democrat', dem_selection)
    candidates += generate_candidates(num_third_parties, 'Third Party', 'Even Mix')  # Assuming 3rd parties are always even mix

    # Dictionary to keep track of candidate counts per party and type
    party_type_counts = {}

    if not candidates:
        progress_queue.put("error: No candidates to simulate.")
        return

    num_candidates = len(candidates)

    # Initialize candidate probabilities
    candidate_probs_red = [0.0] * len(candidates)
    candidate_probs_blue = [0.0] * len(candidates)

    # Map candidate index to candidate object for easy access
    candidate_index_map = {idx: candidate for idx, candidate in enumerate(candidates)}

    # For each voter type (Red and Blue)
    for voter_type in ['Red', 'Blue']:
        voter_probs = VOTING_PROBABILITIES[voter_type]

        # Parties with candidates
        parties_with_candidates = set(c.party for c in candidates)
        
        # Calculate total available base probability for the voter type
        total_available_base_prob = 0.0
        
        # Calculate base probabilities for parties with candidates
        party_base_probs = {}
        for party in parties_with_candidates:
            if party == 'Third Party':
                total_available_base_prob += voter_probs['Third Party']
                party_base_probs[party] = voter_probs['Third Party']
            else:
                # Determine candidate types present for the party
                candidate_types_in_party = set(c.type for c in candidates if c.party == party)
                base_prob = sum([voter_probs[party][ctype] for ctype in candidate_types_in_party])
                print(f"{voter_type}.{party} base_prob = {base_prob}")
                total_available_base_prob += base_prob
                party_base_probs[party] = base_prob
        
        # Calculate adjustment factor
        adjustment_factor = 1.0 / total_available_base_prob
        print(f"{voter_type} total_available_base_prob  = {total_available_base_prob:.3f}\n")
        print(f"{voter_type} adjustment_factor  = {adjustment_factor:.3f}\n")

        # Adjust base probabilities
        for party in party_base_probs:
            party_base_probs[party] *= adjustment_factor

        # Proceed with candidate probability calculations
        # For Third Party candidates
        tp_candidates = [(idx, c) for idx, c in enumerate(candidates) if c.party == 'Third Party']
        num_tp_candidates = len(tp_candidates)
        if num_tp_candidates > 0:
            base_prob = party_base_probs['Third Party']
            prob_per_candidate = base_prob / num_tp_candidates
            for idx, candidate in tp_candidates:
                if voter_type == 'Red':
                    candidate_probs_red[idx] = prob_per_candidate
                else:
                    candidate_probs_blue[idx] = prob_per_candidate

        # For Republican and Democrat candidates
        for party in ['Republican', 'Democrat']:
            if party not in parties_with_candidates:
                continue
            party_candidates = [(idx, c) for idx, c in enumerate(candidates) if c.party == party]

            # Determine if voter is same party
            is_same_party = (voter_type == 'Red' and party == 'Republican') or (voter_type == 'Blue' and party == 'Democrat')

            if is_same_party:
                # Combine all candidates of this party into one group
                group_candidates = party_candidates
                n = len(group_candidates)
                # Determine candidate types present in the group
                candidate_types_in_group = set(c.type for idx, c in group_candidates)
                # Sum adjusted base probabilities for the candidate types present
                total_base_prob = sum([voter_probs[party][ctype] for ctype in candidate_types_in_group]) * adjustment_factor
                # Determine if funding disparity applies
                if party == 'Republican':
                    if rep_heavy_funding_var.get():
                        decay_function = quadratic_decay
                    elif rep_funding_var.get():
                        decay_function = linear_decay
                    else:
                        decay_function = None
                elif party == 'Democrat':
                    if dem_heavy_funding_var.get():
                        decay_function = quadratic_decay
                    elif dem_funding_var.get():
                        decay_function = linear_decay
                    else:
                        decay_function = None
                # Compute decay factors
                decay_factors = []
                if decay_function and n > 1:
                    for k in range(1, n + 1):
                        decay_factor = decay_function(n, k)
                        decay_factors.append(decay_factor)
                else:
                    decay_factors = [1.0] * n
                # Normalize decay factors
                sum_decay_factors = sum(decay_factors)
                normalized_decay_factors = [df / sum_decay_factors for df in decay_factors]
                # Multiply total base probability by decay factors and assign to candidates
                for (idx, candidate), decay_factor in zip(group_candidates, normalized_decay_factors):
                    prob = total_base_prob * decay_factor
                    if voter_type == 'Red':
                        candidate_probs_red[idx] = prob
                    else:
                        candidate_probs_blue[idx] = prob
            else:
                # Opposite-party candidates are grouped by type (Extreme and Moderate separately)
                candidate_types = set(c.type for idx, c in party_candidates)
                for type_ in candidate_types:
                    group_candidates = [(idx, c) for idx, c in party_candidates if c.type == type_]
                    n = len(group_candidates)
                    # Get adjusted base probability for this group
                    base_prob = voter_probs[party][type_] * adjustment_factor
                    # Determine if funding disparity applies
                    if party == 'Republican':
                        if rep_heavy_funding_var.get():
                            decay_function = quadratic_decay
                        elif rep_funding_var.get():
                            decay_function = linear_decay
                        else:
                            decay_function = None
                    elif party == 'Democrat':
                        if dem_heavy_funding_var.get():
                            decay_function = quadratic_decay
                        elif dem_funding_var.get():
                            decay_function = linear_decay
                        else:
                            decay_function = None
                    # Compute decay factors
                    decay_factors = []
                    if decay_function and n > 1:
                        for k in range(1, n + 1):
                            decay_factor = decay_function(n, k)
                            decay_factors.append(decay_factor)
                    else:
                        decay_factors = [1.0] * n
                    # Normalize decay factors
                    sum_decay_factors = sum(decay_factors)
                    normalized_decay_factors = [df / sum_decay_factors for df in decay_factors]
                    # Multiply base probability by decay factors and assign to candidates
                    for (idx, candidate), decay_factor in zip(group_candidates, normalized_decay_factors):
                        prob = base_prob * decay_factor
                        if voter_type == 'Red':
                            candidate_probs_red[idx] = prob
                        else:
                            candidate_probs_blue[idx] = prob

    # Verify that probabilities sum to 1
    total_prob_red = sum(candidate_probs_red)
    total_prob_blue = sum(candidate_probs_blue)

    if not np.isclose(total_prob_red, 1.0):
        progress_queue.put(f"error: probabilities for Red voters do not sum to 1 (sum={total_prob_red})")
        return
    if not np.isclose(total_prob_blue, 1.0):
        progress_queue.put(f"error: probabilities for Blue voters do not sum to 1 (sum={total_prob_blue})")
        return



    # ======= Pre-Normalization Logging =======
    print("\n--- Candidate Voting Probabilities ---")
    for idx, candidate in enumerate(candidates):
        party_abbr = {
            'Republican': 'Rep',
            'Democrat': 'Dem',
            'Third Party': 'TP'
        }.get(candidate.party, 'Other')
        candidate_name = f"{party_abbr}.Cand.{candidate.type}.{candidate.index_within_group}"
        prob_red = candidate_probs_red[idx]
        prob_blue = candidate_probs_blue[idx]
        print(f"{candidate_name} = Blue {prob_blue:.3f}, Red {prob_red:.3f}")
    print("--- End of Probabilities ---\n")
    # ======= Pre-Normalization Logging =======

    results = {}

    # Process simulations in batches to manage memory
    batch_size = 10000  # Try increasing to 5000 or 10000 if memory allows
    num_batches = (num_simulations + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        if stop_flag.is_set():
            progress_queue.put("cancelled")
            return

        current_batch_size = min(batch_size, num_simulations - batch_idx * batch_size)

        # Generate voter types
        voter_types = np.random.choice(
            ['Red', 'Blue'],
            size=(current_batch_size, population_size),
            p=[red_percentage, 1 - red_percentage]
        )

        # Initialize vote counts for each simulation in the batch
        vote_counts = np.zeros((current_batch_size, num_candidates), dtype=int)

        # Simulate votes
        for i in range(current_batch_size):
            if stop_flag.is_set():
                progress_queue.put("cancelled")
                return

            # For each voter type, select candidates based on probabilities
            voter_type_counts = np.unique(voter_types[i], return_counts=True)
            counts = dict(zip(voter_type_counts[0], voter_type_counts[1]))

            # Red voters
            num_red_voters = counts.get('Red', 0)
            if num_red_voters > 0:
                votes_red = np.random.choice(
                    num_candidates,
                    size=num_red_voters,
                    p=candidate_probs_red
                )
                vote_counts[i] += np.bincount(votes_red, minlength=num_candidates)

            # Blue voters
            num_blue_voters = counts.get('Blue', 0)
            if num_blue_voters > 0:
                votes_blue = np.random.choice(
                    num_candidates,
                    size=num_blue_voters,
                    p=candidate_probs_blue
                )
                vote_counts[i] += np.bincount(votes_blue, minlength=num_candidates)

        # Determine top 4 candidates for each simulation
        for i in range(current_batch_size):
            if stop_flag.is_set():
                progress_queue.put("cancelled")
                return

            top_indices = np.argsort(vote_counts[i])[-4:]
            top_parties = [candidates[idx].party for idx in top_indices]
            # Count the number of candidates from each party
            party_counts = {}
            for party in set(top_parties):
                party_counts[party] = top_parties.count(party)
            # Create a sorted tuple for consistent keys
            outcome = tuple(sorted(party_counts.items(), key=lambda x: (-x[1], x[0])))
            results[outcome] = results.get(outcome, 0) + 1

        # Update progress
        progress = ((batch_idx + 1) * current_batch_size) / num_simulations * 100
        progress_queue.put(progress)
        print(f"Starting batch {batch_idx + 1}/{num_batches}")


    progress_queue.put("done")
    progress_queue.put(results)

# Function to handle the simulation in a separate thread
def simulate(num_simulations, population_size, red_percentage, 
             num_republicans, rep_selection, num_democrats, dem_selection, 
             num_third_parties, progress_queue, stop_flag):
    try:
        run_simulation_optimized(
            num_simulations=num_simulations,
            population_size=population_size,
            red_percentage=red_percentage,
            num_republicans=num_republicans,
            rep_selection=rep_selection,
            num_democrats=num_democrats,
            dem_selection=dem_selection,
            num_third_parties=num_third_parties,
            progress_queue=progress_queue,
            stop_flag=stop_flag
        )
    except Exception as e:
        progress_queue.put(f"error: {e}")

# Function to start the simulation
def start_simulation():
    if run_button['state'] == 'disabled':
        messagebox.showwarning("Simulation Running", "A simulation is already running. Please wait until it completes or cancel it before starting a new one.")
        return

    # Reset the stop flag
    stop_flag.clear()
    # Enable the Stop button
    stop_button.config(state='normal')

    try:
        num_simulations = int(num_simulations_var.get())
        population_size = int(population_size_var.get())
        red_percentage = red_slider.get() / 100.0

        num_republicans = int(repub_var.get())
        rep_selection = rep_type_var.get()

        num_democrats = int(dem_var.get())
        dem_selection = dem_type_var.get()

        num_third_parties = int(tp_var.get())

        if num_simulations <= 0 or population_size <= 0:
            messagebox.showerror("Input Error", "Number of simulations and population size must be positive integers.")
            return

        if num_simulations > 1000000:
            if not messagebox.askyesno("Large Number of Simulations", 
                "Running a very large number of simulations may take considerable time and resources. Do you want to continue?"):
                return

        # Disable the run button
        run_button.config(state='disabled')

        # Clear previous results
        result_text.delete(1.0, tk.END)

        # Reset progress bar
        progress_var.set(0)

        # Create a queue to receive progress updates
        progress_queue = Queue()

        # Start simulation in a new thread
        simulation_thread = threading.Thread(target=simulate, args=(
            num_simulations, population_size, red_percentage, 
            num_republicans, rep_selection, num_democrats, dem_selection, 
            num_third_parties, progress_queue, stop_flag))
        simulation_thread.start()

        # Start monitoring the queue
        root.after(100, lambda: check_queue(progress_queue, num_simulations, result_text, run_button))

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")

# Function to check the queue for progress updates
def check_queue(progress_queue, num_simulations, result_text, run_button):
    try:
        while not progress_queue.empty():
            msg = progress_queue.get_nowait()
            if isinstance(msg, float):
                progress_var.set(msg)
            elif isinstance(msg, str):
                if msg.startswith("error"):
                    messagebox.showerror("Simulation Error", msg)
                    run_button.config(state='normal')
                    stop_button.config(state='disabled')
                    return
                elif msg == "done":
                    # Simulation completed successfully
                    # Do not return here; wait for results dictionary
                    pass
                elif msg == "cancelled":
                    result_text.insert(tk.END, "Simulation cancelled by user.\n")
                    run_button.config(state='normal')
                    stop_button.config(state='disabled')
                    return
            elif isinstance(msg, dict):
                # Simulation complete, process results
                results = msg
                # Calculate probabilities
                df_results = pd.DataFrame.from_dict(results, orient='index').reset_index()
                df_results.columns = ['Outcome', 'Count']
                df_results['Probability (%)'] = (df_results['Count'] / num_simulations) * 100
                df_results = df_results.sort_values(by='Probability (%)', ascending=False)

                # Prepare results string
                result_str = ""
                for _, row in df_results.iterrows():
                    outcome = row['Outcome']
                    # Format outcome
                    outcome_str = ', '.join([f"{count} {party}" + ("s" if count > 1 else "") for party, count in outcome])
                    probability = f"{row['Probability (%)']:.2f}%"
                    result_str += f"{outcome_str}: {probability}\n"

                # Update the result_text widget
                result_text.insert(tk.END, result_str)

                # Re-enable the run button and disable the stop button
                run_button.config(state='normal')
                stop_button.config(state='disabled')
                return
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        run_button.config(state='normal')
        stop_button.config(state='disabled')
        return

    # Schedule the next queue check
    root.after(100, lambda: check_queue(progress_queue, num_simulations, result_text, run_button))

# Function to reset the slider
def reset_slider():
    red_slider.set(50)

# Create the main window
root = ttkb.Window(themename="darkly")
root.title("Election Monte Carlo Simulator")

# Create a frame for inputs
input_frame = ttkb.Frame(root, padding="10")
input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# 1. Number of Simulated Elections (Dropdown)
ttkb.Label(input_frame, text="Number of Simulated Elections:").grid(row=0, column=0, sticky=tk.W, pady=5)
num_simulations_var = tk.StringVar()
num_simulations_dropdown = ttkb.Combobox(input_frame, textvariable=num_simulations_var, state='readonly')
num_simulations_dropdown['values'] = ["500", "1000", "5000", "10000", "50000", "100000"]
num_simulations_dropdown.current(3)  # Set "10000" as default
num_simulations_dropdown.grid(row=0, column=1, pady=5)

# 2. Size of Voting Population (Dropdown)
ttkb.Label(input_frame, text="Size of Voting Population:").grid(row=1, column=0, sticky=tk.W, pady=5)
population_size_var = tk.StringVar()
population_size_dropdown = ttkb.Combobox(input_frame, textvariable=population_size_var, state='readonly')
population_size_dropdown['values'] = ["50", "100", "500", "1000", "5000", "10000"]
population_size_dropdown.current(3)  # Set "1000" as default
population_size_dropdown.grid(row=1, column=1, pady=5)

# 3. Distribution slider
ttkb.Label(input_frame, text="Distribution of Voting Electorate:").grid(row=2, column=0, sticky=tk.W, pady=5)
slider_frame = ttkb.Frame(input_frame)
slider_frame.grid(row=2, column=1, pady=5)
red_slider = ttkb.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=200)
red_slider.set(50)
red_slider.pack(side=tk.LEFT)
slider_label = ttkb.Label(slider_frame, text="50% Red")
slider_label.pack(side=tk.LEFT, padx=10)

def update_slider_label(event):
    value = red_slider.get()
    red_percentage = int(value)
    blue_percentage = 100 - red_percentage
    if red_percentage > blue_percentage:
        slider_label.config(text=f"{red_percentage}% Red")
    elif blue_percentage > red_percentage:
        slider_label.config(text=f"{blue_percentage}% Blue")
    else:
        slider_label.config(text="50/50")

red_slider.bind("<Motion>", update_slider_label)
red_slider.bind("<ButtonRelease-1>", update_slider_label)

# Reset button for slider
reset_button = ttkb.Button(input_frame, text="Reset to 50/50", command=reset_slider)
reset_button.grid(row=2, column=2, padx=10, pady=5)

# Republican funding disparity variables
rep_funding_var = BooleanVar(value=True)  # Assume funding disparity (checked by default)
rep_heavy_funding_var = BooleanVar(value=False)  # Assume heavy funding disparity

# Democrat funding disparity variables
dem_funding_var = BooleanVar(value=True)  # Assume funding disparity (checked by default)
dem_heavy_funding_var = BooleanVar(value=False)  # Assume heavy funding disparity

# 4. Number and type of Republican candidates
ttkb.Label(input_frame, text="Number of Republican Candidates:").grid(row=3, column=0, sticky=tk.W, pady=5)
repub_var = tk.StringVar()
repub_dropdown = ttkb.Combobox(input_frame, textvariable=repub_var, state='readonly')
repub_dropdown['values'] = [str(i) for i in range(0,7)]
repub_dropdown.current(0)
repub_dropdown.grid(row=3, column=1, pady=5)

# Republican Candidates Funding Disparity Checkboxes
rep_funding_check = ttkb.Checkbutton(input_frame, text="Assume funding disparity",
                                     variable=rep_funding_var, onvalue=True, offvalue=False)
rep_funding_check.grid(row=3, column=2, padx=5, pady=5, sticky=tk.W)

rep_heavy_funding_check = ttkb.Checkbutton(input_frame, text="Assume heavy funding disparity",
                                           variable=rep_heavy_funding_var, onvalue=True, offvalue=False,
                                           command=lambda: toggle_funding('rep'))
rep_heavy_funding_check.grid(row=3, column=3, padx=5, pady=5, sticky=tk.W)

ttkb.Label(input_frame, text="Republican Candidate Type:").grid(row=4, column=0, sticky=tk.W, pady=5)
rep_type_var = tk.StringVar()
rep_type_dropdown = ttkb.Combobox(input_frame, textvariable=rep_type_var, state='readonly')
rep_type_dropdown['values'] = CANDIDATE_TYPES
rep_type_dropdown.current(0)
rep_type_dropdown.grid(row=4, column=1, pady=5)

# 5. Number and type of Democrat candidates
ttkb.Label(input_frame, text="Number of Democrat Candidates:").grid(row=5, column=0, sticky=tk.W, pady=5)
dem_var = tk.StringVar()
dem_dropdown = ttkb.Combobox(input_frame, textvariable=dem_var, state='readonly')
dem_dropdown['values'] = [str(i) for i in range(0,7)]
dem_dropdown.current(0)
dem_dropdown.grid(row=5, column=1, pady=5)

# Democrat Candidates Funding Disparity Checkboxes
dem_funding_check = ttkb.Checkbutton(input_frame, text="Assume funding disparity",
                                     variable=dem_funding_var, onvalue=True, offvalue=False)
dem_funding_check.grid(row=5, column=2, padx=5, pady=5, sticky=tk.W)

dem_heavy_funding_check = ttkb.Checkbutton(input_frame, text="Assume heavy funding disparity",
                                           variable=dem_heavy_funding_var, onvalue=True, offvalue=False,
                                           command=lambda: toggle_funding('dem'))
dem_heavy_funding_check.grid(row=5, column=3, padx=5, pady=5, sticky=tk.W)

ttkb.Label(input_frame, text="Democrat Candidate Type:").grid(row=6, column=0, sticky=tk.W, pady=5)
dem_type_var = tk.StringVar()
dem_type_dropdown = ttkb.Combobox(input_frame, textvariable=dem_type_var, state='readonly')
dem_type_dropdown['values'] = CANDIDATE_TYPES
dem_type_dropdown.current(0)
dem_type_dropdown.grid(row=6, column=1, pady=5)

def toggle_funding(party):
    if party == 'rep':
        if rep_heavy_funding_var.get():
            rep_funding_var.set(True)
            rep_funding_check.config(state='disabled')
        else:
            rep_funding_check.config(state='normal')
    elif party == 'dem':
        if dem_heavy_funding_var.get():
            dem_funding_var.set(True)
            dem_funding_check.config(state='disabled')
        else:
            dem_funding_check.config(state='normal')

# Initially disable the "Assume funding disparity" checkboxes if "Assume heavy funding disparity" is checked
toggle_funding('rep')
toggle_funding('dem')

# 6. Number of 3rd Party Candidates
ttkb.Label(input_frame, text="Number of 3rd Party Candidates:").grid(row=7, column=0, sticky=tk.W, pady=5)
tp_var = tk.StringVar()
tp_dropdown = ttkb.Combobox(input_frame, textvariable=tp_var, state='readonly')
tp_dropdown['values'] = [str(i) for i in range(0,7)]
tp_dropdown.current(0)
tp_dropdown.grid(row=7, column=1, pady=5)

# Progress Bar
progress_var = tk.DoubleVar()
progress_bar = ttkb.Progressbar(root, variable=progress_var, maximum=100, length=400)
progress_bar.grid(row=1, column=0, padx=10, pady=10)

# Global stop flag
stop_flag = threading.Event()

# Function to stop the simulation
def stop_simulation():
    stop_flag.set()

# Run and Stop Buttons Frame
buttons_frame = ttkb.Frame(root)
buttons_frame.grid(row=2, column=0, padx=10, pady=10)

# Run Elections Button
run_button = ttkb.Button(buttons_frame, text="Run Elections", command=start_simulation)
run_button.pack(side=tk.LEFT, padx=5)

# Stop Simulation Button
stop_button = ttkb.Button(buttons_frame, text="Cancel Simulation", command=stop_simulation, state='disabled')
stop_button.pack(side=tk.LEFT, padx=5)

# Results Display
ttkb.Label(root, text="Simulation Results:").grid(row=3, column=0, sticky=tk.W, padx=10)
result_text = tk.Text(root, height=15, width=80)
result_text.grid(row=4, column=0, padx=10, pady=10)

# Determine the appropriate icon file based on the platform
if sys.platform == "darwin":
    icon_file = "icon.png"
elif sys.platform == "win32":
    icon_file = "icon.ico"
else:
    icon_file = None

if icon_file:
    icon_path = os.path.join(bundle_dir, icon_file)

    if sys.platform == "win32":
        if os.path.exists(icon_path):
            try:
                # Set the window icon
                root.iconbitmap(icon_path)
            except Exception as e:
                print(f"Error loading iconbitmap: {e}")
    else:
        if os.path.exists(icon_path):
            try:
                root.iconphoto(False, tk.PhotoImage(file=icon_path))
            except Exception as e:
                print(f"Error loading iconphoto: {e}")

# Start the Tkinter event loop
root.mainloop()
