import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading
import random

# Constants for candidate types
CANDIDATE_TYPES = ["All Extreme", "All Moderate", "Even Mix"]

# Historical Probability Estimates (Assumed Values)
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
    def __init__(self, party, type_):
        self.party = party
        self.type = type_

# Function to generate candidates based on type
def generate_candidates(num, party, selection):
    candidates = []
    if selection == "All Extreme":
        types = ["Extreme"] * num
    elif selection == "All Moderate":
        types = ["Moderate"] * num
    elif selection == "Even Mix":
        types = ["Extreme" if i % 2 == 0 else "Moderate" for i in range(num)]
    for t in types:
        candidates.append(Candidate(party, t))
    return candidates

# Simulation Function
def run_simulation(num_simulations, population_size, red_percentage, 
                   num_republicans, rep_selection, num_democrats, dem_selection, 
                   num_third_parties, progress_callback):
    
    results = {}
    for sim in range(1, num_simulations + 1):
        # Update progress
        if sim % (num_simulations // 100) == 0:
            progress_callback(sim / num_simulations * 100)

        # Generate candidates
        candidates = []
        candidates += generate_candidates(num_republicans, 'Republican', rep_selection)
        candidates += generate_candidates(num_democrats, 'Democrat', dem_selection)
        candidates += generate_candidates(num_third_parties, 'Third Party', 'Even Mix')  # Assuming 3rd parties are always even mix

        # If no candidates, skip
        if not candidates:
            continue

        # Initialize vote counts
        vote_counts = [0] * len(candidates)

        # Simulate voting
        for _ in range(population_size):
            # Determine voter type
            if random.random() < red_percentage:
                voter_type = 'Red'
            else:
                voter_type = 'Blue'
            
            # Select a candidate to vote for
            vote = select_vote(voter_type, candidates)
            if vote is not None:
                vote_counts[vote] += 1

        # Determine top 4 candidates
        top_indices = np.argsort(vote_counts)[-4:]
        top_parties = [candidates[i].party for i in top_indices]
        top_parties_sorted = sorted(top_parties, key=lambda x: x)  # Sort for consistency

        # Create outcome key
        outcome = tuple(sorted(top_parties, key=lambda x: x))
        results[outcome] = results.get(outcome, 0) + 1

    return results

# Function to select a vote based on voter type and candidates
def select_vote(voter_type, candidates):
    probabilities = []
    for candidate in candidates:
        if candidate.party == 'Republican':
            if candidate.type == 'Extreme':
                prob = VOTING_PROBABILITIES[voter_type]['Republican']['Extreme']
            else:
                prob = VOTING_PROBABILITIES[voter_type]['Republican']['Moderate']
        elif candidate.party == 'Democrat':
            if candidate.type == 'Extreme':
                prob = VOTING_PROBABILITIES[voter_type]['Democrat']['Extreme']
            else:
                prob = VOTING_PROBABILITIES[voter_type]['Democrat']['Moderate']
        else:  # Third Party
            prob = VOTING_PROBABILITIES[voter_type]['Third Party']
        probabilities.append(prob)
    
    total = sum(probabilities)
    if total == 0:
        return None  # No vote cast

    # Normalize probabilities
    probabilities = [p / total for p in probabilities]
    return np.random.choice(len(candidates), p=probabilities)

# Function to update the progress bar
def update_progress(progress_var, value):
    progress_var.set(value)

# Function to handle the simulation in a separate thread
def start_simulation():
    try:
        num_simulations = int(entry_num_simulations.get())
        population_size = int(entry_population_size.get())
        red_percentage = red_slider.get() / 100.0
        blue_percentage = 1 - red_percentage

        num_republicans = int(repub_var.get())
        rep_selection = rep_type_var.get()

        num_democrats = int(dem_var.get())
        dem_selection = dem_type_var.get()

        num_third_parties = int(tp_var.get())

        # Disable the run button
        run_button.config(state='disabled')

        # Clear previous results
        result_text.delete(1.0, tk.END)

        # Start simulation in a new thread
        simulation_thread = threading.Thread(target=simulate, args=(
            num_simulations, population_size, red_percentage, 
            num_republicans, rep_selection, num_democrats, dem_selection, 
            num_third_parties, progress_var, result_text, run_button))
        simulation_thread.start()
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")

def simulate(num_simulations, population_size, red_percentage, 
             num_republicans, rep_selection, num_democrats, dem_selection, 
             num_third_parties, progress_var, result_text, run_button):
    
    results = run_simulation(
        num_simulations=num_simulations,
        population_size=population_size,
        red_percentage=red_percentage,
        num_republicans=num_republicans,
        rep_selection=rep_selection,
        num_democrats=num_democrats,
        dem_selection=dem_selection,
        num_third_parties=num_third_parties,
        progress_callback=lambda x: update_progress(progress_var, x)
    )

    # Calculate probabilities
    df_results = pd.DataFrame.from_dict(results, orient='index').reset_index()
    df_results.columns = ['Outcome', 'Count']
    df_results['Probability (%)'] = (df_results['Count'] / num_simulations) * 100
    df_results = df_results.sort_values(by='Probability (%)', ascending=False)

    # Prepare results string
    result_str = ""
    for _, row in df_results.iterrows():
        outcome = ', '.join(row['Outcome'])
        probability = f"{row['Probability (%)']:.2f}%"
        result_str += f"{outcome}: {probability}\n"

    # Update the result_text widget
    result_text.insert(tk.END, result_str)

    # Re-enable the run button
    run_button.config(state='normal')

# Function to reset the slider
def reset_slider():
    red_slider.set(50)

# Create the main window
root = tk.Tk()
root.title("Election Monte Carlo Simulator")

# Create a frame for inputs
input_frame = ttk.Frame(root, padding="10")
input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# 1. Number of simulated elections
ttk.Label(input_frame, text="Number of Simulated Elections:").grid(row=0, column=0, sticky=tk.W, pady=5)
entry_num_simulations = ttk.Entry(input_frame)
entry_num_simulations.insert(0, "1000000")
entry_num_simulations.grid(row=0, column=1, pady=5)

# 2. Size of voting population
ttk.Label(input_frame, text="Size of Voting Population:").grid(row=1, column=0, sticky=tk.W, pady=5)
entry_population_size = ttk.Entry(input_frame)
entry_population_size.insert(0, "10000")
entry_population_size.grid(row=1, column=1, pady=5)

# 3. Distribution slider
ttk.Label(input_frame, text="Distribution of Voting Electorate:").grid(row=2, column=0, sticky=tk.W, pady=5)
slider_frame = ttk.Frame(input_frame)
slider_frame.grid(row=2, column=1, pady=5)
red_slider = ttk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=200)
red_slider.set(50)
red_slider.pack(side=tk.LEFT)
slider_label = ttk.Label(slider_frame, text="50% Red")
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
reset_button = ttk.Button(input_frame, text="Reset to 50/50", command=reset_slider)
reset_button.grid(row=2, column=2, padx=10, pady=5)

# 4. Number and type of Republican candidates
ttk.Label(input_frame, text="Number of Republican Candidates:").grid(row=3, column=0, sticky=tk.W, pady=5)
repub_var = tk.StringVar()
repub_dropdown = ttk.Combobox(input_frame, textvariable=repub_var, state='readonly')
repub_dropdown['values'] = [str(i) for i in range(0,7)]
repub_dropdown.current(0)
repub_dropdown.grid(row=3, column=1, pady=5)

ttk.Label(input_frame, text="Republican Candidate Type:").grid(row=4, column=0, sticky=tk.W, pady=5)
rep_type_var = tk.StringVar()
rep_type_dropdown = ttk.Combobox(input_frame, textvariable=rep_type_var, state='readonly')
rep_type_dropdown['values'] = CANDIDATE_TYPES
rep_type_dropdown.current(0)
rep_type_dropdown.grid(row=4, column=1, pady=5)

# 4. Number and type of Democrat candidates
ttk.Label(input_frame, text="Number of Democrat Candidates:").grid(row=5, column=0, sticky=tk.W, pady=5)
dem_var = tk.StringVar()
dem_dropdown = ttk.Combobox(input_frame, textvariable=dem_var, state='readonly')
dem_dropdown['values'] = [str(i) for i in range(0,7)]
dem_dropdown.current(0)
dem_dropdown.grid(row=5, column=1, pady=5)

ttk.Label(input_frame, text="Democrat Candidate Type:").grid(row=6, column=0, sticky=tk.W, pady=5)
dem_type_var = tk.StringVar()
dem_type_dropdown = ttk.Combobox(input_frame, textvariable=dem_type_var, state='readonly')
dem_type_dropdown['values'] = CANDIDATE_TYPES
dem_type_dropdown.current(0)
dem_type_dropdown.grid(row=6, column=1, pady=5)

# 5. Number of 3rd Party Candidates
ttk.Label(input_frame, text="Number of 3rd Party Candidates:").grid(row=7, column=0, sticky=tk.W, pady=5)
tp_var = tk.StringVar()
tp_dropdown = ttk.Combobox(input_frame, textvariable=tp_var, state='readonly')
tp_dropdown['values'] = [str(i) for i in range(0,7)]
tp_dropdown.current(0)
tp_dropdown.grid(row=7, column=1, pady=5)

# Progress Bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100, length=400)
progress_bar.grid(row=1, column=0, padx=10, pady=10)

# Run Elections Button
run_button = ttk.Button(root, text="Run Elections", command=start_simulation)
run_button.grid(row=2, column=0, padx=10, pady=10)

# Results Display
ttk.Label(root, text="Simulation Results:").grid(row=3, column=0, sticky=tk.W, padx=10)
result_text = tk.Text(root, height=15, width=80)
result_text.grid(row=4, column=0, padx=10, pady=10)

# Start the Tkinter event loop
root.mainloop()
