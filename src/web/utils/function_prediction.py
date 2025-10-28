import gradio as gr
import pandas as pd
import os
import sys
import subprocess
import time
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Generator, Optional, Tuple, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
from dataclasses import dataclass
import re
import json

def load_constant():
    """Load constant values from config files"""
    try:
        return json.load(open("src/constant.json"))
    except Exception as e:
        print(f"Error loading constant.json: {e}")
        return {"error": f"Failed to load constant.json: {str(e)}"}

constant = load_constant()


def get_api_key(ai_provider: str, user_input_key: str = "") -> Optional[str]:
    """Get API key based on provider and user input."""
    model_config = AI_MODELS.get(ai_provider, {})
    env_key = model_config.get("env_key")
    
    if env_key:
        env_api_key = os.getenv(env_key)
        if env_api_key and env_api_key.strip():
            return env_api_key.strip()
        if user_input_key and user_input_key.strip():
            return user_input_key.strip()
        return None
    
    if user_input_key and user_input_key.strip():
        return user_input_key.strip()
    return None


@dataclass
class AIConfig:
    """Configuration for AI API calls."""
    api_key: str
    ai_model_name: str
    api_base: str
    model: str

def call_ai_api(config: AIConfig, prompt: str) -> str:
    """Make API call to AI service."""
    if config.ai_model_name == "ChatGPT":
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert protein scientist. Provide clear, structured, and insightful analysis based on the data provided. Do not ask interactive questions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        endpoint = f"{config.api_base}/chat/completions"
        
    elif config.ai_model_name == "Gemini":
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "contents": [{
                "parts": [{
                    "text": f"You are an expert protein scientist. {prompt}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 2000
            }
        }
        endpoint = f"{config.api_base}/models/{config.model}:generateContent?key={config.api_key}"
        
    else:
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert protein scientist. Provide clear, structured, and insightful analysis based on the data provided. Do not ask interactive questions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        endpoint = f"{config.api_base}/chat/completions"
    
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        response_json = response.json()
        
        if config.ai_model_name == "Gemini":
            if "candidates" in response_json and len(response_json["candidates"]) > 0:
                return response_json["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "❌ Gemini API returned empty response"
        else:
            return response_json["choices"][0]["message"]["content"]
            
    except requests.exceptions.RequestException as e:
        return f"❌ Network error: {str(e)}"
    except KeyError as e:
        return f"❌ API response format error: {str(e)}"
    except Exception as e:
        return f"❌ API call failed: {str(e)}"
    
def check_ai_config_status(ai_provider: str, user_api_key: str) -> tuple[bool, str]:
    """Check AI configuration status and return validity and message."""
    model_config = AI_MODELS.get(ai_provider, {})
    env_key = model_config.get("env_key")
    
    if env_key:
        env_api_key = os.getenv(env_key)
        if env_api_key and env_api_key.strip():
            return True, "✓ Using the Provided API Key"
        elif user_api_key and user_api_key.strip():
            return True, "✓ The server will not save your API Key"
        else:
            return False, "⚠ No API Key found in .env file"
    else:
        if user_api_key and user_api_key.strip():
            return True, "✓ Manual API Key provided"
        else:
            return False, "⚠ Manual API Key required"  

def generate_expert_analysis_prompt_residue(results_df: pd.DataFrame, task: str) -> str:
    """
    Generates a prompt for an LLM to analyze residue-level prediction results.
    """
    prompt = f"""
        You are a senior protein biochemist and structural biologist with deep expertise in enzyme mechanisms and protein engineering. 
        A colleague has just brought you the following prediction results identifying key functional residues, specifically for the task '{task}'.
        Please analyze these residue-specific results from a practical, lab-focused perspective:

        {results_df.to_string(index=False)}

        Provide a concise, practical analysis in a single paragraph, focusing ONLY on:
        1. The predicted functional role of the individual high-scoring residues (e.g., activity site, binding site, conserved site, and motif).
        2. The significance of the confidence scores (i.e., how certain are we that a specific residue is critical for the function).
        3. Specific experimental validation steps, especially which residues are the top candidates for site-directed mutagenesis to confirm their roles.
        4. Any notable patterns or outliers (e.g., a cluster of high-scoring residues forming a potential active site, or a surprising critical residue far from the expected functional region).
        5. Do not output formatted content, just one paragraph is sufficient
        Use simple, direct language that a bench scientist would use. Do NOT mention:
        - The model used or its training data.
        - Any computational or implementation details.
        - Complex statistical concepts beyond confidence.

        Keep your response under 200 words and speak as if you are discussing next steps for an experiment with a colleague.
        """
    return prompt

def generate_expert_analysis_prompt(results_df: pd.DataFrame, task: str) -> str:
    protein_count = len(results_df)

    prompt = f"""
        You are a senior protein biochemist with extensive laboratory experience. A colleague has just shown you protein function prediction results for the task '{task}'. 
        Please analyze these results from a practical biologist's perspective:
        {results_df.to_string(index=False)}
        Provide a concise, practical analysis focusing ONLY on:
        0. The task '{task}' with "Stability" and "Optimum temperature" are regression task
        1. What the prediction results mean for each protein
        2. The biological significance of the confidence scores
        3. Practical experimental recommendations based on these predictions
        4. Any notable patterns or outliers in the results
        5. Do not output formatted content, just one paragraph is sufficient
        Use simple, clear language that a bench scientist would appreciate. Do NOT mention:
        - Training datasets or models
        - Technical implementation details  
        - Computational methods
        - Statistical concepts beyond confidence scores
        Keep your response under 200 words and speak as if you're having a conversation with a colleague in the lab.
        """
    return prompt

def format_expert_response(ai_response: str) -> str:
    """Format AI response as HTML with expert avatar and speech bubble."""
    escaped_response = ai_response.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
    
    html = f"""
    <div style="min-height: 150px; padding: 10px; display: flex; align-items: flex-end; font-family: Arial, sans-serif;">
        <div style="margin-right: 15px; text-align: center;">
            <div style="width: 60px; height: 60px; background-color: #4A90E2; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 24px; color: white; border: 3px solid #357ABD;">
                👨‍🔬
            </div>
            <div style="font-size: 12px; color: #666; margin-top: 5px; font-weight: bold;">
                Expert
            </div>
        </div>
        <div style="flex: 1; position: relative;">
            <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 15px; padding: 15px; position: relative; max-width: 90%; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="position: absolute; left: -10px; bottom: 20px; width: 0; height: 0; border-top: 10px solid transparent; border-bottom: 10px solid transparent; border-right: 10px solid #f8f9fa;"></div>
                <div style="position: absolute; left: -11px; bottom: 20px; width: 0; height: 0; border-top: 10px solid transparent; border-bottom: 10px solid transparent; border-right: 10px solid #dee2e6;"></div>
                <div style="font-size: 15px; line-height: 1.4; color: #333;">
                    {escaped_response}
                </div>
            </div>
        </div>
    </div>
    """
    return html

def generate_mutation_ai_prompt(results_df: pd.DataFrame, model_name: str, function_selection: str) -> str:
    """Generate AI prompt for mutation analysis."""
    if 'Mutant' not in results_df.columns:
        return "Error: 'Mutant' column not found."
    
    score_col = next(
        (col for col in results_df.columns if 'Prediction Rank' in col.lower()), 
        results_df.columns[1] if len(results_df.columns) > 1 else None
    )
    
    if not score_col:
        return "Error: Score column not found."

    num_rows = len(results_df)
    top_count = max(20, int(num_rows * 0.05)) if num_rows >= 5 else num_rows
    top_mutations_str = results_df.head(top_count)[['Mutant', score_col]].to_string(index=False)
    
    return f"""
        Please act as an expert protein engineer and analyze the following mutation prediction results generated by the '{model_name}' model for the function '{function_selection}'.
        A deep mutational scan was performed. The results are sorted from most beneficial to least beneficial based on the '{score_col}'. Below are the top 5% of mutations.

        ### Top 5% Predicted Mutations (Potentially Most Beneficial):
        ```
        {top_mutations_str}
        ```

        ### Your Analysis Task:
        Based on this data, provide a structured scientific analysis report that includes the following sections:
        1. Executive Summary: Briefly summarize the key findings. Are there clear hotspot regions?
        2. Analysis of Beneficial Mutations: Discuss the top mutations and their potential biochemical impact on '{function_selection}'.
        3. Analysis of Detrimental Mutations: What do the most harmful mutations suggest about critical residues?
        4. Recommendations for Experimentation: Suggest 3-5 specific mutations for validation, with justifications.
        5. Do not output formatted content, just one paragraph is sufficient
        Provide a concise, clear, and insightful report in a professional scientific tone, summarize the above content into 1-2 paragraphs and output unformatted content.
        """

def generate_ai_summary_prompt(results_df: pd.DataFrame, task: str, model: str) -> str:
    """Generate AI prompt for function prediction summary."""
    prompt = f"""
        You are a senior protein biochemist with extensive laboratory experience. A colleague has just shown you protein function prediction results for the task '{task}' on model '{model}'. 
        Please analyze these results from a practical biologist's perspective:
        {results_df.to_string(index=False)}
        Provide a concise, practical analysis focusing ONLY on:
        0. The task '{task}' with "Stability" and "Optimum temperature" are regression task
        1. What the prediction results mean for each protein
        2. The biological significance of the confidence scores
        3. Practical experimental recommendations based on these predictions
        4. Any notable patterns or outliers in the results
        5. Do not output formatted content, just one paragraph is sufficient
        Use simple, clear language that a bench scientist would appreciate. Do NOT mention:
        - Training datasets or models
        - Technical implementation details  
        - Computational methods
        - Statistical concepts beyond confidence scores
        Keep your response under 200 words and speak as if you're having a conversation with a colleague in the lab.
        """
    return prompt

def run_zero_shot_prediction(model_type: str, model_name: str, file_path: str) -> Tuple[str, pd.DataFrame]:
    """Run zero-shot mutation prediction."""
    try:
        temp_dir = Path("temp_outputs")
        temp_dir_ = temp_dir / "Zero_shot_result"
        timestamp = str(int(time.time()))
        sequence_dir = temp_dir_ / timestamp
        sequence_dir.mkdir(parents=True, exist_ok=True)
        output_csv = sequence_dir / f"{model_type}.csv"
        script_name = MODEL_MAPPING_ZERO_SHOT.get(model_name)
        
        if not script_name:
            return f"Error: Model '{model_name}' has no script.", pd.DataFrame()

        script_path = f"src/mutation/models/{script_name}.py"
        if not os.path.exists(script_path):
            return f"Script not found: {script_path}", pd.DataFrame()
        
        file_argument = "--pdb_file" if model_type == "structure" else "--fasta_file"
        cmd = [
            sys.executable, script_path, 
            file_argument, file_path, 
            "--output_csv", output_csv
        ]
        
        subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True, 
            encoding='utf-8', 
            errors='ignore'
        )

        if os.path.exists(output_csv):
            df = pd.read_csv(output_csv)
            os.remove(output_csv)
            return "Prediction completed successfully!", df
        
        return "Prediction finished but no output file was created.", pd.DataFrame()
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or "Unknown subprocess error"
        return f"Prediction script failed: {error_msg}", pd.DataFrame()
    except Exception as e:
        return f"An unexpected error occurred: {e}", pd.DataFrame()

def parse_fasta_paste_content(fasta_content):
    if not fasta_content or not fasta_content.strip():
        return "No file selected", gr.update(choices=["Sequence_1"], value="Sequence_1", visible=False), {}, "Sequence_1", ""
   
    try:
        sequences = {}
        current_header = None
        current_sequence = ""
        sequence_counter = 1
       
        for line in fasta_content.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                if current_header is not None and current_sequence:
                    sequences[current_header] = current_sequence
                
                current_header = line[1:].strip()
                current_sequence = ""
            else:
                sequence_data = ''.join(c.upper() for c in line if c.isalpha())
                
                if current_header is None:
                    current_header = f"Sequence_{sequence_counter}"
                    sequence_counter += 1
                
                current_sequence += sequence_data

        if current_header is not None and current_sequence:
            sequences[current_header] = current_sequence
       
        if not sequences:
            return "No valid protein sequences found in FASTA content", gr.update(choices=["Sequence_1"], value="Sequence_1", visible=False), {}, "Sequence_1", ""
        
        fasta_lines = []
        for header, sequence in sequences.items():
            fasta_lines.append(f">{header}")
            fasta_lines.append(sequence)
        modify_fasta_content = "\n".join(fasta_lines)
       
        sequence_choices = list(sequences.keys())
        default_sequence = sequence_choices[0]
        display_sequence = sequences[default_sequence]
        selector_visible = len(sequence_choices) > 1
        
        timestamp = str(int(time.time()))
        sequence_dir = get_save_path("Upload_dataset")
        temp_fasta_path = os.path.join(sequence_dir, f"paste_content_seq_{sanitize_filename(default_sequence)}_{timestamp}.fasta")
        save_selected_sequence_fasta(modify_fasta_content, default_sequence, temp_fasta_path)
        return display_sequence, gr.update(choices=sequence_choices, value=default_sequence, visible=selector_visible), sequences, default_sequence, temp_fasta_path, modify_fasta_content
       
    except Exception as e:
        print(f"Error in parse_fasta_paste_content: {str(e)}")
        return f"Error parsing FASTA content: {str(e)}", gr.update(choices=["Sequence_1"], value="Sequence_1", visible=False), {}, "Sequence_1", "", ""

def get_total_residues_count(df: pd.DataFrame) -> int:
    """Get total number of unique residue positions from mutation data."""
    if 'mutant' not in df.columns:
        return 0
    
    try:
        positions = df['mutant'].str.extract(r'(\d+)').dropna()
        return positions[0].astype(int).nunique() if not positions.empty else 0
    except Exception:
        return 0

def prepare_top_residue_heatmap_data(df: pd.DataFrame) -> Tuple:
    """Prepare data for heatmap visualization."""
    score_col = next((col for col in df.columns if 'score' in col.lower()), None)
    rank_col = "Prediction Rank"
    if score_col is None:
        return (None,) * 5

    valid_df = df[
        df['mutant'].apply(
            lambda m: isinstance(m, str) and 
            re.match(r'^[A-Z]\d+[A-Z]$', m) and 
            m[0] != m[-1]
        )
    ].copy()
    
    if valid_df.empty:
        return ([], [], np.array([[]]), np.array([[]]), score_col)

    min_score, max_score = valid_df[score_col].min(), valid_df[score_col].max()
    if max_score == min_score:
        valid_df['scaled_score'] = 0.0
    else:
        valid_df['scaled_score'] = -1 + 2 * (valid_df[score_col] - min_score) / (max_score - min_score)
    
    valid_df['position'] = valid_df['mutant'].str[1:-1].astype(int)

    effect_scores = valid_df.groupby('position')['scaled_score'].mean()
    sorted_positions = effect_scores.sort_values(ascending=False)
    top_positions = sorted_positions.head(20).index if len(sorted_positions) > 20 else sorted_positions.index
    top_df = valid_df[valid_df['position'].isin(top_positions)]
    x_labels = list("ACDEFGHIKLMNPQRSTVWY")
    x_map = {label: i for i, label in enumerate(x_labels)}
    wt_map = {pos: mut[0] for pos, mut in zip(top_df['position'], top_df['mutant'])}
    y_labels = [f"{wt_map.get(pos, '?')}{pos}" for pos in top_positions]
    y_map = {pos: i for i, pos in enumerate(top_positions)}
    
    z_data = np.full((len(y_labels), len(x_labels)), np.nan)
    score_matrix = np.full((len(y_labels), len(x_labels)), np.nan)
    rank_matrix = np.full((len(y_labels), len(x_labels)), np.nan)

    for _, row in top_df.iterrows():
        pos, mut_aa = row['position'], row['mutant'][-1]
        if pos in y_map and mut_aa in x_map:
            y_idx, x_idx = y_map[pos], x_map[mut_aa]
            z_data[y_idx, x_idx] = row['scaled_score']
            score_matrix[y_idx, x_idx] = round(row[score_col], 3)
            rank_matrix[y_idx, x_idx] = row[rank_col]
            
    return x_labels, y_labels, z_data, score_matrix, rank_matrix


def generate_plotly_heatmap(x_labels: List, y_labels: List, z_data: np.ndarray, score_data: np.ndarray) -> go.Figure:
    """Generate Plotly heatmap visualization."""
    if z_data is None or z_data.size == 0:
        return go.Figure().update_layout(title="Not enough data for heatmap")

    num_residues = len(y_labels)
    fig = go.Figure(data=go.Heatmap(
        z=z_data, 
        x=x_labels, 
        y=y_labels, 
        customdata=score_data,
        hovertemplate=(
            "<b>Position</b>: %{y}<br>"
            "<b>Mutation</b>: %{x}<br>"
            "<b>Effect</b>: %{z:.2f}"
            "<extra></extra>"
        ),
        colorscale='Blues', 
        zmin=-1, 
        zmax=1, 
        showscale=True,
        colorbar={
            'title': 'Normalized Effect', 
            'tickvals': [-1, 0, 1], 
            'ticktext': ['Low', 'Neutral', 'High']
        }
    ))
    
    fig.update_layout( 
        xaxis_title='Mutant Amino Acid', 
        yaxis_title='Residue Position (by impact)',
        height=max(400, 30 * num_residues + 150), 
        yaxis_autorange='reversed'
    )
    
    return fig

def process_fasta_file(file_path: str) -> str:
    sequences = []
    current_seq = ""
    current_header = ""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header and current_seq:
                    sequences.append((current_header, current_seq))
                current_header = line
                current_seq = ""
            else:
                current_seq += line

        if current_header and current_seq:
            sequences.append((current_header, current_seq))
    
    if len(sequences) <= 1:
        return file_path
    

    original_path = Path(file_path)
    timestamp = str(int(time.time()))
    fasra_dir = get_save_path("Upload_data")

    new_file_path = fasra_dir / f"filtered_{original_path.name}_{timestamp}"
    
    with open(new_file_path, 'w', encoding='utf-8') as f:
        f.write(f"{sequences[0][0]}\n")
        seq = sequences[0][1]
        f.write(f"{seq}\n")
    
    return str(new_file_path)


def generate_plots_for_all_results(results_df: pd.DataFrame) -> go.Figure:
    """Generate plots for function prediction results with consistent Dardana font and academic styling."""
    # Filter data
    plot_df = results_df[
        (results_df['header'] != "ERROR") & 
        (results_df['Dataset'].apply(
            lambda d: DATASET_TO_TASK_MAP.get(d) not in REGRESSION_TASKS_FUNCTION
        ))
    ].copy()

    if plot_df.empty:
        return go.Figure().update_layout(
            title_text="<b>No Visualization Available</b>", 
            title_x=0.5,
            font_family="Dardana",
            title_font_size=14,
            xaxis={"visible": False}, 
            yaxis={"visible": False},
            annotations=[{
                "text": "This task does not support visualization.", 
                "xref": "paper", "yref": "paper", 
                "x": 0.5, "y": 0.5, 
                "showarrow": False,
                "font": {"family": "Dardana", "size": 12}
            }]
        )

    sequences = plot_df['header'].unique()
    datasets = plot_df['Dataset'].unique()
    n_seq, n_data = len(sequences), len(datasets)
    
    titles = [
        f"{seq[:15]}<br>{ds[:20]}" if n_seq > 1 else f"{ds[:25]}" 
        for seq in sequences for ds in datasets
    ]
    
    fig = make_subplots(
        rows=n_seq, cols=n_data, 
        subplot_titles=titles, 
        vertical_spacing=0.25 if n_seq > 1 else 0.15
    )

    FONT_STYLE = dict(family="Dardana", size=12)
    AXIS_TITLE_STYLE = dict(family="Dardana", size=11)
    TICK_FONT_STYLE = dict(family="Dardana", size=10)
    BAR_WIDTH = 0.7 

    for r_idx, seq in enumerate(sequences, 1):
        for c_idx, ds in enumerate(datasets, 1):
            row_data = plot_df[(plot_df['header'] == seq) & (plot_df['Dataset'] == ds)]
            if row_data.empty:
                continue
            
            row = row_data.iloc[0]
            prob_col = next((col for col in row.index if 'probab' in col.lower()), None)
            
            if not prob_col or pd.isna(row[prob_col]):
                continue

            try:
                confidences = (json.loads(row[prob_col]) if isinstance(row[prob_col], str) 
                             else row[prob_col])
                
                if not isinstance(confidences, list):
                    continue
                
                task = DATASET_TO_TASK_MAP.get(ds)
                labels_key = ("DeepLocMulti" if ds == "DeepLocMulti" 
                            else "DeepLocBinary" if ds == "DeepLocBinary" 
                            else task)
                
                labels = LABEL_MAPPING_FUNCTION.get(
                    labels_key, 
                    [f"Class {k}" for k in range(len(confidences))]
                )
                
                colors = [
                    constant["COLOR_MAP_FUNCTION"].get(lbl, constant["COLOR_MAP_FUNCTION"]["Default"]) 
                    for lbl in labels
                ]
                
                plot_data = sorted(
                    zip(labels, confidences, colors), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                sorted_labels, sorted_conf, sorted_colors = zip(*plot_data)
                
                fig.add_trace(
                    go.Bar(
                        x=sorted_labels, 
                        y=sorted_conf, 
                        marker_color=sorted_colors,
                        width=BAR_WIDTH,
                        marker_line_width=0,
                        hovertemplate="<b>%{x}</b><br>Confidence: %{y:.2f}<extra></extra>"
                    ), 
                    row=r_idx, col=c_idx
                )
                
                # Axis styling
                fig.update_xaxes(
                    tickfont=TICK_FONT_STYLE,
                    row=r_idx, col=c_idx
                )
                fig.update_yaxes(
                    range=[0, 1], 
                    title_text="Confidence" if c_idx == 1 else "",
                    title_font=AXIS_TITLE_STYLE,
                    tickfont=TICK_FONT_STYLE,
                    row=r_idx, col=c_idx
                )
                
            except Exception as e:
                print(f"Plotting error for {seq}/{ds}: {e}")

    # Global layout adjustments
    main_title = "<b>Prediction Confidence Scores</b>"
    if n_seq == 1:
        main_title += f"<br><sub>{sequences[0][:80]}</sub>"
    
    fig.update_layout(
        title=dict(
            text=main_title, 
            x=0.5,
            font=dict(family="Dardana", size=16)
        ), 
        showlegend=False,
        font=FONT_STYLE,
        height=max(400, 300 * n_seq + 100),
        margin=dict(l=50, r=50, b=80, t=100),
        plot_bgcolor='white',
        paper_bgcolor='white',
        bargap=0.2 
    )
    
    # Subplot title styling
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(family="Dardana", size=12)
    
    return fig

def generate_plots_for_residue_results(results_df: pd.DataFrame, task: str = "Functional Prediction") -> go.Figure:
    """Generate plots for residue prediction results with consistent styling."""
    if results_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data to visualize",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, family="Arial")
        )
        return fig
    
    residues = results_df['residue'].tolist()
    probabilities = results_df['probability'].tolist()
    positions = results_df['index'].tolist() if 'index' in results_df.columns else list(range(1, len(residues) + 1))
    predicted_labels = results_df['predicted_label'].tolist() if 'predicted_label' in results_df.columns else [1 if p >= 0.5 else 0 for p in probabilities]
    functional_residues = [i for i, label in enumerate(predicted_labels) if label == 1]

    fig = go.Figure()

    sequence_length = len(residues)
    y_position = 0

    fig.add_trace(go.Scatter(
        x=[1, sequence_length],
        y=[y_position, y_position],
        mode='lines',
        line=dict(color='lightgray', width=12),
        showlegend=False,
        hoverinfo='skip'
    ))

    if functional_residues:
        segments = []
        current_segment = [functional_residues[0]]

        for i in range(1, len(functional_residues)):
            if functional_residues[i] == functional_residues[i-1] + 1:
                current_segment.append(functional_residues[i])
            else:
                segments.append(current_segment)
                current_segment = [functional_residues[i]]
        segments.append(current_segment)

        for i, segment in enumerate(segments):
            start_pos = positions[segment[0]]
            end_pos = positions[segment[-1]]

            segment_residues = [residues[j] for j in segment]
            segment_probs = [probabilities[j] for j in segment]
            hover_text = f"Task: {task}<br>Range: {start_pos}-{end_pos}<br>Residues: {''.join(segment_residues)}<br>Avg Probability: {np.mean(segment_probs):.3f}"
            
            fig.add_trace(go.Scatter(
                x=[start_pos, end_pos],
                y=[y_position, y_position],
                mode='lines',
                line=dict(color='red', width=12),
                showlegend=False,
                hovertemplate=hover_text + "<extra></extra>",
                name=f"Functional Region {i+1}"
            ))

    interval = max(1, sequence_length // 10)
    position_labels = list(range(1, sequence_length + 1, interval))
    if position_labels[-1] != sequence_length:
        position_labels.append(sequence_length)
    
    fig.update_layout(
        title=dict(
            text=f"<b>Functional Residue Prediction</b> - {task}",
            x=0.02, y=0.95, xanchor='left', yanchor='top',
            font=dict(size=14, family="Arial", color="black")
        ),
        xaxis=dict(
            title="Residue Position", 
            tickmode='array', 
            tickvals=position_labels,
            ticktext=[str(pos) for pos in position_labels],
            tickfont=dict(size=10), 
            showgrid=False,
            zeroline=False, 
            range=[0, sequence_length + 1]
        ),
        yaxis=dict(
            title="", 
            showticklabels=False, 
            showgrid=False,
            zeroline=False, 
            range=[-0.5, 0.5]
        ),
        showlegend=False, 
        height=150, 
        margin=dict(l=50, r=50, t=50, b=40),
        plot_bgcolor='white', 
        paper_bgcolor='white', 
        font=dict(family="Arial", size=10), 
        hovermode='closest'
    )
    return fig

def expand_residue_predictions(df):
    expanded_rows = []
    
    for _, row in df.iterrows():
        header = row['header']
        sequence = row['sequence']
        task = row['Task']
        dataset = row['Dataset']
        try:
            predictions = json.loads(row['predicted_class'])
            probabilities = json.loads(row['probabilities']) if isinstance(row['probabilities'], str) else row['probabilities']
            
            
            if isinstance(predictions[0], list):
                predictions = predictions[0]
            if isinstance(probabilities[0], list):
                probabilities = probabilities[0]
            for i, (residue, pred, prob) in enumerate(zip(sequence, predictions, probabilities)):
                if isinstance(prob, list):
                    max_prob = max(prob)
                    predicted_label = prob.index(max_prob)
                else:
                    max_prob = prob
                    predicted_label = pred
                
                expanded_rows.append({
                    'index': i,
                    'residue': residue,
                    'predicted_label': predicted_label,
                    'probability': max_prob,
                })
                
        except Exception as e:
            print(f"Error processing row for {header}: {e}")
            continue
    
    return pd.DataFrame(expanded_rows)

def save_selected_sequence_fasta(original_fasta_content, selected_sequence, output_path):
    sequences = {}
    current_header = None
    current_sequence = ""
    sequence_counter = 1
   
    for line in original_fasta_content.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('>'):
            if current_header is not None and current_sequence:
                sequences[current_header] = current_sequence
            
            current_header = line[1:].strip()
            current_sequence = ""
        else:
            sequence_data = ''.join(c.upper() for c in line if c.isalpha())
            
            if current_header is None:
                current_header = f"Sequence_{sequence_counter}"
                sequence_counter += 1
            
            current_sequence += sequence_data

    if current_header is not None and current_sequence:
        sequences[current_header] = current_sequence
   
    if not sequences or selected_sequence not in sequences:
        print(f"Error: Sequence '{selected_sequence}' not found in parsed sequences")
        return

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f">{selected_sequence}\n")
            f.write(sequences[selected_sequence])
    except Exception as e:
        print(f"Error saving file: {str(e)}")

def handle_paste_sequence_selection(selected_sequence, sequences_dict, original_fasta_content_from_state):
    if not sequences_dict or selected_sequence not in sequences_dict:
        return "No file selected", ""
    
    # Check if the content is valid
    if not original_fasta_content_from_state or original_fasta_content_from_state == "No file selected":
        return "No file selected", ""
    
    try:
        timestamp = str(int(time.time()))
        sequence_dir = get_save_path("Upload_dataset")
        temp_pdb_path = os.path.join(sequence_dir, f"paste_content_seq_{selected_sequence}_{timestamp}.fasta")
        save_selected_sequence_fasta(original_fasta_content_from_state, selected_sequence, temp_pdb_path)
        
        return sequences_dict[selected_sequence], temp_pdb_path
        
    except Exception as e:
        return f"Error processing chain selection: {str(e)}", ""

def handle_fasta_sequence_change(selected_sequence, sequences_dict, original_fasta_path):
    if not sequences_dict or selected_sequence not in sequences_dict:
        return "No file selected", ""
    
    # Check if the file path is valid and exists
    if not original_fasta_path or original_fasta_path == "No file selected" or not os.path.exists(original_fasta_path):
        return "No file selected", ""
    
    try:
        with open(original_fasta_path, 'r') as f:
            lines = f.read()

        new_fasta_lines = []
        new_fasta_lines.append(">"+selected_sequence)
        new_fasta_lines.append(sequences_dict[selected_sequence])
        
        dir_path = os.path.dirname(original_fasta_path)
        base_name, extension = os.path.splitext(os.path.basename(original_fasta_path))
        new_filename = f"{base_name}_1{extension}"
        new_fasta_path = os.path.join(dir_path, new_filename)
        with open(new_fasta_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_fasta_lines))
        
        return sequences_dict[selected_sequence], new_fasta_path

    except Exception as e:
        return f"Error processing sequence selection: {str(e)}", ""

def parse_pdb_paste_content(pdb_content):
    if not pdb_content.strip():
        return "No file selected", gr.update(choices=["A"], value="A", visible=False), {}, "A", ""
    
    try:
        chains = {}
        current_chain = None
        sequence = ""
        
        for line in pdb_content.strip().split('\n'):
            if line.startswith('ATOM'):
                chain_id = line[21:22].strip()
                if chain_id == "":
                    chain_id = "A"
                
                if current_chain != chain_id:
                    if current_chain is not None and sequence:
                        chains[current_chain] = sequence
                    current_chain = chain_id
                    sequence = ""
                
                res_name = line[17:20].strip()
                if res_name in ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']:
                    aa_map = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 
                             'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 
                             'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 
                             'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
                    
                    res_num = int(line[22:26].strip())
                    if len(sequence) < res_num:
                        sequence += aa_map[res_name]
        
        if current_chain is not None and sequence:
            chains[current_chain] = sequence
        
        if not chains:
            return "No valid protein chains found in PDB content", gr.update(choices=["A"], value="A", visible=False), {}, "A", ""
        
        chain_choices = list(chains.keys())
        default_chain = chain_choices[0]
        display_sequence = chains[default_chain]
        selector_visible = len(chain_choices) > 1
        timestamp = str(int(time.time()))
        sequence_dir = get_save_path("Upload_dataset")
        temp_pdb_path = os.path.join(sequence_dir, f"paste_content_chain_{default_chain}_{timestamp}.pdb")
        save_selected_chain_pdb(pdb_content, default_chain, temp_pdb_path)
        return display_sequence, gr.update(choices=chain_choices, value=default_chain, visible=selector_visible), chains, default_chain, temp_pdb_path
        
    except Exception as e:
        return f"Error parsing PDB content: {str(e)}", gr.update(choices=["A"], value="A", visible=False), {}, "A", ""

def save_selected_chain_pdb(original_pdb_content, selected_chain, output_path):
    new_pdb_lines = []
    atom_counter = 1
    
    for line in original_pdb_content.strip().split('\n'):
        if line.startswith('ATOM'):
            chain_id = line[21:22].strip()
            if chain_id == "":
                chain_id = "A"
            
            if chain_id == selected_chain:
                new_line = line[:21] + 'A' + line[22:]
                new_line = f"ATOM  {atom_counter:5d}" + new_line[11:]
                new_pdb_lines.append(new_line)
                atom_counter += 1
        elif not line.startswith('ATOM'):
            new_pdb_lines.append(line)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(new_pdb_lines))

def handle_paste_chain_selection(selected_chain, chains_dict, original_pdb_content_from_state):
    if not chains_dict or selected_chain not in chains_dict:
        return "No file selected", ""
    
    # Check if the content is valid
    if not original_pdb_content_from_state or original_pdb_content_from_state == "No file selected":
        return "No file selected", ""
    
    try:
        timestamp = str(int(time.time()))
        sequence_dir = get_save_path("Upload_dataset")
        temp_pdb_path = os.path.join(sequence_dir, f"paste_content_chain_{selected_chain}_{timestamp}.pdb")
        save_selected_chain_pdb(original_pdb_content_from_state, selected_chain, temp_pdb_path)
        
        return chains_dict[selected_chain], temp_pdb_path
        
    except Exception as e:
        return f"Error processing chain selection: {str(e)}", ""

def handle_pdb_chain_change(selected_chain, chains_dict, original_file_path):
    if not chains_dict or selected_chain not in chains_dict:
        return "No file selected", ""
    
    # Check if the file path is valid and exists
    if not original_file_path or original_file_path == "No file selected" or not os.path.exists(original_file_path):
        return "No file selected", ""
        
    try:
        with open(original_file_path, 'r') as f:
            pdb_content = f.read()
        
        new_pdb_lines = []
        atom_counter = 1
        
        for line in pdb_content.strip().split('\n'):
            if line.startswith('ATOM'):
                chain_id = line[21:22].strip()
                if chain_id == "":
                    chain_id = "A"
                
                if chain_id == selected_chain:
                    new_line = line[:21] + 'A' + line[22:]
                    new_line = f"ATOM  {atom_counter:5d}" + new_line[11:]
                    new_pdb_lines.append(new_line)
                    atom_counter += 1
            elif not line.startswith('ATOM'):
                new_pdb_lines.append(line)
        
        dir_path = os.path.dirname(original_file_path)
        base_name, extension = os.path.splitext(os.path.basename(original_file_path))
        new_filename = f"{base_name}_A{extension}"
        new_pdb_path = os.path.join(dir_path, new_filename)
        
        with open(new_pdb_path, 'w') as f:
            f.write('\n'.join(new_pdb_lines))
        
        return chains_dict[selected_chain], gr.update(value=new_pdb_path)
        
    except Exception as e:
        return f"Error processing chain selection: {str(e)}", ""

def process_pdb_file_upload(file_path):
    if not file_path:
        return "No file selected", gr.update(choices=["A"], value="A", visible=False), {}, "A", "", ""
    try:
        with open(file_path, 'r') as f:
            pdb_content = f.read()
        sequence, chain_update, chains_dict, default_chain, _ = parse_pdb_paste_content(pdb_content)
        return sequence, chain_update, chains_dict, default_chain, file_path, file_path
    except Exception as e:
        return f"Error reading PDB file: {str(e)}", gr.update(choices=["A"], value="A", visible=False), {}, "A", "", ""

def process_fasta_file_upload(file_path):
    if not file_path:
        return "No file selected", gr.update(choices=["Sequence_1"], value="Sequence_1", visible=False), {}, "Sequence_1", "", ""
    try:
        with open(file_path, 'r') as f:
            fasta_content = f.read()
        sequence, selector_update, sequences_dict, default_sequence, file_path, modify_fasta_content = parse_fasta_paste_content(fasta_content)
        return sequence, selector_update, sequences_dict, default_sequence, file_path, file_path
    except Exception as e:
        return f"Error reading FASTA file: {str(e)}", gr.update(choices=["Sequence_1"], value="Sequence_1", visible=False), {}, "Sequence_1", "", ""

def handle_file_upload(file_obj: Any) -> str:
    if not file_obj:
        return "No file selected", gr.update(choices=["Sequence_1"], value="Sequence_1", visible=False), {}, "Sequence_1", "", ""
    if isinstance(file_obj, str):
        file_path = file_obj
    else:
        file_path = file_obj.name
    if file_path.lower().endswith((".fasta", ".fa")):
        return process_fasta_file_upload(file_path)
    elif file_path.lower().endswith(".pdb"):
        return process_pdb_file_upload(file_path)
    else:
        return "No file selected", gr.update(choices=["Sequence_1"], value="Sequence_1", visible=False), {}, "Sequence_1", "", ""

def sanitize_filename(name: str) -> str:
    """Sanitize filename for safe file operations."""
    name = re.split(r'[|\s/]', name)[0]
    return re.sub(r'[^\w\-. ]', '_', name)


def toggle_ai_section(is_checked: bool):
    """Toggle visibility of AI configuration section."""
    return gr.update(visible=is_checked)

def create_zip_archive(files_to_zip: Dict[str, str], zip_filename: str) -> str:
    """Create ZIP archive with specified files."""
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files_to_zip.items():
            if os.path.exists(src):
                zf.write(src, arcname=arc)
    return zip_filename


