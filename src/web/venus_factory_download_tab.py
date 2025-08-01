import os
import subprocess
import tempfile
import json
import hashlib
import zipfile
import gradio as gr

from datetime import datetime
from gradio_molecule3d import Molecule3D
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union

# Molecule3D representations for different coloring schemes
RCSB_REPS =  [
    {
      "model": 0,
      "chain": "",
      "resname": "",
      "style": "cartoon",
      "color": "chain",
      "around": 0,
      "byres": False,
    }
  ]

AF2_REPS =  [
    {
      "model": 0,
      "chain": "",
      "resname": "",
      "style": "cartoon",
      "color": "alphafold",
      "around": 0,
      "byres": False,
    }
  ]


def generate_task_id(user_hash: str = None) -> str:
    if not user_hash:
        user_hash = "default_user"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{user_hash}/{timestamp}"

def create_task_folder(task_id: str, task_type: str) -> str:
    task_folder = f"temp_downloads/tasks/{task_id}/{task_type}"
    os.makedirs(task_folder, exist_ok=True)
    return task_folder

def zip_task_results(task_folder: str, task_id: str, task_type: str) -> str:
    zip_path = f"temp_downloads/zips/{task_id}_{task_type}.zip"
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(task_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, task_folder)
                zipf.write(file_path, arcname)
    
    return zip_path

def create_download_tool_tab(constant: Dict[str, Any]):
    def run_download_script(script_name: str, **kwargs) -> str:
        """Runs an external Python script using subprocess."""
        cmd = ["python", f"src/crawler/{script_name}"]
        for k, v in kwargs.items():
            if v is None: continue
            if isinstance(v, bool):
                if v: cmd.append(f"--{k}")
            elif isinstance(v, str) and v.startswith("--"):
                cmd.append(v)
            else:
                cmd.extend([f"--{k}", str(v)])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return f"Download completed successfully:\n{result.stdout}"
        except subprocess.CalledProcessError as e:
            return f"Error during download script execution:\n{e.stderr}"
        except FileNotFoundError:
            return f"Error: Script not found at 'src/crawler/{script_name}'. Please check the path."
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

    def parse_pdb_chains(pdb_file_path: str) -> List[str]:
        """Parses a PDB file to find all unique chain IDs."""
        chains = set()
        try:
            with open(pdb_file_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        chain_id = line[21:22].strip()
                        if chain_id:
                            chains.add(chain_id)
        except Exception as e:
            print(f"Error parsing PDB chains: {e}")
        return sorted(list(chains))

    def read_uploaded_file(file_path: str, max_lines: int = 20) -> Tuple[List[str], str]:
        """Read and preview uploaded file content."""
        if not file_path or not os.path.exists(file_path):
            return [], "No file uploaded"
        preview_text = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            preview_text += "\n".join(lines[:max_lines])
            if len(lines) > max_lines:
                preview_text += f"\n... and {len(lines) - max_lines} more entries"
            return lines, preview_text
        except Exception as e:
            return [], f"Error reading file: {str(e)}"

    def read_json_file(file_path: str, max_items: int = 10) -> Tuple[Any, str]:
        """Read and preview uploaded JSON file content."""
        if not file_path or not os.path.exists(file_path):
            return None, "No file uploaded"
        preview_text = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                for i, item in enumerate(data[:max_items]):
                    preview_text += f"{i+1}. {item}\n"
                if len(data) > max_items:
                    preview_text += f"... and {len(data) - max_items} more entries"
            elif isinstance(data, dict):
                keys = list(data.keys())[:max_items]
                for key in keys:
                    preview_text += f"- {key}: {str(data[key])[:50]}...\n"
                if len(data) > max_items:
                    preview_text += f"... and {len(data) - max_items} more keys"
            else:
                preview_text = f"JSON content: {str(data)[:200]}..."
            return data, preview_text
        except Exception as e:
            return None, f"Error reading JSON file: {str(e)}"

    def handle_interpro_download(method: str, id_val: str, json_file, user_hash: str, error: bool) -> tuple:
        """Handles InterPro metadata download."""
        task_id = generate_task_id(user_hash)
        task_folder = create_task_folder(task_id, "interpro_metadata")
        json_val = json_file.name if (method == "From JSON" and json_file) else None
        args = {
            "interpro_id": id_val if method == "Single ID" else None,
            "interpro_json": json_val,
            "out_dir": task_folder,
            "error_file": f"{task_folder}/failed.txt" if error else None
        }
        download_result = run_download_script("metadata/download_interpro.py", **args)
        zip_path = zip_task_results(task_folder, task_id, "interpro_metadata")
        return download_result, gr.update(visible=True, value=zip_path)

    def handle_rcsb_download(method: str, id_val: str, file_val, user_hash: str, error: bool) -> tuple:
        """Handles RCSB metadata download."""
        task_id = generate_task_id(user_hash)
        task_folder = create_task_folder(task_id, "rcsb_metadata")
        file_path = file_val.name if (method == "From File" and file_val) else None
        args = {
            "pdb_id": id_val if method == "Single ID" else None,
            "pdb_id_file": file_path,
            "out_dir": task_folder,
            "error_file": f"{task_folder}/failed.txt" if error else None
        }
        download_result = run_download_script("metadata/download_rcsb.py", **args)
        zip_path = zip_task_results(task_folder, task_id, "rcsb_metadata")
        return download_result, gr.update(visible=True, value=zip_path)

    def handle_uniprot_download(method: str, id_val: str, file_val, user_hash: str, merge: bool, error: bool) -> tuple:
        """Handles UniProt sequence download."""
        task_id = generate_task_id(user_hash)
        task_folder = create_task_folder(task_id, "uniprot_sequence")
        file_path = file_val.name if (method == "From File" and file_val) else None
        args = {
            "uniprot_id": id_val if method == "Single ID" else None,
            "file": file_path,
            "out_dir": task_folder,
            "merge": "--merge" if merge else None,
            "error_file": f"{task_folder}/failed.txt" if error else None
        }
        download_result = run_download_script("sequence/download_uniprot_seq.py", **args)
        zip_path = zip_task_results(task_folder, task_id, "uniprot_sequence")
        return download_result, gr.update(visible=True, value=zip_path)

    def handle_struct_download(method: str, id_val: str, file_upload, user_hash: str, type_val: str, error: bool) -> tuple:
        """Handles RCSB structure download and updates the UI."""
        task_id = generate_task_id(user_hash)
        task_folder = create_task_folder(task_id, "rcsb_structure")
        file_path = file_upload.name if (method == "From File" and file_upload) else None
        download_output = run_download_script(
            "structure/download_rcsb.py",
            pdb_id=id_val if method == "Single ID" else None,
            pdb_id_file=file_path,
            out_dir=task_folder, 
            type=type_val,
            unzip="--unzip",
            error_file=f"{task_folder}/failed.txt" if error else None
        )
        viz_status = "Download finished."
        zip_path = zip_task_results(task_folder, task_id, "rcsb_structure")
        
        # Update viewer for single ID downloads
        viewer_update = gr.update()
        if "Download completed successfully" in download_output and method == "Single ID":
            # Find the downloaded structure file
            pdb_file = Path(task_folder) / f"{id_val}.{type_val}"
            if pdb_file.exists():
                viewer_update = gr.update(value=str(pdb_file), reps=RCSB_REPS)
                viz_status = f"✅ Viewing: Structure {id_val}.{type_val} from RCSB PDB"
            else:
                viz_status = f"✅ Downloaded {id_val} but file not found for viewing"
        elif "Download completed successfully" in download_output:
            viz_status = "Batch download complete. Select a single ID to view."
        else:
            viz_status = "Download failed."
        return download_output, viz_status, viewer_update, gr.update(visible=True, value=zip_path)

    def handle_af_download(method: str, id_val: str, file_upload, user_hash: str, error: bool) -> tuple:
        """Handles AlphaFold structure download and updates the UI."""
        task_id = generate_task_id(user_hash)
        task_folder = create_task_folder(task_id, "AlphaFold_structure")
        file_path = file_upload.name if (method == "From File" and file_upload) else None
        download_output = run_download_script(
            "structure/download_alphafold.py",
            uniprot_id=id_val if method == "Single ID" else None,
            uniprot_id_file=file_path,
            out_dir=task_folder, error_file=f"{task_folder}/failed.txt" if error else None
        )
        viz_status = "Download finished."
        zip_path = zip_task_results(task_folder, task_id, "AlphaFold_structure")
        
        # Update viewer for single ID downloads
        viewer_update = gr.update()
        if "Download completed successfully" in download_output and method == "Single ID":
            # Find the downloaded structure file
            pdb_file = Path(task_folder) / f"{id_val}.pdb"
            if pdb_file.exists():
                viewer_update = gr.update(value=str(pdb_file), reps=AF2_REPS)
                viz_status = f"✅ Viewing: Structure {id_val}.pdb from AlphaFold2"
            else:
                viz_status = f"✅ Downloaded {id_val} but file not found for viewing"
        elif "Download completed successfully" in download_output:
            viz_status = "Batch download complete. Select a single ID to view."
        else:
            viz_status = "Download failed."
        return download_output, viz_status, viewer_update, gr.update(visible=True, value=zip_path)

    # --- Gradio UI Layout ---
    with gr.Blocks() as download_tab_content:
        gr.Markdown("## Download Center")

        with gr.Tabs():
            # --- Sequence and Metadata Download Tab ---
            with gr.TabItem("🧬 Sequence & Metadata Download"):
                gr.Markdown("### Protein Sequence and Annotation Information Download")
                with gr.Tabs():
                    # RCSB Metadata Download
                    with gr.TabItem("RCSB Metadata"):
                        gr.Markdown("#### Download metadata for PDB entries from RCSB")
                        rcsb_method = gr.Radio(["Single ID", "From File"], label="Download Method", value="Single ID")
                        rcsb_id = gr.Textbox(label="PDB ID", value="1a0j", visible=True, placeholder="e.g., 1a0j")
                        with gr.Column(visible=False) as rcsb_file_column:
                            rcsb_file_upload = gr.File(label="Upload PDB ID List", file_types=[".txt"])
                            rcsb_file_example = gr.Examples(
                                examples=["./download/rcsb.txt"],
                                inputs=rcsb_file_upload,
                                label="Click example to load"
                            )
                            rcsb_preview = gr.Textbox(label="File Preview", interactive=False, lines=5)
                        rcsb_btn = gr.Button("Start Download", variant="primary")
                        rcsb_output = gr.Textbox(label="Download Status", interactive=False, lines=5)
                        rcsb_download_btn = gr.DownloadButton(label="Save Downloaded Data", visible=False)
                        rcsb_error = gr.Checkbox(label="Save error file", value=True)
                        rcsb_user = gr.Textbox(label="User Hash", value="default_user", visible=False)
                        # Event handlers for RCSB
                        rcsb_method.change(
                            lambda method: gr.update(visible=(method == "From File")),
                            inputs=rcsb_method,
                            outputs=rcsb_file_column
                        )
                        rcsb_file_upload.change(
                            lambda f: read_uploaded_file(f.name if f else None)[1] if f else "",
                            inputs=rcsb_file_upload,
                            outputs=rcsb_preview
                        )
                        rcsb_btn.click(
                            fn=handle_rcsb_download, 
                            inputs=[rcsb_method, rcsb_id, rcsb_file_upload, rcsb_user, rcsb_error], 
                            outputs=[rcsb_output, rcsb_download_btn]
                        )
                    # UniProt Sequences Download
                    with gr.TabItem("UniProt Sequences"):
                        gr.Markdown("#### Download protein sequences from UniProt in FASTA format")
                        with gr.Row():
                            uniprot_method = gr.Radio(["Single ID", "From File"], label="Download Method", value="Single ID")
                        uniprot_id = gr.Textbox(label="UniProt ID", value="P00734", placeholder="e.g., P00734")
                        with gr.Column(visible=False) as uniprot_file_column:
                            uniprot_file_upload = gr.File(label="Upload UniProt ID List", file_types=[".txt"])
                            uniprot_file_example = gr.Examples(
                                examples=["./download/uniprot.txt"],
                                inputs=uniprot_file_upload,
                                label="Click example to load"
                            )
                            uniprot_preview = gr.Textbox(label="File Preview", interactive=False, lines=5)
                        uniprot_btn = gr.Button("Start Download", variant="primary")
                        uniprot_output = gr.Textbox(label="Download Status", interactive=False, lines=5)
                        uniprot_download_btn = gr.DownloadButton(label="Save Downloaded Data", visible=False)
                        uniprot_merge = gr.Checkbox(label="Merge into single FASTA file", value=False)
                        uniprot_error = gr.Checkbox(label="Save error file", value=True)
                        uniprot_user = gr.Textbox(label="User Hash", value="default_user", visible=False)
                        # Event handlers for UniProt
                        uniprot_method.change(
                            lambda method: gr.update(visible=(method == "From File")),
                            inputs=uniprot_method,
                            outputs=uniprot_file_column
                        )
                        uniprot_file_upload.change(
                            lambda f: read_uploaded_file(f.name if f else None)[1] if f else "",
                            inputs=uniprot_file_upload,
                            outputs=uniprot_preview
                        )
                        uniprot_btn.click(
                            fn=handle_uniprot_download, 
                            inputs=[uniprot_method, uniprot_id, uniprot_file_upload, uniprot_user, uniprot_merge, uniprot_error], 
                            outputs=[uniprot_output, uniprot_download_btn]
                        )
                    
                    # InterPro Metadata Download
                    with gr.TabItem("InterPro Metadata"):
                        gr.Markdown("#### Download protein domain, family, and Gene Ontology annotation metadata from InterPro")
                        with gr.Row():
                            interpro_method = gr.Radio(["Single ID", "From JSON"], label="Download Method", value="Single ID")
                        interpro_id = gr.Textbox(label="InterPro ID", value="IPR000001", visible=True, placeholder="e.g., IPR000001")
                        interpro_json_upload = gr.File(label="Upload JSON File", file_types=[".json"], visible=False)
                        interpro_preview = gr.Textbox(label="File Preview", interactive=False, lines=5, visible=False)
                        interpro_btn = gr.Button("Start Download", variant="primary")
                        interpro_output = gr.Textbox(label="Download Status", interactive=False, lines=5)
                        interpro_download_btn = gr.DownloadButton(label="Save Downloaded Data", visible=False)
                        interpro_error = gr.Checkbox(label="Save error file", value=True)
                        interpro_user = gr.Textbox(label="User Hash", value="default_user", visible=False)
                        # Event handlers for InterPro
                        interpro_method.change(
                            lambda method: [
                                gr.update(visible=(method == "Single ID")),
                                gr.update(visible=(method == "From JSON")),
                                gr.update(visible=(method == "From JSON")),
                            ],
                            inputs=interpro_method,
                            outputs=[interpro_id, interpro_json_upload, interpro_preview]
                        )
                        interpro_json_upload.change(
                            lambda f: read_json_file(f.name if f else None)[1] if f else "",
                            inputs=interpro_json_upload,
                            outputs=interpro_preview
                        )
                        interpro_btn.click(
                            fn=handle_interpro_download, 
                            inputs=[interpro_method, interpro_id, interpro_json_upload, interpro_user, interpro_error], 
                            outputs=[interpro_output, interpro_download_btn]
                        )
            # --- Structure Download Tab ---
            with gr.TabItem("⚛️ Structure Download"):
                gr.Markdown("### Protein Structure Download and Visualization")
                
                # Main Tabs to switch between RCSB and AlphaFold
                with gr.Tabs():

                    # --- RCSB PDB Tab ---
                    with gr.TabItem("RCSB PDB"):
                        # A row to create the left-right layout
                        with gr.Row(equal_height=False):
                            
                            # Left Column: Controls
                            with gr.Column(scale=2):
                                gr.Markdown("#### Download from RCSB PDB")
                                struct_method = gr.Radio(["Single ID", "From File"], label="Method", value="Single ID")
                                struct_id = gr.Textbox(label="PDB ID", value="1a0j", placeholder="e.g., 1crn", visible=True)
                                struct_type = gr.Dropdown(["pdb", "cif"], value="pdb", label="File Type")
                                struct_unzip = gr.Checkbox(label="Unzip files", value=True)
                                struct_error = gr.Checkbox(label="Save error log", value=True)
                                with gr.Column(visible=False) as struct_file_column:
                                    struct_file_upload = gr.File(label="Upload PDB ID List", file_types=[".txt"])
                                    struct_file_example = gr.Examples(
                                        examples=["./download/rcsb.txt"],
                                        inputs=struct_file_upload,
                                        label="Click example to load"
                                    )
                                    struct_preview = gr.Textbox(label="File Preview", interactive=False, lines=5)

                                struct_btn = gr.Button("Start Download", variant="primary")
                                struct_user = gr.Textbox(label="User Hash", value="default_user", visible=False) # Hidden field
                                struct_output = gr.Textbox(label="Download Status", interactive=False, lines=5)
                            # Right Column: Viewer and Outputs
                            with gr.Column(scale=3):
                                struct_viz_status = gr.Textbox(label="Visualization Status", interactive=False, value="Enter a PDB ID and click 'Download'")
                                struct_molecule_viewer = Molecule3D(
                                    label="RCSB Structure Viewer",
                                    reps=RCSB_REPS,
                                    height=600
                                )
                                
                                struct_download_btn = gr.DownloadButton(label="Save Downloaded Data", visible=False)

                    # --- AlphaFold2 Tab ---
                    with gr.TabItem("AlphaFold2"):
                        # A row to create the left-right layout
                        with gr.Row(equal_height=False):

                            # Left Column: Controls
                            with gr.Column(scale=2):
                                gr.Markdown("#### Download from AlphaFold DB")
                                af_method = gr.Radio(["Single ID", "From File"], label="Method", value="Single ID")
                                af_id = gr.Textbox(label="UniProt ID", value="P00734", placeholder="e.g., P0DTD1", visible=True)
                                af_unzip = gr.Checkbox(label="Unzip files", value=True) # Note: AF files are not zipped, so this may not be needed
                                af_error = gr.Checkbox(label="Save error log", value=True)
                                with gr.Column(visible=False) as af_file_column:
                                    af_file_upload = gr.File(label="Upload PAlphaFold DB ID List", file_types=[".txt"])
                                    af_file_example = gr.Examples(
                                        examples=["./download/uniprot.txt"],
                                        inputs=af_file_upload,
                                        label="Click example to load"
                                    )
                                    af_preview = gr.Textbox(label="File Preview", interactive=False, lines=5)

                                af_btn = gr.Button("Start Download", variant="primary")
                                af_user = gr.Textbox(label="User Hash", value="default_user", visible=False) # Hidden field
                                af_output = gr.Textbox(label="Download Status", interactive=False, lines=5)
                            # Right Column: Viewer and Outputs
                            with gr.Column(scale=3):
                                af_viz_status = gr.Textbox(label="Visualization Status", interactive=False, value="Enter a UniProt ID and click 'Download'")
                                af_molecule_viewer = Molecule3D(
                                    label="AlphaFold Structure Viewer",
                                    reps=AF2_REPS,
                                    height=600
                                )
                                af_download_btn = gr.DownloadButton(label="Save Downloaded Data", visible=False)

    
                        # File preview handlers
                        struct_method.change(
                            lambda method: gr.update(visible=(method == "From File")),
                            inputs=struct_method, 
                            outputs=struct_file_column
                        )
                        af_method.change(
                            lambda method: gr.update(visible=(method == "From File")), 
                            inputs=af_method, 
                            outputs=af_file_column
                        )
                        struct_file_upload.change(
                            lambda f: read_uploaded_file(f.name if f else None)[1] if f else "",
                            inputs=struct_file_upload,
                            outputs=struct_preview
                        )
                        af_file_upload.change(
                            lambda f: read_uploaded_file(f.name if f else None)[1] if f else "",
                            inputs=af_file_upload,
                            outputs=af_preview
                        )
                        struct_btn.click(
                            handle_struct_download, 
                            inputs=[struct_method, struct_id, struct_file_upload, struct_user, struct_type, struct_error], 
                            outputs=[struct_output, struct_viz_status, struct_molecule_viewer, struct_download_btn]
                        )
                        af_btn.click(
                            handle_af_download, 
                            inputs=[af_method, af_id, af_file_upload, af_user, af_error], 
                            outputs=[af_output, af_viz_status, af_molecule_viewer, af_download_btn]
                        )

    return download_tab_content