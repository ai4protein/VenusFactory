import gradio as gr
from .stats_api import get_stats, track_visit, track_usage, reset_stats

def create_stats_routes():
    def stats_api_get():
        try:
            stats = get_stats()
            return gr.JSON(value=stats, visible=False)
        except Exception as e:
            return gr.JSON(value={"error": str(e)}, visible=False)
    
    def stats_api_track_visit():
        try:
            result = track_visit()
            return gr.Text(value="Visit tracked", visible=False)
        except Exception as e:
            return gr.Text(value=f"Error: {str(e)}", visible=False)
    
    def stats_api_track_usage():
        try:
            result = track_usage("page_visit")
            return gr.Text(value="Usage tracked", visible=False)
        except Exception as e:
            return gr.JSON(value=f"Error: {str(e)}", visible=False)
    
    def stats_api_reset():
        try:
            result = reset_stats()
            return gr.Text(value="Usage tracked", visible=False)
        except Exception as e:
            return gr.Text(value=f"Error: {str(e)}", visible=False)
    
    return {
        "get_stats": stats_api_get,
        "track_visit": stats_api_track_visit,
        "track_usage": stats_api_track_usage,
        "reset_stats": stats_api_reset
    }
