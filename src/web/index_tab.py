import gradio as gr

def create_index_tab(constant):
    try:
        with open("img/Show1.gjf", "r") as f:
            gjf_content = f.read()
    except Exception:
        gjf_content = "(Show1.gjf not found)"
    with gr.Blocks() as index_tab:
        gr.HTML(
            f'''

                <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
                body, .gradio-container {{
                    font-family: 'Inter', 'Roboto', Arial, sans-serif !important;
                    background: #f4f7fa;
                    color: #222;
                }}
                .main-content {{
                    margin-top: 80px;
                    max-width: 1400px;
                    margin-left: auto;
                    margin-right: auto;
                    background: #fff;
                    border-radius: 16px;
                    box-shadow: 0 4px 24px rgba(30,41,59,0.07);
                    padding: 40px 36px 32px 36px;
                }}
                h1, h2, h3 {{
                    color: #2563eb;
                    font-weight: 700;
                    margin-bottom: 0.5em;
                }}
                p, li, ul {{
                    font-size: 1.08em;
                    line-height: 1.7;
                }}
                .card {{
                    background: #f4f7fa;
                    border-radius: 10px;
                    box-shadow: 0 2px 8px rgba(30,41,59,0.04);
                    padding: 18px 22px;
                    margin-bottom: 18px;
                }}
                .stats-container {{
                    background: linear-gradient(135deg, #f6fbff 0%, #e3f3fb 100%);
                    border-radius: 16px;
                    padding: 32px;
                    margin-top: 40px;
                    color: #1e293b;
                    box-shadow: 0 8px 32px rgba(37, 99, 235, 0.07);
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 32px;
                    margin-top: 24px;
                    max-width: 1200px;
                    margin-left: auto;
                    margin-right: auto;
                }}
                .stat-item {{
                    background: rgba(255, 255, 255, 0.8);
                    border-radius: 12px;
                    padding: 16px 10px;
                    text-align: center;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(37, 99, 235, 0.15);
                    transition: all 0.3s ease;
                    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.08);
                }}
                .stat-item:hover {{
                    transform: translateY(-4px);
                    box-shadow: 0 12px 40px rgba(37, 99, 235, 0.15);
                    background: rgba(255, 255, 255, 0.9);
                }}
                .stat-number {{
                    font-size: 1.5em;
                    font-weight: 900;
                    margin-bottom: 4px;
                    text-shadow: 0 1px 2px rgba(37, 99, 235, 0.12);
                }}
                .stat-label {{
                    font-size: 0.95em;
                    opacity: 0.9;
                    font-weight: 500;
                }}
                .stat-icon {{
                    font-size: 1.3em;
                    margin-bottom: 6px;
                    display: block;
                }}
                

                </style>
                <div class="main-content">

            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5em;">
                <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venusfactory_logo.png" alt="Venus Head" style="height: 150px; margin-left: 10px;" />
                <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png" alt="Venus Logo" style="height: 100px; margin-right: 10px; margin-top: 20px;" />
            </div>
            <div style="text-align: center; margin-top: -80px; margin-bottom: 40px;">
                <h1 style="font-size:3.5em; font-weight:900;">Welcome to <span style='font-weight:900;'>VenusFactory</span> !</h1>
            </div>
            <div style="max-width: 1400px; margin: 0 auto; font-size: 1.2em; text-align: left;">
                <p style="font-size:1.2em; margin-bottom: 0.7em;"><b>VenusFactory</b> is a unified open-source platform for protein engineering, designed to simplify data acquisition, model fine-tuning, and functional analysis for both biologists and AI researchers.<br>
                The Web UI features four core modules:</p>
                <ul style="font-size:1.0em;">
                    <li>🤖 <b>VenusAgent-0.1</b> is an integrated AI assistant that answers questions related to the platform and protein AI.</li>
                    <li>🛠️ <b>Quick Tools</b> offers one-click protein analysis tools designed as a convenient method, making common tasks easy and accessible.</li>
                    <li>⚡ <b>Advanced Tools</b> enables zero-shot prediction, function analysis, and advanced data options for experienced users.</li>
                    <li>💾 <b>Download</b> allows you to get various protein data like AlphaFold2 Structures, RCSB PDB and InterPro.</li>
                </ul>
            </div>
            <hr style="margin: 40px 0; border: 1px solid #eee;">

            <div style="text-align: left; max-width: 1400px; margin: 0 auto;">
                <h1 style="font-size:2.2em; font-weight:900; color:#222; margin-bottom: 0.7em;">
                    How to Use VenusFactory
                </h1>
                <div style="font-size:1.2em;">
                    <p style="font-size:1.2em;">Depending on your needs, VenusFactory can provide different services.</p>
                    <p style="font-size:1.2em;">If you want a quick answer about protein mutations, use VenusAgent-0.1. Upload your file, and the AI Assistant will give you a helpful reply.</p>
                    <p style="font-size:1.2em;">If you want to know possible mutation methods or protein functions, go to Quick Tools, choose the task you need, and you will get the result in a few minutes.</p>
                    <p style="font-size:1.2em;">If you have some knowledge about different protein models, you can use the Advanced Tools tab. All major models are available to meet your needs.</p>
                    <p style="font-size:1.2em;">If you want to get some protein data files, click the download tab, input the PDB ID or UniProt ID, and download it for further research.</p>
                    
                    <p style="font-size:1.2em; font-weight:bold; margin-top:2em;">Module Demonstrations:</p>
                    
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 1em;">

                        <div style="text-align: center;">
                            <h3 style="color: #2563eb; margin-bottom: 10px;">🤖 VenusAgent-0.1</h3>
                            <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/agent.gif" alt="VenusAgent-0.1 Demo" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
                        </div>
                        

                        <div style="text-align: center;">
                            <h3 style="color: #2563eb; margin-bottom: 10px;">🛠️ Quick Tools</h3>
                            <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/quick_tool.gif" alt="Quick Tools Demo" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
                        </div>
                        

                        <div style="text-align: center;">
                            <h3 style="color: #2563eb; margin-bottom: 10px;">⚡ Advanced Tools</h3>
                            <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/advanced_tool.gif" alt="Advanced Tools Demo" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
                        </div>
                        

                        <div style="text-align: center;">
                            <h3 style="color: #2563eb; margin-bottom: 10px;">💾 Download</h3>
                            <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/download.gif" alt="Download Demo" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
                        </div>
                    </div>
                </div>
            </div>
            <hr style="margin: 40px 0; border: 1px solid #eee;">

            <div style="text-align: left; max-width: 1400px; margin: 0 auto; font-size: 1.2em;">
                <h1 style="font-size:2.2em; font-weight:900; color:#222; margin-bottom: 0.7em;">Citation</h1>
                <div style="font-size:1.2em; margin-bottom: 0.7em;">✏️ Please cite our work if you have used VenusFactory.</div>
                <pre style="background:#f8f8f8; border-radius:8px; padding:18px; font-size:1.2em; overflow-x:auto;"><code>@inproceedings{{tan-etal-2025-venusfactory,
            title = "{{V}}enus{{F}}actory: An Integrated System for Protein Engineering with Data Retrieval and Language Model Fine-Tuning",
            author = "Tan, Yang and Liu, Chen and Gao, Jingyuan and Wu, Banghao and Li, Mingchen and Wang, Ruilin and Zhang, Lingrong and Yu, Huiqun and Fan, Guisheng and Hong, Liang and Zhou, Bingxin",
            editor = "Mishra, Pushkar and Muresan, Smaranda and Yu, Tao",
            booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
            month = jul,
            year = "2025",
            address = "Vienna, Austria",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2025.acl-demo.23/",
            doi = "10.18653/v1/2025.acl-demo.23",
            pages = "230--241",
            ISBN = "979-8-89176-253-4",
        }}</code></pre>



                


            </div>
            <div class="stats-container">
                <h1 style="font-size:2.2em; font-weight:900; color:#1e293b; margin-bottom: 0.5em; text-align: center;">
                    📊 Platform Usage Statistics
                </h1>
                <p style="text-align: center; font-size: 1.2em; opacity: 0.8; margin-bottom: 0; color:#1e293b;">
                    Real-time platform usage statistics
                </p>
                
                <div class="stats-grid">

                    <div class="stat-item">
                        <span class="stat-icon">🌐</span>
                        <div class="stat-number" id="total-visits">0</div>
                        <div class="stat-label">Total Website Visits</div>
                    </div>
                    

                    <div class="stat-item">
                        <span class="stat-icon">🤖</span>
                        <div class="stat-number" id="agent-usage">0</div>
                        <div class="stat-label">VenusAgent Usage</div>
                    </div>
                    

                    <div class="stat-item">
                        <span class="stat-icon">🧬</span>
                        <div class="stat-number" id="evolution-usage">0</div>
                        <div class="stat-label">Mutation Prediction</div>
                    </div>
                    
                    <div class="stat-item">
                        <span class="stat-icon">⚡</span>
                        <div class="stat-number" id="prediction-usage">0</div>
                        <div class="stat-label">Function Prediction</div>
                    </div>
                </div>
                

                
                <script>
                    async function loadStats() {{
                        try {{
                            console.log('Loading statistics data...');
                            
                            const response = await fetch('/api/stats/get_stats');
                            const data = await response.json();
                            
                            if (data.error) {{
                                throw new Error(data.error);
                            }}
                            
                            console.log('Statistics data received:', data);
                            
                            const mutationTotal = (data.mutation_prediction_quick || 0) + (data.mutation_prediction_advanced || 0);
                            const functionTotal = (data.function_prediction_quick || 0) + (data.function_prediction_advanced || 0);
                            
                            const totalVisitsElement = document.getElementById('total-visits');
                            const agentUsageElement = document.getElementById('agent-usage');
                            const evolutionUsageElement = document.getElementById('evolution-usage');
                            const predictionUsageElement = document.getElementById('prediction-usage');
                            
                            if (totalVisitsElement) {{
                                totalVisitsElement.textContent = formatNumber(data.total_visits || 0);
                                console.log('Updated total-visits:', formatNumber(data.total_visits || 0));
                            }} else {{
                                console.log('Element total-visits not found');
                            }}
                            
                            if (agentUsageElement) {{
                                agentUsageElement.textContent = formatNumber(data.agent_usage || 0);
                                console.log('Updated agent-usage:', formatNumber(data.agent_usage || 0));
                            }} else {{
                                console.log('Element agent-usage not found');
                            }}
                            
                            if (evolutionUsageElement) {{
                                evolutionUsageElement.textContent = formatNumber(mutationTotal);
                                console.log('Updated evolution-usage:', formatNumber(mutationTotal));
                            }} else {{
                                console.log('Element evolution-usage not found');
                            }}
                            
                            if (predictionUsageElement) {{
                                predictionUsageElement.textContent = formatNumber(functionTotal);
                                console.log('Updated prediction-usage:', formatNumber(functionTotal));
                            }} else {{
                                console.log('Element prediction-usage not found');
                            }}
                            
                            console.log('All statistics updated to cards');
                            
                        }} catch (error) {{
                            console.log('Failed to load statistics:', error);
                            loadFromFile();
                        }}
                    }}
                    
                    function loadFromFile() {{
                        fetch('/file/stats_data.json')
                            .then(response => response.json())
                            .then(data => {{
                                const mutationTotal = (data.mutation_prediction_quick || 0) + (data.mutation_prediction_advanced || 0);
                                document.getElementById('evolution-usage').textContent = formatNumber(mutationTotal);
                                
                                const functionTotal = (data.function_prediction_quick || 0) + (data.function_prediction_advanced || 0);
                                document.getElementById('prediction-usage').textContent = formatNumber(functionTotal);
                                
                                document.getElementById('agent-usage').textContent = formatNumber(data.agent_usage || 0);
                                document.getElementById('total-visits').textContent = formatNumber(data.total_visits || 0);
                                
                                console.log('Statistics loaded from file successfully');
                            }})
                            .catch(error => {{
                                console.log('File loading also failed:', error);
                                document.getElementById('evolution-usage').textContent = '0';
                                document.getElementById('prediction-usage').textContent = '0';
                                document.getElementById('agent-usage').textContent = '0';
                                document.getElementById('total-visits').textContent = '0';
                            }});
                    }}
                    
                    function formatNumber(num) {{
                        return new Intl.NumberFormat().format(num);
                    }}
                    
                    function initializeStats() {{
                        console.log('Starting statistics initialization...');
                        
                        const totalVisitsElement = document.getElementById('total-visits');
                        const agentUsageElement = document.getElementById('agent-usage');
                        const evolutionUsageElement = document.getElementById('evolution-usage');
                        const predictionUsageElement = document.getElementById('prediction-usage');
                        
                        if (totalVisitsElement && agentUsageElement && evolutionUsageElement && predictionUsageElement) {{
                            console.log('All statistics elements found, starting data loading...');
                            
                            loadStats();
                            
                            setTimeout(() => {{
                                console.log('Checking statistics again in 3 seconds...');
                                loadStats();
                            }}, 3000);
                            
                            setInterval(loadStats, 30000);
                        }} else {{
                            console.log('Statistics elements not loaded yet, retrying in 100ms...');
                            setTimeout(initializeStats, 100);
                        }}
                    }}
                    
                    document.addEventListener('DOMContentLoaded', () => {{
                        console.log('Page DOM loaded');
                        initializeStats();
                        
                        setTimeout(() => {{
                            console.log('Loading statistics immediately after page load...');
                            loadStats();
                        }}, 100);
                    }});
                    
                    if (document.readyState === 'loading') {{
                        document.addEventListener('DOMContentLoaded', initializeStats);
                    }} else {{
                        initializeStats();
                        setTimeout(() => {{
                            console.log('Page already loaded, loading statistics immediately...');
                            loadStats();
                        }}, 100);
                    }}
                    
                    function trackUsage(module) {{
                        fetch('/api/stats/track_usage', {{
                            method: 'POST',
                            headers: {{'Content-Type': 'application/json'}},
                            body: JSON.stringify({{module: module}})
                        }})
                        .then(response => response.json())
                        .then(result => {{
                            if (result.status === 'success') {{
                                console.log('Usage statistics recorded successfully:', module);
                                setTimeout(loadStats, 1000);
                            }}
                        }})
                        .catch(error => {{
                            console.log('Failed to record usage statistics:', error);
                        }});
                    }}
                    

                </script>
            </div>
            '''
        )
    return {"index_tab": index_tab}
