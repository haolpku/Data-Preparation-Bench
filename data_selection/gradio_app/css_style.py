CSS = """
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.gradio-container {
    max-width: 1400px !important;
    background-color: #f8f9fa;
}

.contain {
    border-radius: 12px !important;
    border: 1px solid #e0e0e0 !important;
    background: white !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
}

h1 {
    font-weight: 800 !important;
    background: linear-gradient(90deg, #2D3436 0%, #0984E3 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px !important;
}

button.primary {
    background: linear-gradient(45deg, #0984E3, #00CEC9) !important;
    border: none !important;
    transition: all 0.3s ease !important;
    font-weight: bold !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(9, 132, 227, 0.4) !important;
}

.output-markdown, .gr-code {
    background-color: #2d3436 !important;
    color: #dfe6e9 !important;
    border-radius: 8px !important;
    border: 1px solid #636e72 !important;
}

.tabs button.selected {
    border-bottom: 3px solid #0984E3 !important;
    color: #0984E3 !important;
    font-weight: bold !important;
}
"""