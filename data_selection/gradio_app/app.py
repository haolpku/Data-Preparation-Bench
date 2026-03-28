import gradio as gr
import yaml
import threading
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import json
import re
import plotly.express as px
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from pipeline import ResearchPlatform, PROJECT_ROOT

from gradio_app.css_style import CSS

DATASET_MAP = {
    "None": "None",
    "OpenHermes-Sample": "/home/hxy/filter/dataset/intermediate/openhermes2_5_extracted_sample/openhermes2_5_extracted_sample.jsonl",
    "OpenHermes-Full": "/home/hxy/filter/dataset/intermediate/openhermes2_5/openhermes2_5.json"
}

class ExperimentVisualizer:
    @staticmethod
    def parse_train_loss(log_path: Path):
        losses = []
        steps = []
        if not log_path.exists():
            return None
        
        with open(log_path, "r") as f:
            content = f.read()
            # 匹配典型 Transformers/LLaMA-Factory 的日志格式
            # 例如: {'loss': 1.23, 'learning_rate': 5e-5, 'epoch': 0.1}
            matches = re.findall(r"'loss':\s*([\d\.]+).*?'epoch':\s*([\d\.]+)", content)
            for loss, epoch in matches:
                losses.append(float(loss))
                steps.append(float(epoch))
        
        if not losses: 
            return None
        
        df = pd.DataFrame({"Epoch": steps, "Loss": losses})
        fig = px.line(df, x="Epoch", y="Loss", title="Training Loss Curve")
        return fig

    @staticmethod
    def load_eval_results(exp_root: Path):
        eval_dir = exp_root / "eval_results"
        summary = {}
        if not eval_dir.exists():
            return pd.DataFrame()  
        
        for json_file in eval_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    task_name = json_file.stem
                    results = data.get("core_metrics", {})
                    for sub_task, metrics in results.items():
                        score = metrics
                        if score:
                            summary[f"{sub_task}"] = round(score, 4)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        return pd.DataFrame([summary]) if summary else pd.DataFrame()

    @staticmethod
    def preview_jsonl(file_path: Path, n=10):
        if not file_path.exists():
            return pd.DataFrame()
        
        data = []
        try:
            with open(file_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= n: 
                        break
                    data.append(json.loads(line))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return pd.DataFrame()
        
        return pd.DataFrame(data)

    @staticmethod
    def get_all_experiments(output_root: str):
        root = Path(output_root)
        exps = []
        for d in root.iterdir():
            if d.is_dir():
                status = "Completed" if (d / "eval_results").exists() else "Running/Failed"
                exps.append({
                    "Experiment": d.name,
                    "Created": d.stat().st_mtime,
                    "Status": status
                })
        df = pd.DataFrame(exps)
        if not df.empty:
            df = df.sort_values("Created", ascending=False)
        return df


class GradioApp:
    def __init__(self):
        self.output_dir = PROJECT_ROOT / "output"
        self.config_dir = PROJECT_ROOT / "configs"
        self.visualizer = ExperimentVisualizer()
        self.current_exp_id = None  

    def get_exp_list(self):
        if not self.output_dir.exists(): 
            return []
        dirs = [d.name for d in self.output_dir.iterdir() if d.is_dir()]
        return sorted(dirs, reverse=True)

    def get_model_list(self):
        model_base_path = Path("/home/hxy/huggingface_cache/hub") 
        models = []
        if model_base_path.exists():
            models = [str(p) for p in model_base_path.iterdir() if p.is_dir()]
        defaults = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
        return list(set(models + defaults))

    def get_baseline_choices(self):
        baseline_root = Path("./baseline")
        choices = []
        if not baseline_root.exists():
            return ["bench/dclm"] 
        
        categories = ["agent", "bench", "surrogate_based", "surrogate_free"]
        for cat in categories:
            cat_path = baseline_root / cat
            if cat_path.is_dir():
                subs = [p.name for p in cat_path.iterdir() if not p.name.startswith("__")]
                for sub in subs:
                    # 生成类似 "bench/dclm" 或 "agent/dataflow" 的格式
                    choices.append(f"{cat}/{sub}")
        return sorted(choices)

    def _get_flexible_config(self):
        template_path = self.config_dir / "default_template.yaml"
        if template_path.exists():
            with open(template_path, 'r') as f:
                return yaml.safe_load(f)
        
        return {
            "name": "exp_" + datetime.now().strftime("%m%d_%H%M"),
            "output_root": "./output",
            "filter": {"type": "bench", "method": "dclm", "train_file": "dataset/sample.jsonl"},
            "train": {"model_path": "Qwen/Qwen2.5-7B-Instruct", "config_profile": "configs/qwen2.5_instruct_lora_sft.yaml"},
            "eval": {"tasks": ["mmlu", "gsm8k"]}
        }

    def update_yaml_logic(self, baselines, server_ds_key, file_obj, model_path, metrics, current_yaml):
        try:
            cfg = yaml.safe_load(current_yaml)
            final_ds_path = DATASET_MAP.get(server_ds_key)
            if (not final_ds_path or final_ds_path == "None") and file_obj:
                final_ds_path = file_obj.name
            
            if final_ds_path:
                cfg['filter']['train_file'] = final_ds_path
                cat, method = baselines[0].split('/')
                cfg['filter']['type'] = cat
                cfg['filter']['method'] = method
            cfg['train']['model_path'] = model_path
            cfg['eval']['tasks'] = metrics
            return yaml.dump(cfg, sort_keys=False, allow_unicode=True)
        except Exception as e:
            return f"# Error: {str(e)}\n{current_yaml}"

    def refresh_all_experiments(self, exp_id=None):
        """刷新实验列表并选中指定的 exp_id"""
        new_choices = self.get_exp_list()
        
        # 如果没有传入 exp_id，优先使用当前选中的，否则用第一个
        if exp_id and exp_id in new_choices:
            target_val = exp_id
        elif self.current_exp_id and self.current_exp_id in new_choices:
            target_val = self.current_exp_id
        else:
            target_val = new_choices[0] if new_choices else None
            
        # 更新当前选中的实验
        self.current_exp_id = target_val
        
        return (
            gr.update(choices=new_choices, value=target_val), 
            gr.update(choices=new_choices, value=target_val)
        )

    def run_pipeline_thread(self, config_str):
        try:
            config_data = yaml.safe_load(config_str)
            exp_id = config_data.get("name", "exp")
            
            tmp_path = self.config_dir / f"last_run_{exp_id}.yaml"
            with open(tmp_path, "w") as f:
                yaml.dump(config_data, f)

            parser.add_argument("--config", default="configs/default_exp.yaml")

            parser.add_argument("--stage", default="all", help="all, filter, train, eval")
            parser.add_argument("--resume_id", help="previous experiment ID for connecting the subsequent steps.")
            parser.add_argument("--latest_run_path", default="./.latest_run")
            platform = ResearchPlatform(str(tmp_path))
            thread = threading.Thread(target=platform.start, daemon=True)
            thread.start()
            
            self.current_exp_id = platform.exp_manager.exp_id
            
            return platform.exp_manager.exp_id
        except Exception as e:
            raise gr.Error(f"Failed to start: {str(e)}")

    def read_logs(self, exp_id, log_type):
        """读取日志文件"""
        if not exp_id:
            return "⚠️ Please select an experiment."
        
        log_file = self.output_dir / exp_id / f"{log_type}.log"
        if not log_file.exists():
            return f"🔎 Log file not found at: {log_file}\n(Stage may not have started.)"
        
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return "".join(lines[-100:])
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def update_visuals_wrapper(self, exp_id):
        if not exp_id: 
            return None, None, None
        exp_path = self.output_dir / exp_id
        fig = self.visualizer.parse_train_loss(exp_path / "train.log")
        score_df = self.visualizer.load_eval_results(exp_path)
        preview_df = self.visualizer.preview_jsonl(exp_path / "final_filtered_file.jsonl")
        return score_df, preview_df

    def launch(self):
        with gr.Blocks(css=CSS, title="Data-Centric Platform", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🚀 LLM Data-Centric Research Platform")
            
            with gr.Tabs() as tabs:
                # --- Tab 1: Configuration ---
                with gr.TabItem("🎯 Launch Experiment", id="tab_launch"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 🛠️ Quick Settings")
                            server_ds_ui = gr.Dropdown(
                                label="Server Dataset", 
                                choices=list(DATASET_MAP.keys()), 
                                value="None"
                            )
                            ds_ui = gr.File(label="Upload Dataset", file_types=[".jsonl"])
                            model_ui = gr.Dropdown(
                                label="Model", 
                                choices=self.get_model_list(), 
                                allow_custom_value=True
                            )
                            metric_ui = gr.CheckboxGroup(
                                label="Tasks", 
                                choices=["mmlu", "gsm8k", "arc", "ceval"], 
                                value=["mmlu", "gsm8k"]
                            )
                            all_baselines = self.get_baseline_choices()
                            baseline_ui = gr.Dropdown(
                                label="Select Baselines (Category/Method)",
                                choices=all_baselines,
                                value=["bench/dclm"], # 默认值
                                multiselect=True,
                                info="Path: baseline/{category}/{method}"
                            )
                            stage_ui = gr.Radio(
                                label="Execution Stage",
                                choices=["all", "filter", "train", "eval"],
                                value="all",
                                info="Select which stage to run"
                            )
                            
                            resume_id_ui = gr.Textbox(
                                label="Resume ID (Optional)",
                                placeholder="e.g., exp_0324_1430",
                                info="Previous experiment ID to resume from"
                            )
                            apply_btn = gr.Button("Inject to YAML ⬇️")
                            
                        with gr.Column(scale=2):
                            config_input = gr.Code(
                                label="Configuration (YAML)",
                                language="yaml",
                                lines=18,
                                value=yaml.dump(self._get_flexible_config(), sort_keys=False)
                            )
                            run_btn = gr.Button("🚀 Start Experiment", variant="primary")
                        
                        with gr.Column(scale=1):
                            status_box = gr.Textbox(label="Status", interactive=False)
                            refresh_list_btn = gr.Button("🔄 Refresh Experiment List")

                # --- Tab 2: Logs ---
                with gr.TabItem("📜 Logs", id="tab_logs"):
                    with gr.Row():
                        exp_selector_log = gr.Dropdown(
                            label="Experiment", 
                            choices=self.get_exp_list(), 
                            allow_custom_value=True,
                            interactive=True
                        )
                        log_type = gr.Radio(["filter", "train", "eval"], label="Stage", value="filter")
                        refresh_logs_btn = gr.Button("Manual Refresh")
                        auto_refresh_checkbox = gr.Checkbox(label="Auto Refresh Logs (2s)", value=True)
                    log_display = gr.Code(label="Terminal Output", lines=20)

                # --- Tab 3: Analytics ---
                with gr.TabItem("📊 Analytics", id="tab_ana"):
                    exp_selector_ana = gr.Dropdown(
                        label="Select Experiment", 
                        choices=self.get_exp_list(), 
                        allow_custom_value=True,
                        interactive=True
                    )
                    with gr.Row():
                        eval_df = gr.DataFrame(label="Scores")
                    data_preview = gr.DataFrame(label="Data Sample")

            def sync_experiment_selection(exp_id):
                self.current_exp_id = exp_id
                return exp_id, exp_id
            
            exp_selector_log.change(
                sync_experiment_selection, 
                [exp_selector_log], 
                [exp_selector_log, exp_selector_ana]
            )
            
            exp_selector_ana.change(
                sync_experiment_selection, 
                [exp_selector_ana], 
                [exp_selector_ana, exp_selector_log]
            )

            auto_refresh_js = """
            function() {
                setInterval(function() {
                    const refreshBtns = document.querySelectorAll('button');
                    for (let btn of refreshBtns) {
                        if (btn.innerText.includes('Refresh Experiment List')) {
                            btn.click();
                            break;
                        }
                    }
                }, 5000);
                console.log('Auto refresh started - will refresh experiment list every 5 seconds');
            }
            """
            
            demo.load(
                fn=None,
                inputs=None,
                outputs=None,
                js=auto_refresh_js
            )
            
            refresh_list_btn.click(
                lambda: self.refresh_all_experiments(self.current_exp_id), 
                None, 
                [exp_selector_log, exp_selector_ana]
            )

            def auto_refresh_logs(exp_id, log_type, auto_refresh):
                if auto_refresh and exp_id:
                    return self.read_logs(exp_id, log_type)
                return gr.update()
            
            auto_refresh_logs_js = """
            function() {
                let autoRefresh = true;
                const checkbox = document.querySelector('#auto_refresh_checkbox input');
                if (checkbox) {
                    autoRefresh = checkbox.checked;
                }
                if (autoRefresh) {
                    const refreshBtn = document.querySelector('#refresh_logs_btn');
                    if (refreshBtn) {
                        refreshBtn.click();
                    }
                }
            }
            """
            
            demo.load(
                fn=None,
                inputs=None,
                outputs=None,
                js="""
                function() {
                    setInterval(function() {
                        const checkbox = document.querySelector('#auto_refresh_checkbox input');
                        if (checkbox && checkbox.checked) {
                            const refreshBtn = document.querySelector('#refresh_logs_btn');
                            if (refreshBtn) {
                                refreshBtn.click();
                            }
                        }
                    }, 2000);
                }
                """
            )
            
            refresh_logs_btn.click(
                self.read_logs, 
                [exp_selector_log, log_type], 
                log_display
            )

            run_btn.click(
                fn=self.run_pipeline_thread,
                inputs=[config_input, stage_ui, resume_id_ui],
                outputs=[exp_selector_log]
            ).then(
                fn=lambda name: f"🚀 Running: {name}",
                inputs=[exp_selector_log],
                outputs=[status_box]
            ).then(
                fn=lambda exp_id: self.refresh_all_experiments(exp_id),
                inputs=[exp_selector_log],
                outputs=[exp_selector_log, exp_selector_ana]
            )

            exp_selector_ana.change(
                self.update_visuals_wrapper, 
                [exp_selector_ana], 
                [eval_df, data_preview]
            )
            
            apply_btn.click(
                self.update_yaml_logic, 
                [baseline_ui, server_ds_ui, ds_ui, model_ui, metric_ui, config_input], 
                [config_input]
            )
            
            demo.load(
                fn=lambda: self.refresh_all_experiments(None),
                inputs=[],
                outputs=[exp_selector_log, exp_selector_ana]
            )

        demo.queue(default_concurrency_limit=5).launch(
            server_name="0.0.0.0", 
            server_port=7860,
            debug=False
        )


if __name__ == "__main__":
    GradioApp().launch()