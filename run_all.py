import os
import argparse
import subprocess
from pathlib import Path

def run_step(step_num):
    steps = {
        1: "01_langsmith_rag_pipeline.py",
        2: "02_prompt_hub_ab_routing.py",
        3: "03_ragas_evaluation.py",
        4: "04_guardrails_validator.py"
    }
    
    if step_num not in steps:
        print(f"❌ Invalid step: {step_num}")
        return

    script = steps[step_num]
    print(f"\n\n{'='*60}")
    print(f"  RUNNING STEP {step_num}: {script}")
    print(f"{'='*60}\n")
    
    try:
        if step_num == 2:
            # Special handling for Step 2 to save evidence
            with open("evidence/02_ab_routing_log.txt", "w") as f:
                subprocess.run(["python", script], check=True, stdout=f, stderr=subprocess.STDOUT)
            print(f"✅ Step 2 complete. Log saved to evidence/02_ab_routing_log.txt")
        elif step_num == 4:
            # Step 4: Run and split logs as per rubric
            print("Running PII Demo...")
            result = subprocess.run(["python", script, "--demo", "pii"], 
                                    capture_output=True, text=True, check=True)
            Path("evidence/04_pii_demo_log.txt").write_text(result.stdout)
            
            print("Running JSON Demo...")
            result = subprocess.run(["python", script, "--demo", "json"], 
                                    capture_output=True, text=True, check=True)
            Path("evidence/04_json_demo_log.txt").write_text(result.stdout)
            print(f"✅ Step 4 complete. Logs saved to evidence/04_pii_demo_log.txt and 04_json_demo_log.txt")
        else:
            subprocess.run(["python", script], check=True)
            print(f"✅ Step {step_num} complete.")
    except Exception as e:
        print(f"❌ Step {step_num} failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Day 22 Lab Steps")
    parser.add_argument("--step", type=int, help="Run a specific step (1-4)")
    args = parser.parse_args()

    # Ensure directories exist
    os.makedirs("evidence", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    if args.step:
        run_step(args.step)
    else:
        print("🚀 Running all lab steps sequentially...")
        for i in range(1, 5):
            run_step(i)
        
        # Final evidence copy for RAGAS
        if os.path.exists("data/ragas_report.json"):
            subprocess.run(["cp", "data/ragas_report.json", "evidence/03_ragas_report.json"])
            print("✅ Copied RAGAS report to evidence/")

if __name__ == "__main__":
    main()
