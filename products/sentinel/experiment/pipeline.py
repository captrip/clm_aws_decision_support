import sys
import threading
import time
import queue
import random

# ==========================================
# 1. Thread-Safe Global State Configuration
# ==========================================
shared_workspace = {
    "current_multiplier": 1.0, 
    "system_instructions": "PROCEED WITH BASELINE WORKLOAD",
    "max_allowed_value": 75.0   # Dynamic ceiling threshold managed by Stage 3
}
workspace_lock = threading.Lock() 

# Execution Control Flags
stop_worker_flag = threading.Event() 

# Communication Queues
raw_task_queue = queue.Queue(maxsize=10)   
eval_result_queue = queue.Queue(maxsize=10) 

# ==========================================
# 2. Pipeline Stage Implementations
# ==========================================

def stage_1_worker():
    """Stage 1: Generates growing tasks, reads dynamic guidance, stops on failure."""
    print("🤖 Stage 1 (Worker) Started.")
    
    # Generate up to 10 sequential iterations
    for iteration in range(1, 11):
        if stop_worker_flag.is_set():
            print("🛑 Stage 1: Critical stop flag detected! Aborting generation loop immediately.")
            break
            
        # Read the latest configuration updates from Stage 3
        with workspace_lock:
            multiplier = shared_workspace["current_multiplier"]
            instructions = shared_workspace["system_instructions"]
            
        print(f"\n📥 Stage 1 [Task #{iteration}]: Active Instructions -> '{instructions}'")
        
        # Calculate a dynamic base value that changes each iteration
        base_value = iteration * 10 
        final_processed_value = base_value * multiplier
        
        print(f"⚙️ Stage 1 [Task #{iteration}]: Calculated payload value: {final_processed_value}")
        
        task_data = {"id": iteration, "raw_value": final_processed_value}
        raw_task_queue.put(task_data)
        
        # Pacing window to let other processing threads calculate state
        time.sleep(0.15)
        
    raw_task_queue.put(None)
    print("🏁 Stage 1: Finished generating work batches.")


def stage_2_evaluator():
    """Stage 2: Evaluates data payload against dynamic ceiling parameters."""
    print("🔍 Stage 2 (Evaluator) Started.")
    
    while True:
        task = raw_task_queue.get()
        if task is None: 
            break
            
        raw_val = task["raw_value"]
        
        # Read the safety ceiling threshold from the workspace
        with workspace_lock:
            ceiling = shared_workspace["max_allowed_value"]
            
        # RULE A: Fail if the value scales past the maximum safety threshold
        # RULE B: Fail if the number is an exact multiple of 13 (simulated error pattern)
        eval_passed = (raw_val <= ceiling) and (raw_val % 13 != 0)
        
        if not eval_passed:
            print(f"⚠️ Stage 2: CRITICAL FAILURE on Task #{task['id']} (Value {raw_val} violated safety bounds!)")
            stop_worker_flag.set() 
            
        result_data = {
            "id": task["id"],
            "raw_value": raw_val,
            "status": "PASSED" if eval_passed else "FAILED"
        }
        eval_result_queue.put(result_data)
        raw_task_queue.task_done()
        
    eval_result_queue.put(None) 
    print("🏁 Stage 2: Finished evaluation batch loop.")


def stage_3_director():
    """Stage 3: Analyzes progress trends and modifies workspace behaviors in real-time."""
    print("👑 Stage 3 (Director) Started.")
    
    while True:
        result = eval_result_queue.get()
        if result is None: 
            break
            
        print(f"📊 Stage 3: Inspecting Result #{result['id']} -> Status: {result['status']}")
        
        # INTERVENTION 1: If value touches exactly 20, boost the processing scale multiplier
        if result["status"] == "PASSED" and result["raw_value"] == 20.0:
            print("💡 Stage 3: Milestone 20 reached. Escalating multiplier capacity to 2.5x.")
            with workspace_lock:
                shared_workspace["current_multiplier"] = 2.5
                shared_workspace["system_instructions"] = "CAPACITY BOOST INJECTED - SCALE UP OUTPUT"
                
        # INTERVENTION 2: If value scales high but passes, proactively tighten safety parameters
        elif result["status"] == "PASSED" and result["raw_value"] == 75.0:
            print("💡 Stage 3: High value workload detected. Tightening max safety ceiling down to 60.0.")
            with workspace_lock:
                shared_workspace["max_allowed_value"] = 60.0
                shared_workspace["system_instructions"] = "THROTTLING RISK CONSTRAINTS - LOWER MAXIMUM CEILING"
                
        eval_result_queue.task_done()
        
    print("🏁 Stage 3: Finished director tracking monitoring loops.")

# ==========================================
# 3. Main Orchestrator Execution Environment
# ==========================================
if __name__ == "__main__":
    print("========================================")
    print("LAUNCHING MULTI-STAGE RE-WORK PIPELINE")
    print("========================================\n")
    
    t1 = threading.Thread(target=stage_1_worker, name="Worker-Thread")
    t2 = threading.Thread(target=stage_2_evaluator, name="Evaluator-Thread")
    t3 = threading.Thread(target=stage_3_director, name="Director-Thread")
    
    t1.start()
    t2.start()
    t3.start()
    
    t1.join()
    t2.join()
    t3.join()
    
    print("\n========================================")
    print("PIPELINE PROCESSING COMPLETE")
    print("========================================")
