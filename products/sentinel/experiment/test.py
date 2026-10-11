import sys
import threading
import time

def check_gil_status():
    """Verifies whether the GIL is active or disabled in this runtime."""
    if hasattr(sys, "_is_gil_enabled"):
        gil_enabled = sys._is_gil_enabled()
        status = "DISABLED (TRUE MULTITHREADING!)" if not gil_enabled else "ENABLED (Threads run sequentially)"
    else:
        status = "ENABLED (Pre-Python 3.13 environment)"
    
    print("=" * 40)
    print(f"Python Version: {sys.version.split()[:2]}")
    print(f"GIL Status:     {status}")
    print("=" * 40 + "\n")

# Shared storage and state control variables
shared_records = {}
records_lock = threading.Lock()

def heavy_cpu_task(thread_id, use_lock=True):
    """Simulates an enterprise agent workflow processing resource lookups."""
    global shared_records
    
    # Each thread performs 10,000 parallel updates to simulate heavy state mutations
    for step in range(10_000):
        # Unique identifier representing an active worker state row key lookup
        record_key = f"agent_state_{step}"
        
        if use_lock:
            with records_lock:
                # STEP A: Read structural state information
                current_state = shared_records.get(record_key, 0)
                # STEP B: Perform data operation processing window
                new_state = current_state + 1
                # STEP C: Commit resource footprint change back to the dictionary layer
                shared_records[record_key] = new_state
        else:
            # --- THE FLUID COLLISION INTERACTION LAYER ---
            # Even though dict.get() and dict[]= are individually thread-safe, 
            # the gap *between* reading and writing is completely unprotected!
            current_state = shared_records.get(record_key, 0)
            new_state = current_state + 1
            shared_records[record_key] = new_state

def run_experiment(num_threads=4, use_lock=True):
    """Launches parallel execution and calculates metric drops from concurrency overlap."""
    global shared_records
    shared_records = {}  # Clear shared instance reference map
    threads = []
    
    # If threads cooperate perfectly, every key step from 0-9999 should be incremented exactly 4 times.
    # Total sum of all values inside the records tracker dictionary must equal 40,000.
    expected_total_weight = num_threads * 10_000
    
    print(f"🚀 Spawning {num_threads} threads (Simultaneous state Read -> Modify -> Write checks)...")
    start_time = time.perf_counter()
    
    for i in range(num_threads):
        t = threading.Thread(target=heavy_cpu_task, args=(i, use_lock))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Sum up the state values actually recorded inside our shared memory space database
    actual_total_weight = sum(shared_records.values())
    mismatch = "❌ CORRUPTED!" if actual_total_weight != expected_total_weight else "✅ PERFECT Match"
    
    print(f"⏱️ Finished in: {total_time:.4f} seconds")
    print(f"📊 State Operations: Expected {expected_total_weight:,} | Got {actual_total_weight:,} -> {mismatch}\n")

if __name__ == "__main__":
    check_gil_status()
    
    print("--- RUN 1: Thread-Safe Processing ---")
    run_experiment(num_threads=4, use_lock=True)

    print("--- RUN 2: Dangerous Run (No Locks) ---")
    run_experiment(num_threads=4, use_lock=False)